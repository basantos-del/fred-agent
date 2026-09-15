import os
import json
import requests
import gspread #connection to portfolio
import voyageai #embedding_model
import chromadb #vector_db
import time
import re
import statistics

from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from voyageai.error import RateLimitError
from groq import Groq
from groq import RateLimitError
from groq import RateLimitError, BadRequestError
from anthropic import RateLimitError as ClaudeRateLimitError, APIStatusError as ClaudeAPIStatusError, APIConnectionError as ClaudeAPIConnectionError

load_dotenv()
finnhub_key = os.environ["FINNHUB_API_KEY"]

service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
gc = gspread.service_account_from_dict(service_account_info)

sheet = gc.open_by_key(os.environ["PORTFOLIO_SHEET_ID"])
worksheet = sheet.worksheet("Portfolio")

voyage_client = voyageai.Client()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
filings_collection = chroma_client.get_or_create_collection(name="filings")

SEC_HEADERS = {"User-Agent": "Bernardo Santos albasantos.bernardo@gmail.com"}
MAGNIFICENT_7 = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"}
MAGNIFICENT_7_NAMES = ["apple", "microsoft", "alphabet", "google", "amazon", "meta", "facebook", "nvidia", "tesla"]

# --- API usage/cost tracking ---

# USD per 1M tokens. Source: Anthropic pricing docs + Groq pricing page, checked 2026-09-15.
# Groq is free-tier for us right now — these numbers are SHADOW pricing (what it would
# cost at Groq's published paid rate), not actual spend.
API_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
}

_fx_cache = {"date": None, "rate": None}


def _get_cached_usd_to_eur_rate():
    """Fetches USD->EUR once per calendar day and reuses it, instead of hitting
    get_exchange_rate on every single logged API call (a complex query can trigger
    up to ~9 Claude calls, and this func runs on every one of them)."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    if _fx_cache["date"] == today_str and _fx_cache["rate"] is not None:
        return _fx_cache["rate"]
    try:
        rate = get_exchange_rate("USD", "EUR")
        _fx_cache["date"] = today_str
        _fx_cache["rate"] = rate
        return rate
    except Exception as e:
        print(f"Failed to refresh USD->EUR rate for API cost logging: {e}", flush=True)
        return _fx_cache["rate"]  # stale-but-cached, or None — never raise from here


def compute_api_cost(model, input_tokens, output_tokens):
    pricing = API_PRICING.get(model)
    if pricing is None:
        print(f"Warning: no pricing entry for model '{model}' — cost not tracked for this call.", flush=True)
        return None
    cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)


def get_api_usage_worksheet():
    return sheet.worksheet("API Usage")


def log_api_call(provider, model, input_tokens, output_tokens):
    """Best-effort logging — must never raise into the calling pipeline. A Sheets
    hiccup should degrade silently (usage just doesn't get logged that once), not
    break Fred's actual answer."""
    try:
        cost_usd = compute_api_cost(model, input_tokens, output_tokens)
        rate = _get_cached_usd_to_eur_rate()
        cost_eur = round(cost_usd * rate, 6) if (cost_usd is not None and rate is not None) else None

        ws = get_api_usage_worksheet()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([
            timestamp, provider, model, str(input_tokens), str(output_tokens),
            str(cost_usd) if cost_usd is not None else "",
            str(cost_eur) if cost_eur is not None else "",
        ])
    except Exception as e:
        print(f"Failed to log API usage: {e}", flush=True)


def _log_usage_if_present(provider, model, usage_obj, input_field, output_field):
    if usage_obj is None:
        return
    input_tokens = getattr(usage_obj, input_field, None)
    output_tokens = getattr(usage_obj, output_field, None)
    if input_tokens is None or output_tokens is None:
        return
    log_api_call(provider, model, input_tokens, output_tokens)


def get_api_usage_history():
    try:
        ws = get_api_usage_worksheet()
        values = ws.get_all_values()
        history = []
        for row in values[1:]:
            if not row or not row[0]:
                continue
            try:
                history.append({
                    "timestamp": row[0],
                    "date": row[0].split(" ")[0],
                    "provider": row[1],
                    "model": row[2],
                    "input_tokens": int(row[3]) if row[3] else 0,
                    "output_tokens": int(row[4]) if row[4] else 0,
                    "cost_usd": float(row[5]) if len(row) > 5 and row[5] else 0.0,
                    "cost_eur": float(row[6]) if len(row) > 6 and row[6] else 0.0,
                })
            except (ValueError, IndexError):
                continue
        return history
    except Exception as e:
        print(f"Failed to read API usage history: {e}", flush=True)
        return []

def get_sentiment(headline, summary):
    prompt = f"""Headline: {headline}
Summary: {summary}

Classify the sentiment of this news for the company as Positive, Negative, or Neutral,
and give one short reason (max 15 words). Respond in exactly this format:
Sentiment: <Positive/Negative/Neutral>
Reason: <short reason>"""

    response = call_groq_with_retry(
        groq_client,
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def _get_fund_value_lookup():
    holdings = get_portfolio()
    return {h["name"]: h["value_eur"] for h in holdings}

def ticker_exists(ticker):
    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker, "token": finnhub_key}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return bool(data.get("c"))
    except Exception as e:
        print(f"Ticker existence check failed for {ticker}: {e}", flush=True)
        return True

def get_fund_lookthrough_sectors(fund_values):
    ws = sheet.worksheet("Fund Look-Through")
    values = ws.get_all_values()

    sector_totals = {}
    for row in values[1:]:
        if not row or not row[0]:
            continue
        fund_name, sector, weight_str = row[0], row[1], row[2]
        fund_value = fund_values.get(fund_name)
        if fund_value is None:
            continue

        weight_pct = parse_percentage(weight_str)
        if weight_pct is None:
            continue

        contribution = fund_value * (weight_pct / 100)
        sector_totals[sector] = sector_totals.get(sector, 0) + contribution

    return sector_totals


def get_fund_lookthrough_magnificent_7(fund_values):
    ws = sheet.worksheet("Fund Holdings")
    values = ws.get_all_values()

    total = 0
    matched_holdings = []
    for row in values[1:]:
        if not row or not row[0]:
            continue
        fund_name, holding_name, weight_str = row[0], row[1], row[2]

        if not any(name in holding_name.lower() for name in MAGNIFICENT_7_NAMES):
            continue

        fund_value = fund_values.get(fund_name)
        if fund_value is None:
            continue

        weight_pct = parse_percentage(weight_str)
        if weight_pct is None:
            continue

        contribution = fund_value * (weight_pct / 100)
        total += contribution
        matched_holdings.append({"fund": fund_name, "holding": holding_name, "value_eur": round(contribution, 2)})

    return {"total_value_eur": round(total, 2), "matched_holdings": matched_holdings}

def get_industry(ticker):
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": ticker, "token": finnhub_key}
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("finnhubIndustry")

def get_stock_price(ticker):
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": ticker, "token": finnhub_key}
    response = requests.get(url, params=params)
    data = response.json()
    return data["c"]

def to_anthropic_tools(openai_tools):
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"]
        }
        for t in openai_tools
    ]

def get_market_cap(ticker):
    url = "https://finnhub.io/api/v1/stock/profile2"
    params = {"symbol": ticker, "token": finnhub_key}
    response = requests.get(url, params=params)
    data = response.json()
    return data["marketCapitalization"]

def get_key_ratios(ticker):
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": ticker, "metric": "all", "token": finnhub_key}
    response = requests.get(url, params=params)
    data = response.json()
    metrics = data.get("metric")

    if metrics is None:
        print(f"Warning: no metrics returned for {ticker}. Raw response: {data}")
        metrics = {}

    pe_ratio = metrics.get("peBasicExclExtraTTM")
    eps_ttm = metrics.get("epsBasicExclExtraItemsTTM")

    pe_ratio_note = None
    if pe_ratio is not None and pe_ratio > 100:
        eps_text = f"${eps_ttm:.2f}" if eps_ttm is not None else "near zero"
        pe_ratio_note = (
            f"P/E of {pe_ratio:.0f}x reflects near-zero trailing EPS ({eps_text}), not a "
            f"meaningful valuation multiple — treat as a data-quality flag, not a signal."
        )

    return {
        "pe_ratio": pe_ratio,
        "pe_ratio_note": pe_ratio_note,
        "eps_ttm": eps_ttm,
        "pb_ratio": metrics.get("pbQuarterly"),
        "ps_ratio": metrics.get("psAnnual"),
        "roe": metrics.get("roeTTM"),
        "roa": metrics.get("roaTTM"),
        "debt_to_equity": metrics.get("totalDebt/totalEquityQuarterly"),
        "current_ratio": metrics.get("currentRatioQuarterly"),
        "gross_margin": metrics.get("grossMarginTTM"),
        "operating_margin": metrics.get("operatingMarginTTM"),
        "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
        "beta": metrics.get("beta"),
        "52w_high": metrics.get("52WeekHigh"),
        "52w_low": metrics.get("52WeekLow"),
    }

def parse_eur_amount(value_str):
    cleaned = value_str.replace("\xa0", "").replace("€", "").replace(",", ".").strip()
    return float(cleaned)

def parse_percentage(value_str):
    cleaned = value_str.replace("%", "").replace(",", ".").strip()
    return float(cleaned) if cleaned else None

#Portfolio functions

def get_portfolio():
    values = worksheet.get_all_values()
    holdings = []
    for row in values[1:]:  # skip header row
        name, asset_class, value_str, pct_str, ticker, exposure_category, currency = row
        if not name:
            continue
        holdings.append({
            "name": name,
            "asset_class": asset_class,
            "value_eur": parse_eur_amount(value_str),
            "percentage": parse_percentage(pct_str),
            "ticker": ticker if ticker else None,
            "exposure_category": exposure_category,
            "currency": currency
        })
    return holdings

def get_history_worksheet():
    return sheet.worksheet("History")

def log_portfolio_snapshot(total_value_eur):
    ws = get_history_worksheet()
    values = ws.get_all_values()
    today_str = datetime.now().strftime("%Y-%m-%d")

    if len(values) > 1:
        last_date = values[-1][0]
        if last_date == today_str:
            return  # already logged today

    ws.append_row([today_str, str(total_value_eur)])

def get_portfolio_history():
    ws = get_history_worksheet()
    values = ws.get_all_values()

    history = []
    for row in values[1:]:
        if not row or not row[0]:
            continue
        try:
            history.append({"date": row[0], "total_value_eur": float(row[1])})
        except (ValueError, IndexError):
            continue

    return history

def get_portfolio_context():
    holdings = get_portfolio()
    total = sum(h["value_eur"] for h in holdings)
    fund_values = _get_fund_value_lookup()

    by_exposure_value = {}
    by_industry_value = {}
    direct_mag7_value = 0

    for h in holdings:
        by_exposure_value[h["exposure_category"]] = by_exposure_value.get(h["exposure_category"], 0) + h["value_eur"]

        if h["ticker"]:
            industry = get_industry(h["ticker"])
            h["industry"] = industry
            if industry:
                by_industry_value[industry] = by_industry_value.get(industry, 0) + h["value_eur"]
            if h["ticker"] in MAGNIFICENT_7:
                direct_mag7_value += h["value_eur"]

    lookthrough_sectors = get_fund_lookthrough_sectors(fund_values)
    for sector, value in lookthrough_sectors.items():
        by_industry_value[sector] = by_industry_value.get(sector, 0) + value

    lookthrough_mag7 = get_fund_lookthrough_magnificent_7(fund_values)
    true_mag7_value = direct_mag7_value + lookthrough_mag7["total_value_eur"]

    allocation = {k: round(v / total * 100, 1) for k, v in by_exposure_value.items()}
    industry_allocation = {k: round(v / total * 100, 1) for k, v in by_industry_value.items()}
    true_mag7_pct = round(true_mag7_value / total * 100, 1) if total else 0

    return {
        "total_value_eur": total,
        "allocation_by_exposure": allocation,
        "value_by_exposure": by_exposure_value,
        "allocation_by_industry": industry_allocation,
        "magnificent_7_pct": true_mag7_pct,
        "magnificent_7_lookthrough_detail": lookthrough_mag7["matched_holdings"],
        "holdings": holdings
    }

def get_company_news(ticker, days_back=7):
    end = datetime.now()
    start = end - timedelta(days=days_back)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": finnhub_key
    }
    response = requests.get(url, params=params)
    articles = response.json()

    return [
        {
            "headline": a["headline"],
            "source": a["source"],
            "date": datetime.fromtimestamp(a["datetime"]).strftime("%Y-%m-%d"),
            "url": a["url"]
        }
        for a in articles[:10]
    ]


# --- SEC EDGAR helpers: written, not yet tested/wired in as agent tools ---

def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=SEC_HEADERS)
    data = response.json()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None

def get_latest_10k_url(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS)
    data = response.json()

    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]
            print(f"Using 10-K filed on {filing_date}")
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    return None

def fetch_filing_text(url):
    response = requests.get(url, headers=SEC_HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove hidden inline-XBRL metadata — not real filing content
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def _embed_with_retry(texts, input_type):
    while True:
        try:
            return voyage_client.embed(texts, model="voyage-finance-2", input_type=input_type)
        except RateLimitError:
            print("Rate limited, waiting 60s...", flush=True)
            time.sleep(60)

def embed_chunks(chunks, batch_size=20, delay=21):
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = _embed_with_retry(batch, input_type="document")
        all_embeddings.extend(result.embeddings)

        if i + batch_size < len(chunks):
            time.sleep(delay)

    return all_embeddings

#Reranker

def _rerank_with_retry(query, documents, model="rerank-3", top_k=None):
    while True:
        try:
            return voyage_client.rerank(query, documents, model=model, top_k=top_k)
        except RateLimitError:
            print("Rerank rate limited, waiting 60s...", flush=True)
            time.sleep(60)

def query_filings(ticker, query, n_candidates=15, n_results=3):
    query_embedding = _embed_with_retry([query], input_type="query")
    results = filings_collection.query(
        query_embeddings=query_embedding.embeddings,
        n_results=n_candidates,
        where={"ticker": ticker}
    )
    candidates = results["documents"][0]

    if not candidates:
        return []

    reranked = _rerank_with_retry(query, candidates, model="rerank-3", top_k=n_results)
    # Voyage docs say results come back sorted by relevance_score desc already;
    # sorting explicitly costs nothing and removes the dependency on that being true forever.
    ranked_results = sorted(reranked.results, key=lambda r: r.relevance_score, reverse=True)

    return [
        {"text": r.document, "relevance_score": round(r.relevance_score, 4)}
        for r in ranked_results
    ]

def store_filing_chunks(ticker, chunks, embeddings):
    ids = [f"{ticker}_{i}" for i in range(len(chunks))]
    metadatas = [{"ticker": ticker} for _ in chunks]

    filings_collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def get_filing_context(ticker, question):
    existing = filings_collection.get(where={"ticker": ticker}, limit=1)

    if not existing["ids"]:
        if not ticker_exists(ticker):
            return {"error": f"No data found for ticker '{ticker}' — it may not be a valid or listed US ticker."}

        print(f"No filing stored for {ticker} yet — ingesting now, this will take a few minutes...", flush=True)
        cik = get_cik(ticker)
        if cik is None:
            return {"error": f"No SEC filings found for ticker '{ticker}' — it may not be a US-listed company."}

        filing_url = get_latest_10k_url(cik)
        if filing_url is None:
            return {"error": f"No 10-K filing found for ticker '{ticker}'."}

        text = fetch_filing_text(filing_url)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_filing_chunks(ticker, chunks, embeddings)

    passages = query_filings(ticker, question)
    if not passages:
        return {"error": f"No relevant passages found in {ticker}'s 10-K for this question."}

    # One top-level key per passage — see explanation below, this isn't cosmetic.
    return {f"passage_{i+1}": p for i, p in enumerate(passages)}

def fetch_raw_news(ticker, days_back=7):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": finnhub_key
    }
    response = requests.get(url, params=params)
    return response.json()[:4]

news_collection = chroma_client.get_or_create_collection(name="news")

def ingest_news(ticker, days_back=7):
    articles = fetch_raw_news(ticker, days_back=days_back)

    if not articles:
        print(f"No news articles found for {ticker} in the last {days_back} days.", flush=True)
        return

    existing = news_collection.get(where={"ticker": ticker})
    existing_ids = set(existing["ids"])

    ids, documents, metadatas = [], [], []
    for i, a in enumerate(articles):
        article_id = f"{ticker}_news_{i}"
        if article_id in existing_ids:
            continue

        headline = a["headline"]
        summary = a.get("summary", "")
        sentiment = get_sentiment(headline, summary)

        documents.append(f"{headline}. {summary}. {sentiment}")
        ids.append(article_id)
        metadatas.append({
            "ticker": ticker,
            "date": datetime.fromtimestamp(a["datetime"]).strftime("%Y-%m-%d"),
            "url": a["url"],
            "source": a["source"]
        })

    if not documents:
        return

    embeddings = embed_chunks(documents)
    news_collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

def query_news(ticker, query, n_results=3):
    query_embedding = _embed_with_retry([query], input_type="query")
    results = news_collection.query(
        query_embeddings=query_embedding.embeddings,
        n_results=n_results,
        where={"ticker": ticker}
    )
    return results["documents"][0]

def get_news_context(ticker, question, days_back=7):
    ingest_news(ticker, days_back=days_back)
    return query_news(ticker, question)

def get_peers(ticker):
    url = "https://finnhub.io/api/v1/stock/peers"
    params = {"symbol": ticker, "token": finnhub_key}
    response = requests.get(url, params=params)
    peers = response.json()
    return [p for p in peers if p != ticker][:5]

def get_peer_average_ratios(ticker):
    peers = get_peers(ticker)
    per_peer = {p: get_key_ratios(p) for p in peers}

    numeric_keys = [
        "pe_ratio", "pb_ratio", "ps_ratio", "roe", "roa",
        "debt_to_equity", "current_ratio", "gross_margin",
        "operating_margin", "dividend_yield", "beta"
    ]

    averages = {}
    medians = {}
    for key in numeric_keys:
        values = [r[key] for r in per_peer.values() if r.get(key) is not None]
        if values:
            averages[key] = round(sum(values) / len(values), 2)
            medians[key] = round(statistics.median(values), 2)
        else:
            averages[key] = None
            medians[key] = None

    return {
        "peers": peers,
        "peer_averages": averages,
        "peer_medians": medians,
        "per_peer_ratios": per_peer
    }

def evaluate_recommendation(ticker):
    try:
        exchange_rate = get_exchange_rate("USD", "EUR")
    except Exception as e:
        print(f"Warning: get_exchange_rate failed inside evaluate_recommendation: {e}", flush=True)
        exchange_rate = None

    try:
        price = get_stock_price(ticker)
    except Exception as e:
        print(f"Warning: get_stock_price failed inside evaluate_recommendation: {e}", flush=True)
        price = None

    return {
        "ticker": ticker,
        "price": price,
        "ratios": get_key_ratios(ticker),
        "peer_comparison": get_peer_average_ratios(ticker),
        "portfolio_context": get_portfolio_context(),
        "usd_to_eur_rate": exchange_rate
    }

def call_claude_with_retry(client, max_retries=5, **kwargs):
    """Mirrors call_groq_with_retry: wraps claude_client.messages.create() with
    backoff on rate limits (429), transient server errors (5xx), and connection
    errors, so one flaky call doesn't kill an entire multi-call debate chain."""
    attempt = 0
    while True:
        try:
            response = client.messages.create(**kwargs)
            _log_usage_if_present("Claude", kwargs.get("model"), getattr(response, "usage", None),
                                   "input_tokens", "output_tokens")
            return response
        except ClaudeRateLimitError as e:
            attempt += 1  
            if attempt > max_retries:
                raise 
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after"))
            except (AttributeError, TypeError, ValueError):
                pass
            wait_time = retry_after if retry_after else min(2 ** attempt, 60)
            print(f"Claude rate limited, waiting {wait_time:.0f}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait_time)
        except ClaudeAPIConnectionError as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait_time = min(2 ** attempt, 30)
            print(f"Claude connection error, waiting {wait_time:.0f}s (attempt {attempt}/{max_retries}): {e}", flush=True)
            time.sleep(wait_time)
        except ClaudeAPIStatusError as e:
            if e.status_code < 500:
                raise
            attempt += 1
            if attempt > max_retries:
                raise
            wait_time = min(2 ** attempt, 30)
            print(f"Claude server error {e.status_code}, waiting {wait_time:.0f}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait_time)

def call_groq_with_retry(client, max_malformed_retries=3, **kwargs):
    malformed_attempts = 0

    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            _log_usage_if_present("Groq", kwargs.get("model"), getattr(response, "usage", None),
                                   "prompt_tokens", "completion_tokens")
            return response
        except RateLimitError as e:
            wait_time = 30
            match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
            if match:
                minutes = int(match.group(1)) if match.group(1) else 0
                seconds = float(match.group(2))
                wait_time = minutes * 60 + seconds + 2
            print(f"Groq rate limited, waiting {wait_time:.0f}s... ({e})", flush=True)
            time.sleep(wait_time)
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise
            malformed_attempts += 1
            if malformed_attempts >= max_malformed_retries:
                raise
            print(f"Groq generated malformed tool call, retrying ({malformed_attempts}/{max_malformed_retries})...", flush=True)

def estimate_tokens(text):
    """Rough token-count estimate for text headed to Groq. There's no public tokenizer
    for openai/gpt-oss-20b, so this uses ~3 chars/token rather than the usual ~4 —
    the content this guards (JSON tool results) tokenizes worse than prose, and this
    is feeding a hard-cap safety check, so it's better to over-estimate than under."""
    if not text:
        return 0
    return (len(text) // 3) + 1

def get_valid_params(function_name):
    for t in tools:
        if t["function"]["name"] == function_name:
            return set(t["function"]["parameters"]["properties"].keys())
    return None


def filter_args_for_tool(function_name, args):
    valid_params = get_valid_params(function_name)
    if valid_params is None:
        return args
    return {k: v for k, v in args.items() if k in valid_params}

def get_exchange_rate(from_currency="USD", to_currency="EUR"):
    # Primary: dedicated free FX API
    try:
        url = f"https://api.frankfurter.app/latest"
        params = {"from": from_currency, "to": to_currency}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        rate = data.get("rates", {}).get(to_currency)
        if rate is not None:
            return rate
    except Exception as e:
        print(f"Frankfurter FX lookup failed: {e}", flush=True)

    # Fallback: Finnhub (may be gated depending on plan)
    try:
        url = "https://finnhub.io/api/v1/forex/rates"
        params = {"base": from_currency, "token": finnhub_key}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        quote = data.get("quote")
        if quote is not None:
            return quote.get(to_currency)
        print(f"Finnhub FX unavailable: {data}", flush=True)
    except Exception as e:
        print(f"Finnhub FX lookup failed: {e}", flush=True)

    return None

def log_conversation_message(thread_id, role, content, route=""):
    try:
        ws = sheet.worksheet("Conversations")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([timestamp, str(thread_id), role, content[:45000], route])
    except Exception as e:
        print(f"Failed to log conversation: {e}", flush=True)


def log_feedback(thread_id, question, what_was_missing):
    try:
        ws = sheet.worksheet("Feedback")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([timestamp, str(thread_id), question[:5000], what_was_missing[:5000]])
        return True
    except Exception as e:
        print(f"Failed to log feedback: {e}", flush=True)
        return False


def get_recent_conversations(limit_threads=10):
    ws = sheet.worksheet("Conversations")
    values = ws.get_all_values()

    rows = []
    for row in values[1:]:
        if not row or not row[0]:
            continue
        rows.append({
            "timestamp": row[0],
            "thread_id": row[1] if len(row) > 1 else "",
            "role": row[2] if len(row) > 2 else "",
            "content": row[3] if len(row) > 3 else "",
            "route": row[4] if len(row) > 4 else ""
        })

    thread_ids = []
    for r in reversed(rows):
        if r["thread_id"] not in thread_ids:
            thread_ids.append(r["thread_id"])
        if len(thread_ids) >= limit_threads:
            break

    return [r for r in rows if r["thread_id"] in thread_ids]

def get_all_feedback(include_addressed=False):
    ws = sheet.worksheet("Feedback")
    values = ws.get_all_values()

    feedback = []
    for i, row in enumerate(values[1:], start=2):
        if not row or not row[0]:
            continue
        addressed = (row[4].strip().lower() if len(row) > 4 else "") == "yes"
        if addressed and not include_addressed:
            continue
        feedback.append({
            "row_number": i,
            "timestamp": row[0],
            "thread_id": row[1] if len(row) > 1 else "",
            "question": row[2] if len(row) > 2 else "",
            "what_was_missing": row[3] if len(row) > 3 else "",
            "addressed": addressed
        })
    return feedback

def mark_feedback_addressed(row_numbers):
    ws = sheet.worksheet("Feedback")
    for row_number in row_numbers:
        try:
            ws.update_cell(row_number, 5, "yes")
        except Exception as e:
            print(f"Failed to mark feedback row {row_number}: {e}", flush=True)


def log_coach_adoption(summary, row_numbers):
    try:
        ws = sheet.worksheet("Coach Log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([timestamp, summary[:10000], ", ".join(str(r) for r in row_numbers)])
        return True
    except Exception as e:
        print(f"Failed to log coach adoption: {e}", flush=True)
        return False


def get_coach_log():
    try:
        ws = sheet.worksheet("Coach Log")
        values = ws.get_all_values()
        return [
            {"timestamp": row[0], "summary": row[1] if len(row) > 1 else "", "rows": row[2] if len(row) > 2 else ""}
            for row in values[1:] if row and row[0]
        ]
    except Exception as e:
        print(f"Failed to read coach log: {e}", flush=True)
        return []

tool_functions = {
    "get_stock_price": get_stock_price,
    "get_market_cap": get_market_cap,
    "get_key_ratios": get_key_ratios,
    "get_portfolio": get_portfolio,
    "get_company_news": get_company_news,
    "get_filing_context": get_filing_context,
    "get_news_context": get_news_context,
    "evaluate_recommendation": evaluate_recommendation,
    "get_exchange_rate": get_exchange_rate,
}

#tools schema

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock given its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
{
    "type": "function",
    "function": {
        "name": "evaluate_recommendation",
        "description": "Get everything needed to evaluate an investment recommendation for a ticker: its current price, its own valuation/profitability/leverage/yield/risk ratios (including 52-week high/low), the same ratios averaged across its closest peers for sector-relative comparison, and Bernardo's current portfolio context (allocation by asset class and full holdings list) for diversification awareness",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"}
            },
            "required": ["ticker"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "get_news_context",
        "description": "Search recent news about a company for relevant passages, including sentiment analysis for each article (Positive/Negative/Neutral with reasoning)",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"},
                "question": {"type": "string", "description": "What to search for within recent news"},
                "days_back": {"type": "integer", "description": "How many days back to search, defaults to 7 if not specified"}
            },
            "required": ["ticker", "question"]
        }
    }
},
    {
    "type": "function",
    "function": {
        "name": "get_filing_context",
        "description": "Search a company's most recent 10-K annual report for relevant passages, given a specific question (e.g. about risk factors, business strategy, competition, or financial condition)",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"},
                "question": {"type": "string", "description": "What to search for within the filing"}
            },
            "required": ["ticker", "question"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "Get the current exchange rate between two currencies, e.g. USD to EUR",
        "parameters": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "Currency code to convert from, e.g. USD"},
                "to_currency": {"type": "string", "description": "Currency code to convert to, e.g. EUR"}
            },
            "required": []
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_market_cap",
            "description": "Get the market capitalization of a company given its ticker symbol, in millions of USD",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_ratios",
	    "description": "Get key valuation, profitability, and leverage ratios for a company: P/E (with trailing EPS and a data-quality note when the P/E isn't economically meaningful), P/B, P/S, ROE, ROA, debt-to-equity, current ratio, gross margin, operating margin, beta, and 52-week high/low",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Get Bernardo's current investment portfolio holdings, including asset name, asset class, value in EUR, and percentage of total portfolio",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get recent news articles for a company given its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker, e.g. AAPL"},
                    "days_back": {"type": "integer", "description": "How many days back to search, defaults to 7 if not specified"}
                },
                "required": ["ticker"]
            }
        }
    },
]
