import os
import json
import requests
import gspread #connection to portfolio
import voyageai #embedding_model
import chromadb #vector_db
import time

from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from voyageai.error import RateLimitError
from groq import Groq
from groq import RateLimitError

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

    return {
        "pe_ratio": metrics.get("peBasicExclExtraTTM"),
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
            print("Rate limited, waiting 60s...")
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

def store_filing_chunks(ticker, chunks, embeddings):
    ids = [f"{ticker}_{i}" for i in range(len(chunks))]
    metadatas = [{"ticker": ticker} for _ in chunks]

    filings_collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_filings(ticker, query, n_results=3):
    query_embedding = _embed_with_retry([query], input_type="query")
    results = filings_collection.query(
        query_embeddings=query_embedding.embeddings,
        n_results=n_results,
        where={"ticker": ticker}
    )
    return results["documents"][0]

def get_filing_context(ticker, question):
    existing = filings_collection.get(where={"ticker": ticker}, limit=1)

    if not existing["ids"]:
        print(f"No filing stored for {ticker} yet — ingesting now, this will take a few minutes...")
        cik = get_cik(ticker)
        filing_url = get_latest_10k_url(cik)
        text = fetch_filing_text(filing_url)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_filing_chunks(ticker, chunks, embeddings)

    return query_filings(ticker, question)

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
    return response.json()[:10]


news_collection = chroma_client.get_or_create_collection(name="news")


def ingest_news(ticker, days_back=7):
    articles = fetch_raw_news(ticker, days_back=days_back)

    if not articles:
        print(f"No news articles found for {ticker} in the last {days_back} days.")
        return

    ids, documents, metadatas = [], [], []
    for i, a in enumerate(articles):
        headline = a["headline"]
        summary = a.get("summary", "")
        sentiment = get_sentiment(headline, summary)

        documents.append(f"{headline}. {summary}. {sentiment}")
        ids.append(f"{ticker}_news_{i}")
        metadatas.append({
            "ticker": ticker,
            "date": datetime.fromtimestamp(a["datetime"]).strftime("%Y-%m-%d"),
            "url": a["url"],
            "source": a["source"]
        })

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
    all_ratios = [get_key_ratios(p) for p in peers]

    numeric_keys = [
        "pe_ratio", "pb_ratio", "ps_ratio", "roe", "roa",
        "debt_to_equity", "current_ratio", "gross_margin",
        "operating_margin", "dividend_yield", "beta"
    ]

    averages = {}
    for key in numeric_keys:
        values = [r[key] for r in all_ratios if r.get(key) is not None]
        averages[key] = round(sum(values) / len(values), 2) if values else None

    return {"peers": peers, "peer_averages": averages}

def get_portfolio_context():
    holdings = get_portfolio()
    total = sum(h["value_eur"] for h in holdings)

    by_exposure = {}
    for h in holdings:
        by_exposure[h["exposure_category"]] = by_exposure.get(h["exposure_category"], 0) + h["value_eur"]

    allocation = {k: round(v / total * 100, 1) for k, v in by_exposure.items()}
    return {"total_value_eur": total, "allocation_by_exposure": allocation, "holdings": holdings}

def evaluate_recommendation(ticker):
    return {
        "ticker": ticker,
        "ratios": get_key_ratios(ticker),
        "peer_comparison": get_peer_average_ratios(ticker),
        "portfolio_context": get_portfolio_context()
    }

def call_groq_with_retry(client, **kwargs):
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            print(f"Groq rate limited, waiting 30s... ({e})")
            time.sleep(30)

tool_functions = {
    "get_stock_price": get_stock_price,
    "get_market_cap": get_market_cap,
    "get_key_ratios": get_key_ratios,
    "get_portfolio": get_portfolio,
    "get_company_news": get_company_news,
    "get_filing_context": get_filing_context,
    "get_news_context": get_news_context,
    "evaluate_recommendation": evaluate_recommendation,
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
        "description": "Get everything needed to evaluate an investment recommendation for a ticker: its own valuation/profitability/leverage/yield/risk ratios, the same ratios averaged across its closest peers for sector-relative comparison, and Bernardo's current portfolio context (allocation by asset class and full holdings list) for diversification awareness",
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
            "description": "Get key valuation, profitability, and leverage ratios for a company: P/E, P/B, P/S, ROE, ROA, debt-to-equity, current ratio, gross margin, operating margin, beta, and 52-week high/low",
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
