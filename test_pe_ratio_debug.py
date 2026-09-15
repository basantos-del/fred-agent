# test_pe_ratio_debug.py
import requests
from tools import finnhub_key, get_key_ratios, get_stock_price

TICKERS = ["PANW", "CRWD"]

for ticker in TICKERS:
    print(f"\n{'=' * 20} {ticker} {'=' * 20}")

    price = get_stock_price(ticker)
    print(f"Current price (get_stock_price): {price}")

    parsed = get_key_ratios(ticker)
    print(f"Parsed get_key_ratios() output: {parsed}")

    # Raw Finnhub /stock/metric response, unfiltered — to see every field
    # Finnhub actually returns, not just the subset get_key_ratios() picks out.
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": ticker, "metric": "all", "token": finnhub_key}
    response = requests.get(url, params=params)
    raw_metrics = response.json().get("metric", {})

    print("\nAll raw Finnhub metric fields containing 'pe' or 'eps' (case-insensitive):")
    for key, value in raw_metrics.items():
        if "pe" in key.lower() or "eps" in key.lower():
            print(f"  {key}: {value}")

    stated_pe = parsed.get("pe_ratio")
    print(f"\nget_key_ratios() pe_ratio: {stated_pe}")
    for key, value in raw_metrics.items():
        if "eps" in key.lower() and isinstance(value, (int, float)) and value != 0 and price is not None:
            manual_pe = price / value
            print(f"  Manual P/E using price / {key} ({value}): {manual_pe:.2f}")
