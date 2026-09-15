# test_evaluate_recommendation_price_debug.py
from tools import evaluate_recommendation

# Repro for the missing-price bug: evaluate_recommendation() previously never
# fetched price at all, despite RESEARCHER_SYSTEM_PROMPT claiming it did --
# leaving the Advisor with no real price (showed as "Unknown current" for
# CRWD/FTNT, and PANW's 52-week high got mistaken for current price in a
# real Fred answer on 2026-09-15).
for ticker in ["AAPL", "MSFT", "CRWD"]:
    result = evaluate_recommendation(ticker)
    assert "price" in result, f"FAIL: 'price' key missing from evaluate_recommendation({ticker})"
    price = result["price"]
    ratios = result["ratios"]
    print(f"{ticker}: price={price}, 52w_high={ratios.get('52w_high')}, 52w_low={ratios.get('52w_low')}")
    if price is not None:
        assert isinstance(price, (int, float)), f"FAIL: price for {ticker} is not numeric: {price!r}"
        high, low = ratios.get("52w_high"), ratios.get("52w_low")
        if high is not None and low is not None:
            assert low * 0.95 <= price <= high * 1.05, \
                f"FAIL: {ticker} price {price} is well outside its 52-week range [{low}, {high}] — possible stale/wrong data"
    else:
        print(f"  (price came back None for {ticker} — check Finnhub response/rate limit, not necessarily a code bug)")

print("\nPASS: evaluate_recommendation now returns a price field for all tested tickers (or explicit None, never silently missing).")
