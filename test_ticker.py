from tools import ticker_exists, get_filing_context

print("AAPL exists:", ticker_exists("AAPL"))
print("XYZQ exists:", ticker_exists("XYZQ"))
print()
print(get_filing_context("XYZQ", "what are the risks"))
