# test_pe_ratio_note_debug.py
from tools import get_key_ratios

# Regression check for the P/E interpretability fix: extreme P/E values driven by
# near-zero trailing EPS (PANW ~881x, CRWD ~3621x, both confirmed via raw Finnhub
# data on 2026-09-15) should now come with a pe_ratio_note; a normal P/E name
# should not.
for ticker in ["PANW", "CRWD", "AAPL", "MSFT"]:
    ratios = get_key_ratios(ticker)
    pe = ratios.get("pe_ratio")
    eps = ratios.get("eps_ttm")
    note = ratios.get("pe_ratio_note")
    print(f"{ticker}: pe_ratio={pe}, eps_ttm={eps}, note={note!r}")

    if pe is not None and pe > 100:
        assert note is not None, f"FAIL: {ticker} has pe_ratio={pe} (>100) but no pe_ratio_note was set"
    elif pe is not None:
        assert note is None, f"FAIL: {ticker} has a normal pe_ratio={pe} but got a note anyway: {note!r}"

print("\nPASS: pe_ratio_note fires for extreme P/E values and stays quiet for normal ones.")
