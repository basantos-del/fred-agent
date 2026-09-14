# test_extract_claims_debug.py
from pipeline import extract_claims

draft = """Microsoft trades at a P/E of 32.1x versus a peer median of 28.4x, a premium justified
by superior profitability: ROE of 45.3% against a peer median of 38.1%. At current allocation,
this position would represent roughly 12% of the total portfolio."""

claims = extract_claims(draft)
print("Number of claims extracted:", len(claims))
for c in claims:
    print(" ->", c)

assert len(claims) > 0, "Expected at least one claim extracted from a draft full of specific numbers"
print("\nPASS: extraction returned at least one claim")

# --- Stress test: a draft with 15+ numeric claims, closer to a real dense Fred answer ---
dense_draft = """
MSFT trades at a P/E of 27.5x vs peer median 82.1x (66% discount), P/B of 6.3x vs
peer median 9.8x (36% discount), P/S of 11.1x vs peer median 16.8x (34% discount).
ROE is 33.2% vs peer median 13.8% (+141%). ROA is 19.4% vs peer median 6.4% (+203%).
Operating margin is 46.7% vs peer median 11.4% (+310%). Gross margin is 67.9% vs
peer median 74.8% (a 9% laggard). Dividend yield is 0.79% vs peer median 1.38%.
Current ratio is 1.23 vs peer median 1.17. Debt-to-equity is 0.24 vs peer median 0.32.
Current price is $500.32, 90% of the way to the 52-week high of $553.72 (52-week low
$349.20). Portfolio total is €22,940.00. Indirect Horizon Growth Fund exposure is €610.00.
Proposed direct position of €1,200 brings combined exposure to €1,810.00, or 7.8% of
the portfolio. Largest single-name position is Acme Robotics at 9.3%. Magnificent 7
exposure jumps from 8.5% to 13.3%. US Equities allocation climbs from 52% to 57%.
"""
dense_claims = extract_claims(dense_draft)
print(f"\nDense draft: extracted {len(dense_claims)} claims")
for c in dense_claims:
    print(" ->", c)

assert len(dense_claims) >= 15, f"Expected 15+ claims, got {len(dense_claims)} — max_tokens may still be too low"
print("\nPASS: dense draft extraction did not truncate")
