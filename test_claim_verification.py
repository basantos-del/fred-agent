# test_claim_verification.py
from pipeline import extract_claims, verify_claims

# Fake gathered_data standing in for a real Researcher run — real ratios present.
gathered_data = [
    {
        "tool": "get_key_ratios",
        "args": {"ticker": "MSFT"},
        "result": {"pe_ratio": 32.1, "roe": 45.3, "gross_margin": 69.8}
    }
]

def run_case(label, draft):
    print(f"\n=== {label} ===")
    print("Draft:", draft)
    claims = extract_claims(draft)
    print("Extracted claims:", claims)
    checked = verify_claims(claims, gathered_data)
    for c in checked:
        print(" ->", c)
    return checked

# Case 1: legitimate claims, values match the data exactly — expect "verified"
run_case("Legitimate claims", "MSFT trades at a P/E ratio of 32.1x with an ROE of 45.3%, reflecting strong profitability.")

# Case 2: a fabricated ROE, same shape as the real CrowdStrike bug — expect "unverified"
checked_bad = run_case("Fabricated claim", "MSFT trades at a P/E ratio of 32.1x with an ROE of 166%, an exceptional figure.")
assert any(c["status"] == "unverified" for c in checked_bad), "FAIL: fabricated ROE was not flagged"
print("\nPASS: fabricated claim correctly flagged unverified")

# Case 3: a derived number (portfolio %) — should NOT be falsely flagged
checked_derived = run_case("Derived claim", "At current allocation, this position represents 12% of the total portfolio value.")
assert not any(c["status"] == "unverified" for c in checked_derived), "FAIL: derived claim was falsely flagged unverified"
print("\nPASS: derived claim not falsely flagged")
