# test_approver_debug.py
from pipeline import run_approver

gathered_data = [
    {"tool": "get_key_ratios", "args": {"ticker": "MSFT"}, "result": {
        "pe_ratio": 27.5166, "roe": 33.22, "gross_margin": 67.94
    }},
    {"tool": "get_peer_average_ratios", "args": {"ticker": "MSFT"}, "result": {
        "peer_medians": {"pe_ratio": 82.06, "roe": 13.77, "gross_margin": 74.79}
    }}
]

# One deliberately WRONG number (real P/E is 27.5x, this says 45x) — confirms the holistic
# Groq check produces an actual verdict rather than failing open again.
draft = """MSFT trades at a P/E of 45x versus a peer median of 82.1x, and an ROE of 33.2%
versus a peer median of 13.8%, reflecting strong relative profitability."""

verdict = run_approver(gathered_data, draft)
print("Approved:", verdict.get("approved"))
print("Issues:", verdict.get("issues"))
print("Parse error present:", "parse_error" in verdict)
print("Claim check:", verdict.get("claim_check"))

assert "parse_error" not in verdict, "Approver still failed open — reasoning_effort fix did not resolve it"
print("\nPASS: Approver's Groq call returned real, parseable content")
