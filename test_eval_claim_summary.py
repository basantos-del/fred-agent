# test_eval_claim_summary.py
from eval import run_single_eval_case, GOLDEN_SET, summarize_claim_check

# Pick an end_to_end case that actually routes through the Advisor/Approver
# (a recommendation-style question, not a simple lookup)
case = next(c for c in GOLDEN_SET if c["id"] == "recommendation_structure")

result = run_single_eval_case(case)
print("Score:", result["score"])
print("Reasoning:", result["reasoning"])
print("Claim summary:", repr(result["claim_summary"]))

assert result["claim_summary"] != "", "Expected a non-empty claim_summary for a complex-route case"
print("\nPASS: claim_summary populated for an end_to_end complex-route case")

# Sanity-check the summarizer directly against a hand-built verdict
fake_verdict = {"claim_check": [
    {"status": "verified"}, {"status": "verified"}, {"status": "unverified"}, {"status": "derived_not_checked"}
]}
summary = summarize_claim_check(fake_verdict)
print("Hand-built verdict summary:", summary)
assert summary == "2/3 verified (1 derived)", f"Unexpected format: {summary}"
print("PASS: summarizer formats as expected")
