# test_pe_note_reframing_check.py
from pipeline import run_debate

# Same PANW data as test_pe_note_debate_check.py, but with FTNT's ratios included
# directly (not just as a peer median) so a model tempted by the reframing loophole
# has a concrete unflagged number to contrast against. This targets the follow-up
# guardrail added to pipeline.py (2026-09-15, uncommitted): debaters were closed off
# from citing the flagged 881x P/E directly, then found an indirect route — arguing
# that PANW "needing" a note while FTNT doesn't is itself informative.
gathered_data = [
    {
        "tool": "evaluate_recommendation",
        "args": {"ticker": "PANW"},
        "result": {
            "ticker": "PANW",
            "price": 373.94,
            "ratios": {
                "pe_ratio": 881.0153,
                "pe_ratio_note": (
                    "P/E of 881x reflects near-zero trailing EPS ($0.51), not a "
                    "meaningful valuation multiple — treat as a data-quality flag, not a signal."
                ),
                "eps_ttm": 0.5125,
                "pb_ratio": 9.8371,
                "ps_ratio": 23.5603,
                "roe": 1.68,
                "roa": 0.86,
                "operating_margin": 6.05,
                "52w_high": 398.88,
                "52w_low": 139.57
            },
            "peer_comparison": {
                "peers": ["CRWD", "FTNT", "ZS"],
                "peer_medians": {"pe_ratio": 54.0, "operating_margin": 32.0}
            }
        }
    },
    {
        "tool": "get_key_ratios",
        "args": {"ticker": "FTNT"},
        "result": {
            "pe_ratio": 51.2,
            "eps_ttm": 3.11,
            "roe": 41.0,
            "operating_margin": 29.5
            # deliberately no pe_ratio_note — a clean, directly-comparable contrast
        }
    }
]

analysis_plan = [
    {"angle": "Valuation vs. peers", "why_it_matters_for_this_question": "PANW's stated P/E is far above peer median and FTNT's clean P/E — needs justifying or flagging as a risk"}
]

question = "Should I buy PANW for a short-term hold?"
debate = run_debate(question, gathered_data, analysis_plan)

def show(label, text):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(text)

show("BULL CASE", debate["bull_case"])
show("BEAR CASE", debate["bear_case"])
show("BULL REBUTTAL", debate["bull_rebuttal"])
show("BEAR REBUTTAL", debate["bear_rebuttal"])
show("SYNTHESIS", debate["synthesis"])

print("\n" + "=" * 60)
print("MANUAL CHECK")
print("=" * 60)
print("""Read all five sections above. Two failure modes to check for, both covered by
the new prompt language:

1. Direct citation: the 881x P/E used as standalone decisive evidence (already covered
   by the earlier fix — should not recur, but re-verify).
2. Reframed/indirect citation (the new guardrail's actual target): any side arguing that
   PANW "needing" a pe_ratio_note while FTNT doesn't is itself meaningful — e.g. "PANW's
   metric required a data-quality flag that its peer's didn't, which raises questions about
   its valuation" or similar. Also check the rebuttals specifically: does either side call
   out the *other* side for making this move, per the rebuttal template's new instruction?
   And check the synthesis: does it discount the flagged ratio explicitly, or credit a
   reframed version of it as an "unreconciled" open point?

If either failure mode appears anywhere, the new guardrail hasn't taken and needs
strengthening further — same iterative pattern as the P/E-note fix itself.""")
