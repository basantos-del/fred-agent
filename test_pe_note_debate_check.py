# test_pe_note_debate_check.py
from pipeline import run_debate

# Real PANW data from the 2026-09-15 P/E investigation (881x P/E driven by
# near-zero trailing EPS, now flagged via pe_ratio_note in get_key_ratios()).
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
    }
]

analysis_plan = [
    {"angle": "Valuation vs. peers", "why_it_matters_for_this_question": "P/E is far above peer median — needs justifying or flagging as a risk"}
]

question = "Should I buy PANW for a short-term hold?"
debate = run_debate(question, gathered_data, analysis_plan)

def show(label, text):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(text)

show("BULL CASE", debate["bull_case"])
show("BEAR CASE", debate["bear_case"])
show("SYNTHESIS", debate["synthesis"])

print("\n" + "=" * 60)
print("MANUAL CHECK")
print("=" * 60)
print("""Read the bull/bear/synthesis above. The 881x P/E carries a pe_ratio_note flagging
it as not economically meaningful (near-zero EPS). Check whether any side cited "881x P/E"
as standalone decisive evidence (bullish or bearish) without acknowledging the note's caveat.
If it's presented at face value anywhere, the prompt instructions haven't taken and need
strengthening further.""")
