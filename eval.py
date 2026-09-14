import json
import re
from datetime import datetime
from agent import SYSTEM_PROMPT
from agent_claude import claude_client
from pipeline import run_pipeline, run_researcher, run_planner, extract_last_json
from tools import sheet

GOLDEN_SET = [
    {
        "id": "price_accuracy",
        "question": "What's AAPL's current price?",
        "category": "tool_grounding",
        "stage": "end_to_end",
        "criteria": "States a specific AAPL price as a dollar figure, with a sane order of magnitude (hundreds of dollars, not millions or single digits)."
    },
    {
        "id": "recommendation_structure",
        "question": "Analyze Microsoft as an investment, assuming a 5-year horizon and a €1000 position.",
        "category": "structural",
        "stage": "end_to_end",
        "criteria": "Presents both a bull case and a bear case explicitly before concluding. A one-sided argument fails."
    },
    {
        "id": "currency_conversion",
        "question": "If I invested $3,000 in Tesla, what percentage of my total portfolio would that represent?",
        "category": "currency",
        "stage": "end_to_end",
        "criteria": "Converts the USD amount to EUR (or explicitly accounts for the currency difference) before comparing to the EUR portfolio total."
    },
    {
        "id": "portfolio_awareness",
        "question": "Should I add more to my Acme Robotics position?",
        "category": "portfolio",
        "stage": "end_to_end",
        "criteria": "References Bernardo's actual Acme Robotics holding and/or his overall US equity concentration, not just generic company analysis."
    },
    {
        "id": "avoid_overclaiming",
        "question": "What's the exact free cash flow figure for ticker XYZQ?",
        "category": "honesty",
        "stage": "end_to_end",
        "criteria": "Indicates the ticker isn't valid or data isn't available, rather than inventing a plausible-sounding figure."
    },
    {
        "id": "no_fabricated_peer_data",
        "question": "How does Microsoft's valuation compare to its closest peers?",
        "category": "regression",
        "stage": "end_to_end",
        "criteria": "Any individual peer figures cited (e.g. 'CrowdStrike's P/E is X') must be plausible and attributed to real retrieved data. Inventing specific per-peer numbers, or citing a figure for a peer while only an average was available, fails."
    },
    {
        "id": "no_clarification_on_descriptive",
        "question": "Give me a full review of my current portfolio.",
        "category": "regression",
        "stage": "end_to_end",
        "criteria": "Provides an actual portfolio review. Responding with clarifying questions instead of analysis fails — this is a descriptive request answerable from available data."
    },
    {
        "id": "no_questionnaire_questions",
        "question": "Do I have concentration risk right now?",
        "category": "regression",
        "stage": "end_to_end",
        "criteria": "Answers with real concentration analysis. Asking about age, risk tolerance, financial goals, or tax situation fails."
    },
    {
        "id": "researcher_fetches_portfolio",
        "question": "Should I buy Microsoft?",
        "category": "researcher",
        "stage": "researcher",
        "criteria": "The gathered data must include portfolio context (holdings, allocation, or exposure data), not just company data — a recommendation question requires knowing what the user already owns."
    },
    {
        "id": "planner_angles_are_specific",
        "question": "Give me a full review of my current portfolio.",
        "category": "planner",
        "stage": "planner",
        "criteria": "The plan must contain multiple analytical angles that are specific to this portfolio's actual composition (e.g. naming real concentrations or holdings), not generic categories that would apply to any portfolio."
    },
]


JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI financial analyst against a specific criterion.

Question asked: {question}

Criterion: {criteria}

What the analyst produced:
{output}

Score it 0-10:
- 0-3: fails the criterion outright
- 4-6: partially meets it, with real gaps
- 7-8: meets it well
- 9-10: meets it fully and thoroughly

Respond with ONLY valid JSON:
{{"score": <0-10>, "reasoning": "<one or two sentences>"}}"""


def judge_output(question, criteria, output):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        criteria=criteria,
        output=output[:8000]
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": 0}
    )

    parsed = extract_last_json(response.content[0].text.strip())
    if parsed is None or "score" not in parsed:
        return 0, f"Judge parse failed: {response.content[0].text[:200]}"

    return parsed["score"], parsed.get("reasoning", "")


def run_single_eval_case(case):
    stage = case.get("stage", "end_to_end")

    try:
        if stage == "researcher":
            research = run_researcher(case["question"])
            output = json.dumps(research["gathered_data"], indent=2)[:8000]

        elif stage == "planner":
            research = run_researcher(case["question"])
            plan = run_planner(case["question"], research["gathered_data"])
            output = json.dumps(plan, indent=2)

        else:
            result = run_pipeline(case["question"])
            output = result["content"]
            if result["type"] == "clarification":
                output = f"[Responded with a clarifying question instead of an answer]\n{output}"

    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "stage": stage,
            "question": case["question"],
            "output": f"Error: {e}",
            "score": 0,
            "reasoning": f"Execution failed: {e}"
        }

    score, reasoning = judge_output(case["question"], case["criteria"], output)

    return {
        "id": case["id"],
        "category": case["category"],
        "stage": stage,
        "question": case["question"],
        "output": output,
        "score": score,
        "reasoning": reasoning
    }


def log_eval_run(results, run_id=None):
    try:
        ws = sheet.worksheet("Eval Runs")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = run_id or timestamp
        rows = [
            [timestamp, r["id"], r["category"], str(r["score"]), r["reasoning"][:2000], run_id]
            for r in results
        ]
        ws.append_rows(rows)
        return True
    except Exception as e:
        print(f"Failed to log eval run: {e}", flush=True)
        return False


def get_eval_history():
    try:
        ws = sheet.worksheet("Eval Runs")
        values = ws.get_all_values()
        return [
            {
                "timestamp": row[0],
                "case_id": row[1] if len(row) > 1 else "",
                "category": row[2] if len(row) > 2 else "",
                "score": float(row[3]) if len(row) > 3 and row[3] else 0,
                "reasoning": row[4] if len(row) > 4 else "",
                "run_id": row[5] if len(row) > 5 else ""
            }
            for row in values[1:] if row and row[0]
        ]
    except Exception as e:
        print(f"Failed to read eval history: {e}", flush=True)
        return []
