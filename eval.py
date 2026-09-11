from agent import run_agent_turn, SYSTEM_PROMPT
from agent_claude import claude_client

GOLDEN_SET = [
    {
        "id": "price_accuracy",
        "question": "What's AAPL's current price?",
        "category": "tool_grounding",
        "criteria": "The response should state a specific AAPL stock price as a dollar figure. It should not contain any other ticker's price or an obviously wrong order of magnitude (e.g. not $200M, not $3)."
    },
    {
        "id": "recommendation_structure",
        "question": "Should I buy Microsoft stock?",
        "category": "structural",
        "criteria": "The response must explicitly present both a bull case (reasons in favor) and a bear case (reasons against) before any conclusion. A response that only argues one side fails this check."
    },
    {
        "id": "currency_conversion",
        "question": "If I invested $3,000 in Tesla, what percentage of my total portfolio would that represent?",
        "category": "currency",
        "criteria": "The response must convert the USD amount to EUR (or explicitly account for the currency difference) before comparing it to the portfolio total, rather than treating $3,000 as if it were €3,000 directly."
    },
    {
        "id": "portfolio_awareness",
        "question": "Should I add more to my Acme Robotics position?",
        "category": "portfolio",
        "criteria": "The response should reference Bernardo's existing Acme Robotics holding and/or overall US equity concentration from his actual portfolio data, not just generic company analysis with no portfolio context."
    },
    {
        "id": "avoid_overclaiming",
        "question": "What's the exact free cash flow figure for a random small-cap biotech ticker like ABCD?",
        "category": "honesty",
        "criteria": "Since ABCD is not a real, well-covered ticker, the response should indicate uncertainty, missing data, or inability to find the ticker — not confidently invent a plausible-sounding number."
    }
]


def judge_response(question, criteria, response_text):
    prompt = f"""You are evaluating an AI financial analyst's response against a specific criterion.

Question asked: {question}

Criterion to check: {criteria}

The analyst's actual response:
{response_text}

Does this response satisfy the criterion? Respond in exactly this format:
VERDICT: PASS or FAIL
REASONING: <one or two sentences explaining why>"""

    result = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    judge_text = result.content[0].text
    verdict = "PASS" if "VERDICT: PASS" in judge_text else "FAIL"
    reasoning = judge_text.split("REASONING:")[-1].strip() if "REASONING:" in judge_text else judge_text

    return verdict, reasoning


def run_eval_suite():
    results = []

    for case in GOLDEN_SET:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]}
        ]

        try:
            answer, used_tools = run_agent_turn(messages)
        except Exception as e:
            answer = f"Error running agent: {e}"
            used_tools = []

        verdict, reasoning = judge_response(case["question"], case["criteria"], answer)

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer,
            "used_tools": used_tools,
            "verdict": verdict,
            "reasoning": reasoning
        })

    return results
