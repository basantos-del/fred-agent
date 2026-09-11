import json
from agent import client
from agent_claude import claude_client
from tools import tool_functions, tools, call_groq_with_retry


# --- Triage ---

def classify_query(question):
    prompt = f"""Classify this question about financial analysis/portfolio management as either SIMPLE or COMPLEX.

SIMPLE = a single factual lookup answerable with one tool call (e.g. "what's AAPL's price", "what's my portfolio total").
COMPLEX = requires analysis, judgment, a recommendation, or synthesis across multiple sources (e.g. "should I buy X", "how's my portfolio doing", anything involving a bull/bear case).

Question: {question}

Respond with exactly one word: SIMPLE or COMPLEX"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": 0}
    )
    result = response.content[0].text.strip().upper()
    return "COMPLEX" if "COMPLEX" in result else "SIMPLE"


# --- Researcher ---

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant, not an analyst. Given a
question, call whatever tools are necessary to gather relevant raw data needed to
answer it — but be efficient. For a single-stock question, evaluate_recommendation
alone usually already covers price, ratios, peer comparison, and portfolio context
in one call; only call additional tools (like get_news_context) if something is
still missing after that. Do NOT analyze, do NOT give a recommendation, do NOT
write a bull or bear case — just gather facts using tools. Never call the exact
same tool with the exact same arguments twice. As soon as you have enough data,
STOP calling tools and respond with a one-line summary of what you found."""

def run_researcher(question, max_iterations=5):
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    gathered = []
    already_called = {}

    for _ in range(max_iterations):
        response = call_groq_with_retry(
            client,
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            temperature=0
        )
        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name.split("<")[0]
                args = json.loads(tool_call.function.arguments)
                args = {k: v for k, v in args.items() if k}

                call_key = f"{function_name}({json.dumps(args, sort_keys=True)})"

                if call_key in already_called:
                    result = already_called[call_key]
                    result_note = {"note": "Already retrieved — reusing prior result, do not call again.", "data": result}
                else:
                    function_to_call = tool_functions[function_name]
                    try:
                        result = function_to_call(**args)
                    except Exception as e:
                        result = {"error": f"Tool '{function_name}' failed: {e}"}

                    already_called[call_key] = result
                    gathered.append({"tool": function_name, "args": args, "result": result})
                    result_note = result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result_note)
                })
        else:
            return {"summary": message.content, "gathered_data": gathered}

    return {"summary": "Reached max research iterations.", "gathered_data": gathered}
