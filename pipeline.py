import json
import re
from agent import client
from agent_claude import claude_client
from tools import tool_functions, tools, call_groq_with_retry, filter_args_for_tool
from agent import client, SYSTEM_PROMPT as FRED_SYSTEM_PROMPT
from agent import client, run_agent_turn, SYSTEM_PROMPT as FRED_SYSTEM_PROMPT

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


def summarize_gathered_data(question, gathered):
    if not gathered:
        return "No data was gathered."

    data_str = json.dumps(gathered, indent=2)[:4000]
    prompt = f"""Question: {question}

Data gathered by tool calls:
{data_str}

Write a ONE-LINE summary of what data was gathered (not an analysis, not a recommendation)."""

    response = call_groq_with_retry(
        client,
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def run_researcher(question, max_iterations=4):
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
            tool_choice="auto",
            temperature=0
        )
        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name.split("<")[0]
                args = json.loads(tool_call.function.arguments)
                args = {k: v for k, v in args.items() if k}
                args = filter_args_for_tool(function_name, args)

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

    summary = summarize_gathered_data(question, gathered)
    return {"summary": summary, "gathered_data": gathered}

# Planner

PLANNER_PROMPT_TEMPLATE = """You are a senior financial analyst's planning brain. Given a
question and data already gathered about it, your job is to design the analytical
outline a junior analyst (the Advisor) must follow to write a genuinely thorough,
well-reasoned answer — NOT a generic checklist. Different questions need different
analytical angles: think specifically about what THIS question and THIS data demand.

First, check: is the question missing critical information needed to answer it well
(e.g. unclear time horizon, unclear investment goal, ambiguous scope)? If so, don't
build a plan — ask for clarification instead.

If the question is answerable, identify the specific analytical angles a complete
answer needs — these could be anything: sector-specific risks, an upcoming known
catalyst, regulatory exposure, competitive dynamics, balance-sheet quality, whatever
genuinely matters for THIS case. For each angle, note why it matters here specifically,
and what additional data (if any) is still needed beyond what's already been gathered.

Question: {question}

Data already gathered:
{gathered_data}

Respond with ONLY valid JSON in exactly this shape, no other text:
{{
  "clarification_needed": true or false,
  "clarifying_question": "<question to ask, or null>",
  "analysis_plan": [
    {{
      "angle": "<the specific analytical angle, in your own words>",
      "why_it_matters_for_this_question": "<reasoning>",
      "data_still_needed": [{{"tool": "<tool name>", "args": {{}}}}]
    }}
  ]
}}"""


def run_planner(question, gathered_data):
    prompt = PLANNER_PROMPT_TEMPLATE.format(
        question=question,
        gathered_data=json.dumps(gathered_data, indent=2)[:4000]
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text.strip()
    raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "clarification_needed": False,
            "clarifying_question": None,
            "analysis_plan": [],
            "parse_error": raw_text
        }

# --- Advisor ---

def run_advisor(question, gathered_data, analysis_plan):
    already_called = {
        f"{g['tool']}({json.dumps(g['args'], sort_keys=True)})": g["result"]
        for g in gathered_data
    }

    for item in analysis_plan:
        for call in item.get("data_still_needed", []):
            tool_name = call.get("tool")
            args = filter_args_for_tool(tool_name, call.get("args", {}))
            call_key = f"{tool_name}({json.dumps(args, sort_keys=True)})"

            if call_key in already_called or tool_name not in tool_functions:
                continue

            try:
                result = tool_functions[tool_name](**args)
            except Exception as e:
                result = {"error": f"Tool '{tool_name}' failed: {e}"}

            already_called[call_key] = result
            gathered_data.append({"tool": tool_name, "args": args, "result": result})

    plan_text = "\n".join(
        f"- {item['angle']}: {item['why_it_matters_for_this_question']}"
        for item in analysis_plan
    )

    prompt = f"""Question: {question}

Your senior analyst planner has identified these angles that must be covered:
{plan_text}

All available data:
{json.dumps(gathered_data, indent=2)[:6000]}

Write your full analysis, explicitly addressing each angle above, following your
standard analytical rules (bull/bear cases, quantitative, portfolio-aware)."""

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=FRED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    return "".join(block.text for block in response.content if block.type == "text")

# --- Approver ---

APPROVER_PROMPT_TEMPLATE = """You are a fact-checker reviewing a financial analyst's draft
answer against the raw data it was supposed to be based on. Your ONLY job is catching
inconsistencies — you are not judging style, tone, or whether the conclusion is wise.

Look specifically for:
- Numbers in the draft that don't match the retrieved data (wrong values, wrong units,
  wrong order of magnitude)
- Claims presented as fact that the retrieved data doesn't support
- Internally contradictory statements within the draft

Do NOT flag: reasonable rounding, reasonable interpretation of data, judgment calls,
caveated estimates, or anything the draft itself explicitly marks as uncertain.

Raw data that was available:
{gathered_data}

The draft answer:
{draft}

First, think through your review inside <thinking></thinking> tags. Then give your
final verdict as a single JSON object AFTER the closing tag, in exactly this shape:
{{
  "approved": true or false,
  "issues": ["<specific issue found>", ...]
}}"""


def extract_last_json(text):
    matches = re.findall(r"\{.*?\}(?=\s*(?:```|$|\n\n))", text, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{.*\}", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def run_approver(gathered_data, draft):
    prompt = APPROVER_PROMPT_TEMPLATE.format(
        gathered_data=json.dumps(gathered_data, indent=2)[:6000],
        draft=draft
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": 0}
    )

    raw_text = response.content[0].text.strip()
    parsed = extract_last_json(raw_text)

    if parsed is None or "approved" not in parsed:
        print(f"Approver parse failed, failing open. Raw: {raw_text[:300]}", flush=True)
        return {"approved": True, "issues": [], "parse_error": raw_text}

    return parsed


def run_advisor_with_approval(question, gathered_data, analysis_plan):
    draft = run_advisor(question, gathered_data, analysis_plan)
    verdict = run_approver(gathered_data, draft)

    if verdict["approved"]:
        return draft, verdict

    issues_text = "\n".join(f"- {issue}" for issue in verdict["issues"])
    correction_note = f"""

IMPORTANT: A fact-checker reviewed your previous draft and found these issues:
{issues_text}

Rewrite your analysis correcting these specific problems. Use only figures that
actually appear in the provided data."""

    retry_draft = run_advisor(question + correction_note, gathered_data, analysis_plan)
    retry_verdict = run_approver(gathered_data, retry_draft)

    if retry_verdict["approved"]:
        return retry_draft, retry_verdict

    flagged = retry_draft + "\n\n---\n⚠️ **Unverified figures**: " + "; ".join(retry_verdict["issues"])
    return flagged, retry_verdict

# --- Orchestrator ---

def run_pipeline(question, pending_state=None, on_progress=None):
    def report(stage):
        if on_progress:
            on_progress(stage)

    if pending_state is not None:
        original_question = pending_state["question"]
        gathered_data = pending_state["gathered_data"]
        clarifying_question = pending_state["clarifying_question"]

        combined_question = (
            f"{original_question}\n\n"
            f"[Clarification asked: {clarifying_question}]\n"
            f"[User answered: {question}]"
        )

        report("Re-planning with your answer...")
        plan = run_planner(combined_question, gathered_data)
        analysis_plan = plan.get("analysis_plan") or []

        report("Writing the analysis...")
        answer, verdict = run_advisor_with_approval(combined_question, gathered_data, analysis_plan)

        return {
            "type": "answer",
            "content": answer,
            "route": "complex (resumed)",
            "plan": analysis_plan,
            "approver_verdict": verdict,
            "gathered_data": gathered_data
        }

    report("Deciding how to approach this...")
    route = classify_query(question)

    if route == "SIMPLE":
        report("Looking that up...")
        messages = [
            {"role": "system", "content": FRED_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        answer, used_tools = run_agent_turn(messages)
        return {
            "type": "answer",
            "content": answer,
            "route": "simple",
            "used_tools": used_tools
        }

    report("Researching — gathering data...")
    research = run_researcher(question)
    gathered_data = research["gathered_data"]

    report("Planning the analysis...")
    plan = run_planner(question, gathered_data)

    if plan.get("clarification_needed"):
        return {
            "type": "clarification",
            "content": plan["clarifying_question"],
            "route": "complex",
            "pending_state": {
                "question": question,
                "gathered_data": gathered_data,
                "clarifying_question": plan["clarifying_question"]
            }
        }

    analysis_plan = plan.get("analysis_plan") or []

    report("Writing the analysis...")
    answer, verdict = run_advisor_with_approval(question, gathered_data, analysis_plan)

    return {
        "type": "answer",
        "content": answer,
        "route": "complex",
        "plan": analysis_plan,
        "approver_verdict": verdict,
        "gathered_data": gathered_data
    }
