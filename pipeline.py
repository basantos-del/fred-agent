import json
import re
from agent import client, run_agent_turn, SYSTEM_PROMPT as FRED_SYSTEM_PROMPT
from agent_claude import claude_client
from tools import (
    tool_functions, tools, call_groq_with_retry, filter_args_for_tool,
    get_recent_conversations, get_all_feedback
)

# --- Shared helpers ---

def summarize_for_prompt(gathered_data, per_key_limit=1200):
    compact = []
    for g in gathered_data:
        result = g["result"]
        header = f"### {g['tool']}({json.dumps(g['args'])})"

        if isinstance(result, dict):
            parts = []
            for key, value in result.items():
                value_str = json.dumps(value, indent=2)
                if len(value_str) > per_key_limit:
                    value_str = value_str[:per_key_limit] + f"\n... [truncated, {len(value_str)} chars total]"
                parts.append(f"**{key}**:\n{value_str}")
            body = "\n\n".join(parts)
        else:
            body = json.dumps(result, indent=2)[:per_key_limit]

        compact.append(f"{header}\n{body}")

    return "\n\n".join(compact)

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

    prompt = f"""Question: {question}

Data gathered by tool calls:
{summarize_for_prompt(gathered)}

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


# --- Planner ---

PLANNER_PROMPT_TEMPLATE = """You are a senior financial analyst's planning brain. Given a
question and data already gathered about it, design the analytical outline a junior
analyst (the Advisor) must follow to write a genuinely thorough answer — NOT a generic
checklist. Different questions need different analytical angles.

FIRST, decide whether to ask for clarification. The bar is HIGH.

Ask yourself: "Can a useful, defensible answer be written with the data I already have?"
If YES — do not clarify. Write the plan. Where something is genuinely unknown, instruct
the Advisor to state a reasonable assumption explicitly rather than block on it.

DESCRIPTIVE questions (review my portfolio, how is my allocation, what's my concentration,
analyze this stock, what's the outlook for X) are answerable from data. NEVER clarify on
these — the data is already in front of you.

PRESCRIPTIVE questions (should I buy X, should I sell, should I rebalance, how much should
I allocate) may warrant clarification, but only when the answer would genuinely flip
depending on the response — typically time horizon or position size.

NEVER ask about: age, life stage, risk tolerance, financial goals, tax situation, liquidity
needs, or anything else that reads like an onboarding questionnaire. If these matter,
have the Advisor state an assumption and note how the answer would differ.
NEVER ask about anything already present in the gathered data below.
If you do clarify, ask AT MOST 2 questions, and only ones that would materially change
your recommendation.

Then, if answerable, identify the specific analytical angles a complete answer needs —
these could be anything: sector-specific risks, an upcoming catalyst, regulatory exposure,
competitive dynamics, balance-sheet quality, concentration effects, whatever genuinely
matters for THIS case. For each angle, note why it matters here specifically, and what
additional data (if any) is still needed.

Question: {question}

Data already gathered:
{gathered_data}

Respond with ONLY valid JSON in exactly this shape, no other text:
{{
  "clarification_needed": true or false,
  "clarifying_question": "<at most 2 questions, or null>",
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
        gathered_data=summarize_for_prompt(gathered_data)
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
        parsed = extract_last_json(raw_text)
        if parsed:
            return parsed
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
{summarize_for_prompt(gathered_data)}

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

CLAIM_EXTRACTION_PROMPT = """Extract every specific NUMERIC claim from this financial analysis draft
— a ratio, percentage, dollar figure, or count attributed to a company or metric.
Skip qualitative claims (opinions, trends with no number, risk narratives).

For each claim, give:
- "text": the exact phrase from the draft
- "value": the bare number as a float (strip $, %, "x", commas — "32.1x" -> 32.1, "$3.2 billion" -> 3200000000)
- "derived": true if this looks like something the analyst calculated (a growth rate, a
  percentage of a total, a currency conversion, a sum/difference of other figures) rather
  than a value that would appear as-is in a raw data source; false if it looks directly
  quoted (a ratio, price, margin, or count that would appear verbatim in a data source)

Draft:
{draft}

Return ONLY a JSON object: {{"claims": [{{"text": "...", "value": ..., "derived": true or false}}, ...]}}"""

def extract_claims(draft):
    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(draft=draft)}],
        extra_body={"temperature": 0}
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    parsed = extract_last_json(raw_text)
    return parsed.get("claims", []) if parsed else []

def _flatten_numbers(obj, path=""):
    """Yield (path, value) for every int/float anywhere in a nested dict/list."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield (path, obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_numbers(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten_numbers(v, f"{path}[{i}]")

def verify_claims(claims, gathered_data, tolerance=0.02):
    """Check each directly-sourced numeric claim against every tool result actually
    gathered. Verified if some number in some result is within `tolerance` (relative)
    of the claimed value — accounts for Advisor rounding, e.g. 26.42 -> "26.4x"."""
    results = []
    for claim in claims:
        if claim.get("derived"):
            results.append({**claim, "status": "derived_not_checked", "matched_tool": None})
            continue

        value = claim.get("value")
        if value is None:
            results.append({**claim, "status": "skipped", "matched_tool": None})
            continue

        match = None
        for g in gathered_data:
            call_label = f"{g['tool']}({json.dumps(g['args'], sort_keys=True)})"
            for path, number in _flatten_numbers(g["result"]):
                close_enough = (number == value == 0) or (number != 0 and abs(number - value) / abs(number) <= tolerance)
                if close_enough:
                    match = (call_label, path)
                    break
            if match:
                break

        if match:
            results.append({**claim, "status": "verified", "matched_tool": match[0], "matched_path": match[1]})
        else:
            results.append({**claim, "status": "unverified", "matched_tool": None})

    return results

def run_approver(gathered_data, draft):
    claims = extract_claims(draft)
    checked_claims = verify_claims(claims, gathered_data)
    unverified = [c for c in checked_claims if c["status"] == "unverified"]

    prompt = APPROVER_PROMPT_TEMPLATE.format(
        gathered_data=summarize_for_prompt(gathered_data),
        draft=draft
    )
    response = call_groq_with_retry(
        client,
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000
    )

    raw_text = response.choices[0].message.content.strip()
    parsed = extract_last_json(raw_text)

    if parsed is None or "approved" not in parsed:
        print(f"Approver parse failed, failing open. Raw: {raw_text[:300]}", flush=True)
        parsed = {"approved": True, "issues": [], "parse_error": raw_text}

    if unverified:
        parsed["approved"] = False
        parsed["issues"] = parsed.get("issues", []) + [
            f"Unverified numeric claim: '{c['text']}' (value {c['value']}) — no match in retrieved data"
            for c in unverified
        ]

    parsed["claim_check"] = checked_claims  # full detail, worth logging for the eval suite later
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

# --- Coach ---

COACH_PROMPT_TEMPLATE = """You are reviewing how a financial analyst AI ("Fred") has been
performing, to propose concrete improvements. You have two inputs: recent conversation
transcripts, and explicit user feedback about what was missing from specific answers.

Your job is to find PATTERNS, not one-off complaints — things that recur, or that reveal
a systematic gap in how Fred approaches questions. Then propose specific, actionable
changes.

Two kinds of proposal:
1. SYSTEM_PROMPT changes — a new rule or adjusted instruction for how Fred should analyze
   or respond. Quote the exact line you'd add.
2. EVAL_CASE additions — a new golden-set test case that would catch this gap in future.
   Give the question and the pass criteria.

Be conservative: propose only changes you can justify from the evidence below. If there
isn't enough evidence for a pattern, say so rather than inventing proposals.

Recent conversations:
{conversations}

User feedback:
{feedback}

Respond with ONLY valid JSON in exactly this shape:
{{
  "patterns_observed": ["<pattern you noticed, with evidence>", ...],
  "proposed_prompt_changes": [
    {{"rationale": "<why>", "suggested_line": "<exact line to add to SYSTEM_PROMPT>"}}
  ],
  "proposed_eval_cases": [
    {{"rationale": "<why>", "question": "<test question>", "criteria": "<pass criteria>"}}
  ],
  "insufficient_evidence": true or false
}}"""


def run_coach(limit_threads=10):
    conversations = get_recent_conversations(limit_threads=limit_threads)
    feedback = get_all_feedback()

    if not conversations and not feedback:
        return {
            "patterns_observed": [],
            "proposed_prompt_changes": [],
            "proposed_eval_cases": [],
            "insufficient_evidence": True,
            "note": "No conversations or feedback logged yet."
        }

    convo_text = "\n\n".join(
        f"[Thread {c['thread_id']}] {c['role']}: {c['content'][:1500]}"
        for c in conversations
    )
    feedback_text = "\n\n".join(
        f"[Thread {f['thread_id']}] Question: {f['question'][:500]}\nWhat was missing: {f['what_was_missing']}"
        for f in feedback
    ) or "(no explicit feedback submitted yet)"

    prompt = COACH_PROMPT_TEMPLATE.format(
        conversations=convo_text[:20000],
        feedback=feedback_text[:8000]
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text.strip()
    parsed = extract_last_json(raw_text)

    if parsed is None:
        return {
            "patterns_observed": [],
            "proposed_prompt_changes": [],
            "proposed_eval_cases": [],
            "insufficient_evidence": True,
            "parse_error": raw_text
        }

    return parsed
