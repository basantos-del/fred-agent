import streamlit as st
import uuid
import json

st.set_page_config(page_title="Fred", layout="wide")

from datetime import datetime
from agent import run_agent_turn, SYSTEM_PROMPT
from pipeline import run_pipeline, run_coach
from tools import (
    get_portfolio_context, log_portfolio_snapshot, get_portfolio_history,
    log_conversation_message, log_feedback, get_all_feedback,
    mark_feedback_addressed, log_coach_adoption, get_coach_log,
    get_recent_conversations,
)
from eval import run_single_eval_case, GOLDEN_SET, log_eval_result, get_eval_history

st.title("Hi,Bernardo! Let's get your finances up and running")


@st.cache_data(ttl=300)
def get_cached_portfolio_context():
    return get_portfolio_context()


@st.cache_data(ttl=60)
def get_cached_recent_conversations():
    return get_recent_conversations(limit_threads=10)


def _relative_time(timestamp_str):
    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""
    seconds = (datetime.now() - ts).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 172800:
        return "Yesterday"
    return f"{int(seconds // 86400)}d ago"


def _group_by_thread(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["thread_id"], []).append(row)
    return grouped


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
st.session_state.setdefault("current_thread_id", uuid.uuid4().hex[:12])

with st.sidebar:
    st.markdown("### Fred")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            width: 100%;
            display: block;
            text-align: left;
            justify-content: flex-start;
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 6px 10px;
            border-radius: 8px;
            font-weight: 400;
            color: inherit;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(128, 128, 128, 0.12);
            color: inherit;
            border: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: rgba(128, 128, 128, 0.20);
            font-weight: 600;
            color: inherit;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: rgba(128, 128, 128, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+ New chat"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state["current_thread_id"] = uuid.uuid4().hex[:12]
        st.session_state.pop("pending_state", None)
        st.session_state.pop("viewing_thread_id", None)
        st.rerun()

    st.divider()

    current_id = st.session_state.get("current_thread_id")
    current_first_user_msg = next(
        (m["content"] for m in st.session_state.messages
         if isinstance(m, dict) and m.get("role") == "user" and m.get("thread_id") == current_id),
        None
    )
    is_viewing_current = st.session_state.get("viewing_thread_id") is None

    if current_first_user_msg or not is_viewing_current:
        current_label = "Current chat"
        if current_first_user_msg:
            current_label = current_first_user_msg[:60] + ("..." if len(current_first_user_msg) > 60 else "")
        if st.button(
            current_label,
            key="sidebar_current_thread",
            type="primary" if is_viewing_current else "secondary",
        ):
            st.session_state.pop("viewing_thread_id", None)
            st.rerun()

    st.markdown(
        "<div style='font-size:11px; font-weight:600; letter-spacing:0.04em; "
        "text-transform:uppercase; color:#8a8d93; margin:18px 2px 4px 10px;'>Previous</div>",
        unsafe_allow_html=True,
    )
    try:
        history_rows = get_cached_recent_conversations()
    except Exception as e:
        history_rows = []
        st.caption(f"Couldn't load history: {e}")

    history_by_thread = _group_by_thread(history_rows)
    previous_thread_ids = [tid for tid in history_by_thread if tid != str(current_id)]
    previous_thread_ids.reverse()

    if not previous_thread_ids:
        st.caption("No previous conversations yet.")

    for tid in previous_thread_ids:
        rows = history_by_thread[tid]
        first_user_row = next((r for r in rows if r["role"] == "user"), None)
        label = "Conversation"
        if first_user_row:
            label = first_user_row["content"][:60]
            if len(first_user_row["content"]) > 60:
                label += "..."
        is_active = st.session_state.get("viewing_thread_id") == tid
        if st.button(
            label,
            key=f"sidebar_thread_{tid}",
            type="primary" if is_active else "secondary",
        ):
            st.session_state["viewing_thread_id"] = tid
            st.rerun()

tab_chat, tab_dashboard, tab_eval, tab_coach = st.tabs(
    ["Chat", "Dashboard", "Eval", "Coach"]
)

with tab_chat:
    active_thread_id = st.session_state.get("viewing_thread_id")
    viewing_history = active_thread_id is not None
    if not viewing_history:
        active_thread_id = st.session_state.get("current_thread_id", 0)

    displayable = [
        m for m in st.session_state.messages
        if (m["role"] if isinstance(m, dict) else m.role) in ("user", "assistant")
        and isinstance((m.get("content") if isinstance(m, dict) else m.content), str)
    ]

    thread_messages = [m for m in displayable if str(m.get("thread_id", 0)) == str(active_thread_id)]

    if not thread_messages:
        history_rows = get_cached_recent_conversations()
        history_by_thread = _group_by_thread(history_rows)
        rows = history_by_thread.get(str(active_thread_id), [])
        thread_messages = [
            {"role": r["role"], "content": r["content"], "thread_id": active_thread_id}
            for r in rows if r["role"] in ("user", "assistant")
        ]

    is_empty_thread = len(thread_messages) == 0
    quick_prompt = None

    if is_empty_thread and not viewing_history:
        st.write("**Quick questions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Portfolio review"):
                quick_prompt = "Give me a full review of my current portfolio."
        with col2:
            if st.button("Check my top holding"):
                quick_prompt = "Analyze my largest single holding."
        with col3:
            if st.button("Any concentration risk?"):
                quick_prompt = "Do I have any concentration risk in my portfolio right now?"

    for msg in thread_messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                copy_id = f"copy_{active_thread_id}_{id(msg)}"
                st.components.v1.html(
                    f"""
                    <button id="{copy_id}" onclick="
                        navigator.clipboard.writeText({json.dumps(content)});
                        const btn = document.getElementById('{copy_id}');
                        btn.innerText = '✅ Copied';
                        setTimeout(() => {{ btn.innerText = '📋 Copy response'; }}, 1500);
                    " style="
                        background: #f0f2f6;
                        border: 1px solid #d3d6db;
                        border-radius: 6px;
                        padding: 4px 12px;
                        font-size: 13px;
                        font-family: 'Source Sans Pro', sans-serif;
                        cursor: pointer;
                        color: #31333f;
                    ">📋 Copy response</button>
                    """,
                    height=36,
                )

                with st.expander("💬 What was missing from this answer?"):
                    fb_key = f"fb_{active_thread_id}_{id(msg)}"
                    fb_text = st.text_area(
                        "Feedback",
                        key=fb_key,
                        label_visibility="collapsed",
                        placeholder="What angle would you have liked to see that Fred didn't cover?"
                    )
                    if st.button("Submit feedback", key=f"submit_{fb_key}"):
                        if fb_text.strip():
                            q = next(
                                (m["content"] for m in thread_messages if m["role"] == "user"),
                                ""
                            )
                            if log_feedback(active_thread_id, q, fb_text):
                                st.success("Feedback saved — the Coach will review it.")
                            else:
                                st.error("Couldn't save feedback.")

            if isinstance(msg, dict) and msg.get("meta"):
                meta = msg["meta"]
                if meta.get("route"):
                    st.caption(f"Route: {meta['route']}")

                has_details = any(meta.get(k) for k in ("plan", "issues", "used_tools", "skipped_data_requests"))
                if has_details:
                    with st.expander("Details"):
                        if meta.get("plan"):
                            st.write(f"**Planner identified {len(meta['plan'])} analytical angle(s):**")
                            for item in meta["plan"]:
                                st.write(f"- **{item['angle']}** — {item['why_it_matters_for_this_question']}")
                        if meta.get("issues"):
                            st.write("**⚠️ Approver flagged issues:**")
                            for issue in meta["issues"]:
                                st.write(f"- {issue}")
                        if meta.get("used_tools"):
                            st.write(f"**Fred used {len(meta['used_tools'])} tool call(s):**")
                            for t in meta["used_tools"]:
                                st.write(f"- `{t}`")
                        if meta.get("skipped_data_requests"):
                            skipped = meta["skipped_data_requests"]
                            st.write(f"**⚠️ {len(skipped)} data request(s) dropped — unknown tool name:**")
                            for s in skipped:
                                st.write(f"- `{s['requested_tool']}` (for angle: *{s['angle']}*)")

    resolving_followup = bool(st.session_state.get("submitted_prompt"))
    awaiting = (not viewing_history) and bool(st.session_state.get("pending_state")) and not resolving_followup

    if awaiting:
        st.markdown("**Answer to Fred:**")
        follow_up = st.text_input(
            "Your answer",
            key=f"followup_{active_thread_id}",
            label_visibility="collapsed",
            placeholder="Answer Fred's question here..."
        )
        if st.button("Send answer", key=f"send_followup_{active_thread_id}"):
            if follow_up.strip():
                st.session_state["submitted_prompt"] = follow_up
                st.rerun()

    if viewing_history:
        st.info("You're viewing a past conversation. Ask a new question below to start a fresh one.")

    typed_prompt = None if awaiting else st.chat_input("Ask Fred something new...")
    prompt = st.session_state.pop("submitted_prompt", None) or typed_prompt or quick_prompt

    if prompt:
        st.session_state.pop("viewing_thread_id", None)

        if not st.session_state.get("pending_state"):
            st.session_state["current_thread_id"] = uuid.uuid4().hex[:12]
        thread_id = st.session_state["current_thread_id"]

        st.session_state.messages.append({"role": "user", "content": prompt, "thread_id": thread_id})
        log_conversation_message(thread_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Starting...", expanded=False)
            try:
                pending = st.session_state.get("pending_state")
                result = run_pipeline(
                    prompt,
                    pending_state=pending,
                    on_progress=lambda stage: status.update(label=stage)
                )
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                result = {
                    "type": "answer",
                    "content": f"Something went wrong while researching that: {e}. Try rephrasing or asking again.",
                    "route": "error"
                }

        if result["type"] == "clarification":
            st.session_state["pending_state"] = result["pending_state"]
        else:
            st.session_state.pop("pending_state", None)

        verdict = result.get("approver_verdict") or {}
        meta = {
            "route": result.get("route"),
            "plan": result.get("plan"),
            "issues": None if verdict.get("approved", True) else verdict.get("issues"),
            "used_tools": result.get("used_tools"),
            "skipped_data_requests": result.get("skipped_data_requests")
        }

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["content"],
            "thread_id": thread_id,
            "meta": meta
        })
        log_conversation_message(thread_id, "assistant", result["content"], result.get("route", ""))

        get_cached_recent_conversations.clear()
        st.rerun()

    if thread_messages:
        st.components.v1.html(
            """
            <script>
                const doc = window.parent.document;
                doc.querySelector('section.main')?.scrollTo(0, doc.querySelector('section.main').scrollHeight);
            </script>
            """,
            height=0
        )

with tab_dashboard:
    import pandas as pd
    import altair as alt

    def format_k_eur(value):
        """Abbreviated euro format for dashboard display: 15234.56 -> '15.2K€'.
        Falls back to a plain euro format under 1,000 where K-abbreviation reads oddly."""
        if value is None:
            return "€0"
        if abs(value) >= 1000:
            return f"{value / 1000:.1f}K€"
        return f"€{value:,.0f}"

    st.subheader("Portfolio")

    context = get_cached_portfolio_context()

    @st.cache_data(ttl=3600)
    def get_cached_history(total_value_eur):
        log_portfolio_snapshot(total_value_eur)
        return get_portfolio_history()

    history = get_cached_history(context["total_value_eur"])

    st.metric("Total Value", format_k_eur(context["total_value_eur"]))

    st.write("**Allocation by Exposure**")
    cols = st.columns(len(context["value_by_exposure"]))
    for col, (category, value) in zip(cols, context["value_by_exposure"].items()):
        pct = context["allocation_by_exposure"].get(category, 0)
        with col:
            st.metric(category, format_k_eur(value), f"{pct}%")

    chart_df = pd.DataFrame({
        "category": list(context["value_by_exposure"].keys()),
        "value": list(context["value_by_exposure"].values())
    })
    donut = alt.Chart(chart_df).mark_arc(innerRadius=70).encode(
        theta="value",
        color="category",
        tooltip=["category", "value"]
    ).properties(height=350)
    st.altair_chart(donut, width='stretch')

    st.write("**Portfolio Value Over Time**")
    if len(history) >= 2:
        history_df = pd.DataFrame(history)
        history_df["date"] = pd.to_datetime(history_df["date"])
        history_df["date_label"] = history_df["date"].dt.strftime("%b %d")

        min_val = history_df["total_value_eur"].min()
        max_val = history_df["total_value_eur"].max()
        padding = (max_val - min_val) * 0.1 or max_val * 0.05
        y_domain = [max(0, min_val - padding), max_val + padding]

        line = alt.Chart(history_df).mark_line(point=True).encode(
            x=alt.X("date_label:O", title=None, sort=list(history_df["date_label"])),
            y=alt.Y("total_value_eur:Q", title="Total Value (€)", scale=alt.Scale(domain=y_domain)),
            tooltip=[alt.Tooltip("date_label:O", title="Date"), alt.Tooltip("total_value_eur:Q", format=",.0f", title="Value (€)")]
        ).properties(height=350)
        st.altair_chart(line, width='stretch')
    else:
        st.caption("History will build up as you use the dashboard over time (logs once per day).")

    st.write("**Holdings**")
    holdings_df = pd.DataFrame(context["holdings"]).sort_values("value_eur", ascending=False)
    holdings_df["ticker"] = holdings_df["ticker"].fillna("—")
    holdings_df["value_eur"] = holdings_df["value_eur"].apply(lambda v: f"€{v:,.2f}")
    st.dataframe(holdings_df, width='stretch', hide_index=True)

    st.subheader("Concentration Alerts")

    alert_rows = []
    for category, pct in context["allocation_by_exposure"].items():
        if pct >= 40:
            alert_rows.append({"Type": "Exposure Category", "Name": category, "% of Portfolio": pct})
    for industry, pct in context["allocation_by_industry"].items():
        if pct >= 40:
            alert_rows.append({"Type": "Sector", "Name": industry, "% of Portfolio": pct})
    if context["magnificent_7_pct"] >= 40:
        alert_rows.append({"Type": "Magnificent 7", "Name": "Magnificent 7 stocks", "% of Portfolio": context["magnificent_7_pct"]})

    if alert_rows:
        st.warning(f"⚠️ {len(alert_rows)} concentration alert(s) — see table below.")
        st.dataframe(pd.DataFrame(alert_rows), width='stretch', hide_index=True)
    else:
        st.success("✅ No concentration alerts — nothing currently exceeds the 40% threshold.")

    st.subheader("Full Sector & Magnificent 7 Breakdown")

    sector_df = pd.DataFrame(
        [{"Sector": k, "% of Portfolio": v} for k, v in context["allocation_by_industry"].items()]
    ).sort_values("% of Portfolio", ascending=False)
    st.write("**Sector allocation (including fund look-through):**")
    st.dataframe(sector_df, width='stretch', hide_index=True)

    st.write(f"**Magnificent 7 total: {context['magnificent_7_pct']}% of portfolio**")
    if context["magnificent_7_lookthrough_detail"]:
        mag7_df = pd.DataFrame(context["magnificent_7_lookthrough_detail"])
        mag7_df = mag7_df.rename(columns={"holding": "Holding", "fund": "Via Fund", "value_eur": "Value (EUR)"})
        mag7_df["Value (EUR)"] = mag7_df["Value (EUR)"].apply(lambda v: f"€{v:,.2f}")
        st.dataframe(mag7_df, width='stretch', hide_index=True)
    else:
        st.caption("No Magnificent 7 exposure detected via fund look-through.")

with tab_eval:
    st.subheader("Eval Suite")
    st.caption(
        f"{len(GOLDEN_SET)} test cases across end-to-end, researcher, and planner stages. "
        "Scored 0-10 by Claude against per-case criteria."
    )

    st.write("**Select cases to run:**")

    all_stages = sorted({c.get("stage", "end_to_end") for c in GOLDEN_SET})
    all_categories = sorted({c["category"] for c in GOLDEN_SET})

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        stage_filter = st.multiselect("Stage", all_stages, default=all_stages)
    with col_f2:
        category_filter = st.multiselect("Category", all_categories, default=all_categories)

    filtered_cases = [
        c for c in GOLDEN_SET
        if c.get("stage", "end_to_end") in stage_filter and c["category"] in category_filter
    ]

    st.caption(f"{len(filtered_cases)} case(s) selected.")

    if st.button("Run selected cases"):
        run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = []
        progress_placeholder = st.empty()

        for i, case in enumerate(filtered_cases):
            progress_placeholder.info(
                f"Running case {i + 1}/{len(filtered_cases)}: {case['id']} ({case.get('stage', 'end_to_end')})..."
            )
            try:
                result = run_single_eval_case(case)
            except Exception as e:
                result = {
                    "id": case["id"],
                    "category": case["category"],
                    "stage": case.get("stage", "end_to_end"),
                    "question": case["question"],
                    "output": f"Error: {e}",
                    "score": 0,
                    "reasoning": f"Run failed: {e}"
                }

            results.append(result)
            log_eval_result(result, run_id)
            st.session_state["eval_results"] = results
            st.session_state["eval_expected_count"] = len(filtered_cases)

        progress_placeholder.empty()

    results = st.session_state.get("eval_results")
    if results:
        expected = st.session_state.get("eval_expected_count", len(results))
        if len(results) < expected:
            st.warning(
                f"⚠️ Partial run: {len(results)} of {expected} cases completed. "
                "Results logged for those that finished."
            )

        avg_score = sum(r["score"] for r in results) / len(results)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Average Score", f"{avg_score:.1f}/10")
        with col_b:
            failing = sum(1 for r in results if r["score"] < 7)
            st.metric("Cases below 7", f"{failing}/{len(results)}")

        st.write("**By stage:**")
        stage_cols = st.columns(3)
        for col, stage in zip(stage_cols, ["end_to_end", "researcher", "planner"]):
            stage_results = [r for r in results if r["stage"] == stage]
            with col:
                if stage_results:
                    stage_avg = sum(r["score"] for r in stage_results) / len(stage_results)
                    st.metric(stage, f"{stage_avg:.1f}/10", f"{len(stage_results)} case(s)")
                else:
                    st.metric(stage, "—")

        st.divider()

        for r in sorted(results, key=lambda x: x["score"]):
            if r["score"] >= 7:
                icon = "✅"
            elif r["score"] >= 4:
                icon = "⚠️"
            else:
                icon = "❌"

            with st.expander(f"{icon} {r['score']}/10 · {r['id']} ({r['category']} · {r['stage']})"):
                st.write(f"**Question:** {r['question']}")
                st.write(f"**Judge reasoning:** {r['reasoning']}")
                st.write("**Output produced:**")
                st.code(r["output"][:5000], language=None)

    st.divider()
    st.subheader("Score trend over time")

    history = get_eval_history()
    if len(history) < 2:
        st.caption("Run the suite a few times to build a trend.")
    else:
        import pandas as pd

        hist_df = pd.DataFrame(history)
        run_avg = hist_df.groupby("run_id")["score"].mean().reset_index()
        run_avg = run_avg.sort_values("run_id")
        st.line_chart(run_avg.set_index("run_id")["score"])

        st.write("**Per-case trend:**")
        pivot = hist_df.pivot_table(index="run_id", columns="case_id", values="score", aggfunc="mean")
        st.dataframe(pivot.sort_index(ascending=False), width='stretch')

with tab_coach:
    st.subheader("Coach — Self-Improvement Review")
    st.caption(
        "Reviews recent conversations and your unaddressed feedback to propose improvements. "
        "Proposals are never applied automatically — you decide what to adopt."
    )

    if st.button("Run Coach review"):
        with st.spinner("Reviewing conversations and feedback..."):
            st.session_state["coach_result"] = run_coach()

    coach = st.session_state.get("coach_result")
    if coach:
        if coach.get("insufficient_evidence"):
            st.info("Not enough evidence yet to propose meaningful changes.")
            if coach.get("note"):
                st.caption(coach["note"])

        if coach.get("patterns_observed"):
            st.write("**Patterns observed:**")
            for p in coach["patterns_observed"]:
                st.write(f"- {p}")

        if coach.get("proposed_prompt_changes"):
            st.write("**Proposed SYSTEM_PROMPT additions:**")
            for change in coach["proposed_prompt_changes"]:
                with st.container(border=True):
                    st.write(f"_{change['rationale']}_")
                    st.code(change["suggested_line"], language=None)

        if coach.get("proposed_eval_cases"):
            st.write("**Proposed new eval cases:**")
            for case in coach["proposed_eval_cases"]:
                with st.container(border=True):
                    st.write(f"_{case['rationale']}_")
                    st.write(f"**Question:** {case['question']}")
                    st.write(f"**Criteria:** {case['criteria']}")

        if coach.get("parse_error"):
            with st.expander("Raw output (parsing failed)"):
                st.code(coach["parse_error"])

    st.divider()
    st.subheader("Mark feedback as addressed")
    st.caption("Once you've added a proposal to SYSTEM_PROMPT or the golden set, mark the "
               "feedback it came from so the Coach stops resurfacing it.")

    pending_feedback = get_all_feedback()
    if not pending_feedback:
        st.success("✅ No unaddressed feedback.")
    else:
        selected_rows = []
        for fb in pending_feedback:
            checked = st.checkbox(
                f"[{fb['timestamp']}] {fb['what_was_missing'][:120]}",
                key=f"fbmark_{fb['row_number']}"
            )
            if checked:
                selected_rows.append(fb["row_number"])

        adoption_summary = st.text_area(
            "What did you add? (logged as a record)",
            placeholder="e.g. Added a SYSTEM_PROMPT line requiring explicit FX-risk commentary on USD holdings.",
            key="adoption_summary"
        )

        if st.button("Mark selected as addressed"):
            if not selected_rows:
                st.warning("Select at least one feedback item.")
            elif not adoption_summary.strip():
                st.warning("Add a short summary of what you changed.")
            else:
                mark_feedback_addressed(selected_rows)
                log_coach_adoption(adoption_summary, selected_rows)
                st.success(f"Marked {len(selected_rows)} item(s) as addressed.")
                st.rerun()

    st.divider()
    st.subheader("Adoption history")
    log_entries = get_coach_log()
    if not log_entries:
        st.caption("Nothing adopted yet.")
    else:
        for entry in reversed(log_entries):
            with st.container(border=True):
                st.caption(f"{entry['timestamp']} · feedback rows {entry['rows']}")
                st.write(entry["summary"])
