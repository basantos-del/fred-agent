import streamlit as st
from itertools import groupby

st.set_page_config(page_title="Fred", layout="wide")

from agent import run_agent_turn, SYSTEM_PROMPT
from agent_claude import run_agent_turn_claude
from pipeline import run_pipeline, run_coach
from tools import (
    get_portfolio_context, log_portfolio_snapshot, get_portfolio_history,
    log_conversation_message, log_feedback, get_all_feedback,
    mark_feedback_addressed, log_coach_adoption, get_coach_log,
)
from eval import run_single_eval_case, GOLDEN_SET

st.title("Hi,Bernardo! Let's get your finances up and running")


@st.cache_data(ttl=300)
def get_cached_portfolio_context():
    return get_portfolio_context()


tab_chat, tab_dashboard, tab_compare, tab_eval, tab_coach = st.tabs(
    ["Chat", "Dashboard", "Model Comparison", "Eval", "Coach"]
)

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    if st.button("New conversation"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.session_state.pop("pending_state", None)
        st.rerun()

    st.write("**Quick questions:**")
    col1, col2, col3 = st.columns(3)
    quick_prompt = None
    with col1:
        if st.button("Portfolio review"):
            quick_prompt = "Give me a full review of my current portfolio."
    with col2:
        if st.button("Check my top holding"):
            quick_prompt = "Analyze my largest single holding."
    with col3:
        if st.button("Any concentration risk?"):
            quick_prompt = "Do I have any concentration risk in my portfolio right now?"

    displayable = [
        m for m in st.session_state.messages
        if (m["role"] if isinstance(m, dict) else m.role) in ("user", "assistant")
        and isinstance((m.get("content") if isinstance(m, dict) else m.content), str)
    ]

    for thread_id, group in groupby(displayable, key=lambda m: m.get("thread_id", 0) if isinstance(m, dict) else 0):
        group = list(group)
        is_last_thread = (group[-1] is displayable[-1])

        first_user_msg = next(
            (m for m in group if (m["role"] if isinstance(m, dict) else m.role) == "user"),
            None
        )
        if first_user_msg is not None:
            label_text = first_user_msg["content"] if isinstance(first_user_msg, dict) else first_user_msg.content
        else:
            label_text = "Conversation"

        if len(label_text) > 70:
            label_text = label_text[:70] + "..."

        awaiting = is_last_thread and st.session_state.get("pending_state")
        label = f"{'❓ ' if awaiting else ''}{label_text}"

        with st.expander(label, expanded=is_last_thread):
            for msg in group:
                role = msg["role"] if isinstance(msg, dict) else msg.role
                content = msg.get("content") if isinstance(msg, dict) else msg.content
                with st.chat_message(role):
                    st.markdown(content)

                    if role == "assistant":
                        with st.expander("📋 Copy this response"):
                            st.code(content, language=None)

                        with st.expander("💬 What was missing from this answer?"):
                            fb_key = f"fb_{thread_id}_{id(msg)}"
                            fb_text = st.text_area(
                                "Feedback",
                                key=fb_key,
                                label_visibility="collapsed",
                                placeholder="What angle would you have liked to see that Fred didn't cover?"
                            )
                            if st.button("Submit feedback", key=f"submit_{fb_key}"):
                                if fb_text.strip():
                                    q = next(
                                        (m["content"] for m in group
                                         if (m["role"] if isinstance(m, dict) else m.role) == "user"),
                                        ""
                                    )
                                    if log_feedback(thread_id, q, fb_text):
                                        st.success("Feedback saved — the Coach will review it.")
                                    else:
                                        st.error("Couldn't save feedback.")

                    if isinstance(msg, dict) and msg.get("meta"):
                        meta = msg["meta"]
                        if meta.get("route"):
                            st.caption(f"Route: {meta['route']}")
                        if meta.get("plan"):
                            with st.expander(f"Planner identified {len(meta['plan'])} analytical angle(s)"):
                                for item in meta["plan"]:
                                    st.write(f"**{item['angle']}**")
                                    st.write(item["why_it_matters_for_this_question"])
                        if meta.get("issues"):
                            with st.expander("⚠️ Approver flagged issues"):
                                for issue in meta["issues"]:
                                    st.write(f"- {issue}")
                        if meta.get("used_tools"):
                            with st.expander(f"Fred used {len(meta['used_tools'])} tool call(s)"):
                                for t in meta["used_tools"]:
                                    st.write(f"- `{t}`")

            if awaiting:
                st.markdown("**Answer to Fred:**")
                follow_up = st.text_input(
                    "Your answer",
                    key=f"followup_{thread_id}",
                    label_visibility="collapsed",
                    placeholder="Answer Fred's question here..."
                )
                if st.button("Send answer", key=f"send_followup_{thread_id}"):
                    if follow_up.strip():
                        st.session_state["submitted_prompt"] = follow_up
                        st.rerun()

    typed_prompt = st.chat_input("Ask Fred something new...")
    prompt = st.session_state.pop("submitted_prompt", None) or typed_prompt or quick_prompt

    if prompt:
        if st.session_state.get("pending_state"):
            thread_id = st.session_state.get("current_thread_id", 0)
        else:
            thread_id = st.session_state.get("current_thread_id", 0) + 1
            st.session_state["current_thread_id"] = thread_id

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
            "used_tools": result.get("used_tools")
        }

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["content"],
            "thread_id": thread_id,
            "meta": meta
        })
        log_conversation_message(thread_id, "assistant", result["content"], result.get("route", ""))

        st.rerun()

    if st.session_state.get("messages") and len(st.session_state["messages"]) > 1:
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

    st.subheader("Portfolio")

    context = get_cached_portfolio_context()

    @st.cache_data(ttl=3600)
    def get_cached_history(total_value_eur):
        log_portfolio_snapshot(total_value_eur)
        return get_portfolio_history()

    history = get_cached_history(context["total_value_eur"])

    st.metric("Total Value", f"€{context['total_value_eur']:,.2f}")

    st.write("**Allocation by Exposure**")
    cols = st.columns(len(context["value_by_exposure"]))
    for col, (category, value) in zip(cols, context["value_by_exposure"].items()):
        pct = context["allocation_by_exposure"].get(category, 0)
        with col:
            st.metric(category, f"€{value:,.0f}", f"{pct}%")

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
        st.line_chart(history_df.set_index("date")["total_value_eur"])
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

with tab_compare:
    st.subheader("Compare Groq (gpt-oss-20b) vs Claude")
    compare_question = st.text_input("Ask both models the same question:")

    if st.button("Compare"):
        compare_instruction = (
            "\n\nFormat your response using bold text and bullet points for "
            "emphasis — avoid large markdown headers (# or ##), since this "
            "will be shown in a narrow side-by-side comparison."
        )
        full_question = compare_question + compare_instruction

        col_groq, col_claude = st.columns(2)

        with col_groq:
            with st.container(border=True):
                st.write("**Groq (gpt-oss-20b)**")
                with st.spinner("Groq thinking..."):
                    try:
                        groq_messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": full_question}
                        ]
                        groq_answer, groq_tools = run_agent_turn(groq_messages)
                    except Exception as e:
                        groq_answer = f"Error: {e}"
                        groq_tools = []
                st.markdown(groq_answer)
                if groq_tools:
                    st.caption(f"Tools used: {', '.join(groq_tools)}")

        with col_claude:
            with st.container(border=True):
                st.write("**Claude (sonnet-4-5)**")
                with st.spinner("Claude thinking..."):
                    try:
                        claude_answer, claude_tools = run_agent_turn_claude(full_question, SYSTEM_PROMPT)
                    except Exception as e:
                        claude_answer = f"Error: {e}"
                        claude_tools = []
                st.markdown(claude_answer)
                if claude_tools:
                    st.caption(f"Tools used: {', '.join(claude_tools)}")

with tab_eval:
    st.subheader("Eval Suite")
    st.caption(f"{len(GOLDEN_SET)} test cases, judged by Claude against specific criteria.")

    if st.button("Run Eval Suite"):
        results = []
        progress_placeholder = st.empty()

        for i, case in enumerate(GOLDEN_SET):
            progress_placeholder.info(f"Running case {i + 1}/{len(GOLDEN_SET)}: {case['id']}...")
            result = run_single_eval_case(case)
            results.append(result)

        progress_placeholder.empty()
        st.session_state["eval_results"] = results

    if "eval_results" in st.session_state:
        results = st.session_state["eval_results"]
        passed = sum(1 for r in results if r["verdict"] == "PASS")
        st.metric("Pass Rate", f"{passed}/{len(results)}")

        for r in results:
            icon = "✅" if r["verdict"] == "PASS" else "❌"
            with st.expander(f"{icon} {r['id']} ({r['category']})"):
                st.write(f"**Question:** {r['question']}")
                st.write(f"**Fred's answer:**")
                st.markdown(r["answer"])
                st.write(f"**Tools used:** {', '.join(r['used_tools']) if r['used_tools'] else 'none'}")
                st.write(f"**Judge verdict:** {r['verdict']}")
                st.write(f"**Judge reasoning:** {r['reasoning']}")

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
