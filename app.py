import streamlit as st

st.set_page_config(page_title="Fred", layout="wide")

from agent import run_agent_turn, SYSTEM_PROMPT
from agent_claude import run_agent_turn_claude
from tools import get_portfolio_context, log_portfolio_snapshot, get_portfolio_history
from eval import run_eval_suite, GOLDEN_SET

st.title("Hi,Bernardo! Let's get your finances up and running")

@st.cache_data(ttl=300)
def get_cached_portfolio_context():
    return get_portfolio_context()


tab_chat, tab_dashboard, tab_compare, tab_eval = st.tabs(["Chat", "Dashboard", "Model Comparison", "Eval"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    if st.button("New conversation"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
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

    for msg in st.session_state.messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content

        if role in ("user", "assistant") and isinstance(content, str):
            with st.chat_message(role):
                st.markdown(content)

    typed_prompt = st.chat_input("Ask Fred something...")
    prompt = typed_prompt or quick_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, used_tools = run_agent_turn(st.session_state.messages)
                except Exception as e:
                    answer = f"Something went wrong while researching that: {e}. Try rephrasing or asking again."
                    used_tools = []

            st.markdown(answer)
            if used_tools:
                with st.expander(f"Fred used {len(used_tools)} tool call(s)"):
                    for t in used_tools:
                        st.write(f"- `{t}`")

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

    for category, pct in context["allocation_by_exposure"].items():
        if pct >= 40:
            st.warning(f"⚠️ Concentration risk: **{category}** is **{pct}%** of your portfolio.")

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
    st.altair_chart(donut, use_container_width=True)

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
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)

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
        with st.spinner("Running eval suite — this calls the agent and a judge model for each case..."):
            results = st.session_state.get("eval_results")
            results = run_eval_suite()
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
