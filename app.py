import streamlit as st

st.set_page_config(page_title="Fred", layout="wide")

from agent import run_agent_turn, SYSTEM_PROMPT
from agent_claude import run_agent_turn_claude
from tools import get_portfolio_context

st.title("Fred — Financial Analyst")


@st.cache_data(ttl=300)
def get_cached_portfolio_context():
    return get_portfolio_context()


tab_chat, tab_dashboard, tab_compare = st.tabs(["Chat", "Dashboard", "Model Comparison"])

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
    st.subheader("Portfolio")

    context = get_cached_portfolio_context()

    st.metric("Total Value", f"€{context['total_value_eur']:,.2f}")

    st.write("**Allocation by Exposure**")
    st.bar_chart(context["allocation_by_exposure"])

    st.write("**Holdings**")
    st.dataframe(context["holdings"])

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
