import streamlit as st
from agent import run_agent_turn, SYSTEM_PROMPT
from tools import get_portfolio_context

st.title("Fred — Financial Analyst")


@st.cache_data(ttl=300)
def get_cached_portfolio_context():
    return get_portfolio_context()


tab_chat, tab_dashboard = st.tabs(["Chat", "Dashboard"])

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
