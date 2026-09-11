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

    for msg in st.session_state.messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content

        if role in ("user", "assistant") and isinstance(content, str):
            with st.chat_message(role):
                st.markdown(content)

    if prompt := st.chat_input("Ask Fred something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = run_agent_turn(st.session_state.messages)
            st.markdown(answer)

with tab_dashboard:
    st.subheader("Portfolio")

    context = get_cached_portfolio_context()

    st.metric("Total Value", f"€{context['total_value_eur']:,.2f}")

    st.write("**Allocation by Exposure**")
    st.bar_chart(context["allocation_by_exposure"])

    st.write("**Holdings**")
    st.dataframe(context["holdings"])
