import streamlit as st
import httpx
import asyncio

st.set_page_config(page_title="AlphaEdge", page_icon="📈", layout="wide")

st.title("📈 AlphaEdge")
st.markdown("*AI-Powered Market Research Assistant*")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (optional)", placeholder="AAPL")
    api_url = st.text_input("API URL", value="http://localhost:8000")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask about companies, markets, or economics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = httpx.post(
                    f"{api_url}/query",
                    json={
                        "query": prompt,
                        "ticker": ticker if ticker else None,
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                
                st.markdown(data["response"])
                
                # Show metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{data['confidence']:.0%}")
                with col2:
                    st.metric("Intent", data["intent"])
                
                # Show citations
                if data["citations"]:
                    with st.expander("📚 Sources"):
                        for c in data["citations"]:
                            st.markdown(f"**{c['source_id']}** (relevance: {c['relevance_score']:.0%})")
                            st.caption(c["text_excerpt"][:200] + "...")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["response"]
                })
                
            except Exception as e:
                st.error(f"Error: {e}")
