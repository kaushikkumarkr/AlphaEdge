"""
AlphaEdge - Simple Chat Interface
"""

import streamlit as st
import httpx
import os

# Page config
st.set_page_config(
    page_title="AlphaEdge Chat",
    page_icon="💬",
    layout="centered",
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("💬 AlphaEdge")
st.markdown("Ask about stocks, SEC filings, and economic data")
st.divider()

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"query": prompt},
                    timeout=60
                )
                result = response.json()
                answer = result.get("response", "No response received")
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
