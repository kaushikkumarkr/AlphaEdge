"""
AlphaEdge Frontend - Simplified Single-Page Interface
AI-Powered Multi-Agent Financial Research Platform
"""

import streamlit as st
import httpx
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AlphaEdge - Financial Research",
    page_icon="📈",
    layout="wide",
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 60.0


def check_api_health():
    """Check if backend API is healthy"""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


def query_api(query: str, ticker: str = None):
    """Query the AlphaEdge API"""
    try:
        payload = {"query": query}
        if ticker:
            payload["ticker"] = ticker
        
        response = httpx.post(
            f"{API_URL}/query",
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, "Request timed out. Try a simpler query or check backend status."
    except httpx.HTTPError as e:
        return None, f"API Error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    """Main application"""
    
    # Header
    st.title("📈 AlphaEdge")
    st.markdown("**AI-Powered Multi-Agent Financial Research Platform**")
    
    # API Status
    api_healthy = check_api_health()
    col1, col2 = st.columns([3, 1])
    with col2:
        if api_healthy:
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")
    
    st.markdown("---")
    
    # Query Input Section
    st.subheader("🔍 Research Query")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Enter your question",
            placeholder="e.g., What is Apple's current P/E ratio?\nWhat are Tesla's main risk factors?\nHow has GDP growth been trending?",
            height=100,
            help="Ask about stock valuations, SEC filings, economic indicators, or compare companies"
        )
    
    with col2:
        ticker = st.text_input(
            "Ticker (optional)",
            placeholder="AAPL",
            help="Specify a stock ticker for more focused results"
        )
        
        submit = st.button("🚀 Submit Query", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - What is Apple's current P/E ratio?
        - What are Tesla's main risk factors from their 10-K?
        - How has US GDP growth been trending?
        - Compare Microsoft and Google's profit margins
        - What is the current unemployment rate?
        """)
    
    # Process query
    if submit and query:
        if not api_healthy:
            st.error("⚠️ Cannot submit query - API is offline. Please start the backend server.")
            return
        
        with st.spinner("🔄 Processing your query..."):
            result, error = query_api(query, ticker)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Display Results
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            intent = result.get("intent", "Unknown")
            st.metric("Intent", intent.replace("_", " ").title())
        
        with col2:
            confidence = result.get("confidence", 0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            processing_time = result.get("processing_time_ms", 0)
            st.metric("Processing Time", f"{processing_time/1000:.1f}s")
        
        with col4:
            citations_count = len(result.get("citations", []))
            st.metric("Citations", citations_count)
        
        # Response
        st.markdown("### 💬 Response")
        response_text = result.get("response", "No response received")
        st.markdown(response_text)
        
        # Citations
        citations = result.get("citations", [])
        if citations:
            st.markdown("### 📚 Citations")
            for i, citation in enumerate(citations, 1):
                with st.expander(f"Citation {i}: {citation.get('source', 'Unknown')}"):
                    st.markdown(f"**Source:** {citation.get('source', 'N/A')}")
                    st.markdown(f"**Type:** {citation.get('type', 'N/A')}")
                    if citation.get('url'):
                        st.markdown(f"**URL:** [{citation['url']}]({citation['url']})")
                    if citation.get('excerpt'):
                        st.markdown(f"**Excerpt:**")
                        st.info(citation['excerpt'])
        
        # Query Details (expandable)
        with st.expander("🔍 Query Details"):
            st.json(result)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.text_input("API URL", value=API_URL, disabled=True, help="Configure via API_URL environment variable")
        st.text_input("Timeout", value=f"{API_TIMEOUT}s", disabled=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **AlphaEdge** is a multi-agent AI system for financial research.
        
        **Agents:**
        - 📄 SEC RAG Agent (10-K, 10-Q filings)
        - 📊 OpenBB Agent (Market data)
        - 📈 FRED Agent (Economic indicators)
        
        **Built with:**
        - FastAPI + LangGraph
        - MLX-LM (Local inference)
        - Arize Phoenix (Tracing)
        """)
        
        st.markdown(f"*Last checked: {datetime.now().strftime('%H:%M:%S')}*")


if __name__ == "__main__":
    main()
