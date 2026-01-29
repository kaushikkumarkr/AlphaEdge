# Retrieval
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 10
RERANK_TOP_K = 5

# Agent Names
class AgentName:
    SEC_RAG = "sec_rag_agent"
    OPENBB = "openbb_agent"
    FRED = "fred_agent"
    SYNTHESIS = "synthesis_agent"
    CRITIC = "critic_agent"

# Intents
class Intent:
    SEC_FILING = "sec_filing"
    FINANCIALS = "financials"
    MACRO = "macro"
    SYNTHESIS = "synthesis"

# FRED Series
FRED_SERIES = {
    "gdp": "GDP",
    "unemployment": "UNRATE",
    "inflation": "CPIAUCSL",
    "interest_rate": "FEDFUNDS",
}
