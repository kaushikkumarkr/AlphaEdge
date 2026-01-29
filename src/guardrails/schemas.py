from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_type: Literal["sec_filing", "financial_data", "macro_data"]
    source_id: str
    text_excerpt: str = Field(..., max_length=500)
    relevance_score: float = Field(..., ge=0, le=1)


class RetrievedContext(BaseModel):
    source_id: str
    text: str
    relevance_score: float
    metadata: Dict[str, Any] = {}


class AgentInput(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    context: Optional[str] = None
    filters: Dict[str, Any] = {}


class AgentOutput(BaseModel):
    agent_name: str
    response_text: str
    citations: List[Citation] = []
    retrieved_contexts: List[RetrievedContext] = []
    confidence_score: float = Field(..., ge=0, le=1)
    processing_time_ms: int


class CriticOutput(BaseModel):
    faithfulness_score: float = Field(..., ge=0, le=1)
    citation_coverage: float = Field(..., ge=0, le=1)
    unsupported_claims: List[str] = []
    passed: bool


class FinalResponse(BaseModel):
    query: str
    response: str
    citations: List[Citation]
    agents_used: List[str]
    faithfulness_score: float
    confidence_score: float
    needs_human_review: bool = False
