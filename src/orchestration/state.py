from typing import TypedDict, List, Optional, Annotated
from operator import add


class AlphaEdgeState(TypedDict):
    query: str
    conversation_id: str
    
    intent: str
    entities: dict
    
    sec_results: Annotated[List[dict], add]
    openbb_results: Annotated[List[dict], add]
    fred_results: Annotated[List[dict], add]
    
    final_response: str
    citations: List[dict]
    
    faithfulness_score: float
    confidence_score: float
    needs_escalation: bool
    
    iteration_count: int
    error: Optional[str]
