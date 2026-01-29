from typing import TypedDict, List, Optional, Annotated, Dict, Any
from operator import add


class AlphaEdgeState(TypedDict):
    query: str
    conversation_id: str
    
    intent: str
    entities: dict
    
    # Simple routing results
    sec_results: Annotated[List[dict], add]
    openbb_results: Annotated[List[dict], add]
    fred_results: Annotated[List[dict], add]
    
    # Multi-task execution
    is_complex: bool
    task_plan: Optional[Dict[str, Any]]
    task_results: Optional[Dict[str, Any]]
    completed_tasks: Annotated[List[str], add]
    
    final_response: str
    citations: List[dict]
    filters: Optional[Dict[str, Any]]
    
    faithfulness_score: float
    confidence_score: float
    needs_escalation: bool
    
    iteration_count: int
    error: Optional[str]
