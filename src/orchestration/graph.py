from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import AlphaEdgeState
from src.orchestration.nodes import (
    classify_intent,
    run_sec_agent,
    run_openbb_agent,
    run_fred_agent,
    route_by_intent,
)


def create_graph():
    """Create the main AlphaEdge workflow graph."""
    
    workflow = StateGraph(AlphaEdgeState)
    
    # Add nodes
    workflow.add_node("classify", classify_intent)
    workflow.add_node("sec_agent", run_sec_agent)
    workflow.add_node("openbb_agent", run_openbb_agent)
    workflow.add_node("fred_agent", run_fred_agent)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "sec_agent": "sec_agent",
            "openbb_agent": "openbb_agent",
            "fred_agent": "fred_agent",
        }
    )
    
    # All agents go to END
    workflow.add_edge("sec_agent", END)
    workflow.add_edge("openbb_agent", END)
    workflow.add_edge("fred_agent", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Singleton instance
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
