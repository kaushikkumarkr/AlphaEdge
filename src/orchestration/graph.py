from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import AlphaEdgeState
from src.orchestration.nodes import (
    classify_intent,
    run_sec_agent,
    run_openbb_agent,
    run_fred_agent,
    route_by_intent,
    decompose_query,
    execute_tasks,
    aggregate_results,
)


def create_graph():
    """Create the main AlphaEdge workflow graph with multi-task support."""
    
    workflow = StateGraph(AlphaEdgeState)
    
    # Add nodes - simple routing
    workflow.add_node("classify", classify_intent)
    workflow.add_node("sec_agent", run_sec_agent)
    workflow.add_node("openbb_agent", run_openbb_agent)
    workflow.add_node("fred_agent", run_fred_agent)
    
    # Add nodes - multi-task pipeline
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("execute_tasks", execute_tasks)
    workflow.add_node("aggregate_results", aggregate_results)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional routing (handles both simple and complex queries)
    workflow.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            # Simple routing (fast path)
            "sec_agent": "sec_agent",
            "openbb_agent": "openbb_agent",
            "fred_agent": "fred_agent",
            # Multi-task routing (complex queries)
            "decompose_query": "decompose_query",
        }
    )
    
    # Simple agents go to END
    workflow.add_edge("sec_agent", END)
    workflow.add_edge("openbb_agent", END)
    workflow.add_edge("fred_agent", END)
    
    # Multi-task pipeline flow
    workflow.add_edge("decompose_query", "execute_tasks")
    workflow.add_edge("execute_tasks", "aggregate_results")
    workflow.add_edge("aggregate_results", END)
    
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
