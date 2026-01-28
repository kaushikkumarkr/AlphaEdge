"""Sprint 6: Test LangGraph Orchestration."""
print('=== SPRINT 6: Testing LangGraph Orchestration ===')
import asyncio

from src.orchestration.graph import create_graph
from src.orchestration.state import AlphaEdgeState
from src.models.mlx_model import MLXModel

# Create graph
graph = create_graph()
print('LangGraph created successfully')

# Test intent classification and routing
print('\n--- 6.1 Testing Intent Classification ---')

async def test_routing():
    # Test SEC filing query
    state = {
        "query": "What was Apple's revenue in their latest 10-K?",
        "filters": {"ticker": "AAPL"},
        "session_id": "test_session_1",
        "agent_outputs": [],
        "final_response": None,
        "error": None
    }
    
    # Invoke graph with thread config
    config = {"configurable": {"thread_id": "test_thread_1"}}
    result = await graph.ainvoke(state, config=config)
    
    print(f'Query: {state["query"]}')
    print(f'Agent outputs: {len(result.get("agent_outputs", []))}')
    if result.get("agent_outputs"):
        for output in result["agent_outputs"]:
            print(f'  - {output.agent_name}: {output.response_text[:100]}...')
    
    if result.get("final_response"):
        response = result["final_response"]
        if isinstance(response, str):
            print(f'Final response: {response[:200]}...')
        else:
            print(f'Final response: {response.answer[:200]}...')
    
    if result.get("error"):
        print(f'Error: {result["error"]}')
    
    return result

result = asyncio.run(test_routing())
# Check that graph ran and produced output
assert result.get("final_response") is not None, "Should have final_response"
print('\n✓ LangGraph Orchestration PASSED')

print('\n✅ SPRINT 6: LANGGRAPH TEST PASSED')
