"""Sprint 4: Test OpenBB Agent."""
print('=== SPRINT 4: Testing OpenBB Agent ===')
import asyncio

from src.agents.openbb_agent import OpenBBAgent
from src.models.mlx_model import MLXModel
from src.guardrails.schemas import AgentInput

# Create agent with MLX model
mlx_model = MLXModel('mlx-community/Qwen2.5-3B-Instruct-4bit')
agent = OpenBBAgent(model=mlx_model)

async def test_openbb_agent():
    input_data = AgentInput(
        query="What is Apple's current stock price?",
        filters={"ticker": "AAPL"}
    )
    result = await agent.execute(input_data)
    print(f'Agent: {result.agent_name}')
    print(f'Response: {result.response_text[:300]}...')
    print(f'Confidence: {result.confidence_score:.2f}')
    print(f'Processing time: {result.processing_time_ms}ms')
    return result

result = asyncio.run(test_openbb_agent())
assert result.response_text, "Should have response"
print('✓ OpenBB Agent PASSED')

print('\n✅ SPRINT 4: OpenBB AGENT TEST PASSED')
