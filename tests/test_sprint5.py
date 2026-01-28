"""Sprint 5: Test FRED Agent."""
print('=== SPRINT 5: Testing FRED Agent ===')
import asyncio

from src.agents.fred_agent import FREDAgent
from src.models.mlx_model import MLXModel
from src.guardrails.schemas import AgentInput

# Create agent with MLX model
mlx_model = MLXModel('mlx-community/Qwen2.5-3B-Instruct-4bit')
agent = FREDAgent(model=mlx_model)

async def test_fred_agent():
    input_data = AgentInput(
        query="What is the current inflation rate in the US?",
        filters={}
    )
    result = await agent.execute(input_data)
    print(f'Agent: {result.agent_name}')
    print(f'Response: {result.response_text[:300]}...')
    print(f'Confidence: {result.confidence_score:.2f}')
    print(f'Processing time: {result.processing_time_ms}ms')
    return result

result = asyncio.run(test_fred_agent())
assert result.response_text, "Should have response"
print('✓ FRED Agent PASSED')

print('\n✅ SPRINT 5: FRED AGENT TEST PASSED')
