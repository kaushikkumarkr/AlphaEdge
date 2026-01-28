"""Sprint 3: Test Agents with MLX Model."""
print('=== SPRINT 3: Testing Agents ===')
import asyncio

# Test 3.1: SEC RAG Agent
print('\n--- 3.1 Testing SEC RAG Agent ---')
from src.agents.sec_rag_agent import SECRAGAgent
from src.models.mlx_model import MLXModel
from src.guardrails.schemas import AgentInput

# First, add some test data to vector store
from src.data.vector_store import VectorStore
from src.data.chunking import Chunk

print('Adding test SEC data to vector store...')
vector_store = VectorStore()
test_chunks = [
    Chunk(
        text="Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023, a decrease from $394.3 billion in 2022.",
        chunk_id="aapl_10k_2023_rev",
        document_id="AAPL_10K_2023",
        metadata={"ticker": "AAPL", "filing_type": "10-K", "year": "2023"}
    ),
    Chunk(
        text="iPhone revenue was $200.6 billion, representing 52% of total revenue. Services segment revenue reached $85.2 billion.",
        chunk_id="aapl_10k_2023_seg",
        document_id="AAPL_10K_2023",
        metadata={"ticker": "AAPL", "filing_type": "10-K", "year": "2023"}
    ),
]
vector_store.add_documents(test_chunks)
print('Test data added.')

# Create agent with MLX model
mlx_model = MLXModel('mlx-community/Qwen2.5-3B-Instruct-4bit')
agent = SECRAGAgent(model=mlx_model)

async def test_sec_agent():
    input_data = AgentInput(
        query="What was Apple's total revenue in 2023?",
        filters={"ticker": "AAPL"}
    )
    result = await agent.execute(input_data)
    print(f'Agent: {result.agent_name}')
    print(f'Response: {result.response_text[:200]}...')
    print(f'Confidence: {result.confidence_score:.2f}')
    print(f'Citations: {len(result.citations)}')
    print(f'Processing time: {result.processing_time_ms}ms')
    return result

result = asyncio.run(test_sec_agent())
assert result.response_text, "Should have response"
assert "383" in result.response_text or "billion" in result.response_text.lower(), "Should mention revenue figure"
print('✓ SEC RAG Agent PASSED')

print('\n✅ SPRINT 3: AGENTS TEST PASSED')
