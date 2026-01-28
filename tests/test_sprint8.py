"""Sprint 8: End-to-End Integration Test."""
print('=== SPRINT 8: End-to-End Integration Test ===')
import asyncio
from fastapi.testclient import TestClient
from src.api.main import app

# Add test data first
print('\n--- 8.1 Setting up test data ---')
from src.data.vector_store import VectorStore
from src.data.chunking import Chunk

vector_store = VectorStore()
test_chunks = [
    Chunk(
        text="Tesla Inc. (TSLA) reported total revenue of $96.8 billion for fiscal year 2023.",
        chunk_id="tsla_10k_rev",
        document_id="TSLA_10K_2023",
        metadata={"ticker": "TSLA", "filing_type": "10-K", "year": "2023"}
    ),
    Chunk(
        text="Tesla's automotive revenue was $82.4 billion, with energy generation and storage at $6.0 billion.",
        chunk_id="tsla_10k_seg",
        document_id="TSLA_10K_2023",
        metadata={"ticker": "TSLA", "filing_type": "10-K", "year": "2023"}
    ),
]
vector_store.add_documents(test_chunks)
print('Test SEC data added for TSLA')

# Test full flow via API
print('\n--- 8.2 Testing full query flow ---')
client = TestClient(app)

# Test SEC query
query_payload = {
    "query": "What was Tesla's revenue in 2023?",
    "ticker": "TSLA"
}
response = client.post("/query", json=query_payload)
print(f'Query: {query_payload["query"]}')
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Intent: {data.get("intent")}')
    print(f'Confidence: {data.get("confidence")}')
    print(f'Response: {data.get("response", "")[:300]}...')
    print(f'Citations: {len(data.get("citations", []))}')
else:
    print(f'Error: {response.text}')

assert response.status_code == 200, "Should return 200"
print('✓ SEC Query PASSED')

# Test stock price query
print('\n--- 8.3 Testing stock price query ---')
query_payload = {
    "query": "What is Tesla's current stock price?",
    "ticker": "TSLA"
}
response = client.post("/query", json=query_payload)
print(f'Query: {query_payload["query"]}')
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Intent: {data.get("intent")}')
    print(f'Response: {data.get("response", "")[:300]}...')
else:
    print(f'Error: {response.text}')

assert response.status_code == 200, "Should return 200"
print('✓ Stock Price Query PASSED')

print('\n✅ SPRINT 8: END-TO-END TESTS PASSED')
print('\n' + '='*50)
print('ALL SPRINTS COMPLETED SUCCESSFULLY!')
print('='*50)
