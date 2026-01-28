"""Sprint 2: Test Data Layer Components."""
print('=== SPRINT 2: Testing Data Layer ===')

# Test 2.1: Embeddings
print('\n--- 2.1 Testing Embeddings ---')
from src.data.embeddings import EmbeddingModel

embed_model = EmbeddingModel()
test_texts = ["Apple reported strong earnings", "Federal Reserve raises rates"]
embeddings = embed_model.embed_documents(test_texts)
print(f'Generated {len(embeddings)} embeddings')
print(f'Embedding dimension: {len(embeddings[0])}')
assert len(embeddings) == 2, "Should have 2 embeddings"
assert len(embeddings[0]) == 768, "BGE-base should have 768 dimensions"
print('✓ Embeddings PASSED')

# Test 2.2: Chunking
print('\n--- 2.2 Testing Chunking ---')
from src.data.chunking import DocumentChunker

chunker = DocumentChunker(chunk_size=100, overlap=20)
test_text = "This is a test document. " * 50
chunks = chunker.chunk_document(test_text, document_id="test_doc")
print(f'Created {len(chunks)} chunks from test text')
assert len(chunks) > 1, "Should create multiple chunks"
print('✓ Chunking PASSED')

# Test 2.3: Vector Store
print('\n--- 2.3 Testing Vector Store ---')
from src.data.vector_store import VectorStore
from src.data.chunking import Chunk

vector_store = VectorStore()
test_chunks = [
    Chunk(
        text="Apple Inc reported Q4 revenue of $90 billion",
        chunk_id="test_chunk_1",
        document_id="test_doc_1",
        metadata={"ticker": "AAPL", "type": "10-K"}
    ),
    Chunk(
        text="Microsoft cloud revenue grew 25% year over year",
        chunk_id="test_chunk_2",
        document_id="test_doc_2",
        metadata={"ticker": "MSFT", "type": "10-K"}
    )
]
vector_store.add_documents(test_chunks)
print(f'Added {len(test_chunks)} chunks to vector store')

results = vector_store.search("Apple earnings report", top_k=1)
print(f'Search returned {len(results)} results')
assert len(results) >= 1, "Should find at least 1 result"
print(f'Top result: {results[0][0].text[:50]}...')
print('✓ Vector Store PASSED')

print('\n✅ SPRINT 2: DATA LAYER ALL TESTS PASSED')
