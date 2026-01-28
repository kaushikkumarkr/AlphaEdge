from typing import List, Dict, Any, Optional, Tuple
import chromadb
from src.config.settings import settings
from src.data.embeddings import get_embedding_model
from src.data.chunking import Chunk
from src.utils.telemetry import get_tracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace


class VectorStore:
    def __init__(self, use_http: bool = False):
        self.embedding_model = get_embedding_model()
        
        if use_http:
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port
            )
        else:
            self.client = chromadb.PersistentClient(path="./data/vectordb")
        
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Chunk]):
        """Add documents with embedding tracing."""
        if not chunks:
            return
        
        tracer = get_tracer()
        
        with tracer.start_as_current_span(
            "embedding.add_documents",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "EMBEDDING")
            span.set_attribute("embedding.document_count", len(chunks))
            span.set_attribute("embedding.model", self.embedding_model.model_name)
            
            texts = [c.text for c in chunks]
            ids = [c.chunk_id for c in chunks]
            embeddings = self.embedding_model.embed_documents(texts).tolist()
            metadatas = [
                {"document_id": c.document_id, **c.metadata}
                for c in chunks
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            span.set_attribute("embedding.dimension", len(embeddings[0]) if embeddings else 0)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search with retrieval tracing."""
        tracer = get_tracer()
        
        with tracer.start_as_current_span(
            "retriever.vector_search",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)
            span.set_attribute("retriever.top_k", top_k)
            if filters:
                span.set_attribute("retriever.filters", str(filters))
            
            # Embed query with tracing
            with tracer.start_as_current_span(
                "embedding.query",
                kind=trace.SpanKind.INTERNAL
            ) as embed_span:
                embed_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "EMBEDDING")
                embed_span.set_attribute("embedding.model", self.embedding_model.model_name)
                query_embedding = self.embedding_model.embed_query(query).tolist()
            
            where = None
            if filters:
                where = {k: v for k, v in filters.items()}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            for i, chunk_id in enumerate(results["ids"][0]):
                chunk = Chunk(
                    text=results["documents"][0][i],
                    chunk_id=chunk_id,
                    document_id=results["metadatas"][0][i].get("document_id", ""),
                    metadata=results["metadatas"][0][i]
                )
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                chunks.append((chunk, score))
            
            span.set_attribute("retriever.document_count", len(chunks))
            if chunks:
                span.set_attribute("retriever.top_score", chunks[0][1])
            
            return chunks


def get_vector_store(use_http: bool = False) -> VectorStore:
    return VectorStore(use_http=use_http)
