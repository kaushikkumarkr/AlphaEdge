from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.query_prefix = "Represent this sentence for searching relevant passages: "
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            self.query_prefix + query,
            normalize_embeddings=True
        )


_model = None

def get_embedding_model() -> EmbeddingModel:
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model
