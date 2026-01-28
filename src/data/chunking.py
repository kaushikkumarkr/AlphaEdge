from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.constants import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any]


class DocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # ~4 chars per token
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        texts = self.splitter.split_text(text)
        return [
            Chunk(
                text=t,
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                metadata=metadata or {}
            )
            for i, t in enumerate(texts)
        ]
