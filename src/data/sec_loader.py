from typing import List, Dict, Any
from pathlib import Path
import re
import html
from sec_edgar_downloader import Downloader
from src.data.chunking import DocumentChunker, Chunk
from src.data.vector_store import get_vector_store
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SECLoader:
    def __init__(self):
        email = settings.sec_user_agent.split()[-1] if settings.sec_user_agent else "research@example.com"
        self.downloader = Downloader("AlphaEdge", email)
        self.chunker = DocumentChunker()
        self.vector_store = get_vector_store()
    
    def download_filings(
        self,
        ticker: str,
        filing_types: List[str] = ["10-K", "10-Q"],
        limit: int = 5
    ) -> List[Path]:
        files = []
        for filing_type in filing_types:
            try:
                self.downloader.get(filing_type, ticker, limit=limit)
                # Find downloaded files
                base = Path(f"sec-edgar-filings/{ticker}/{filing_type}")
                if base.exists():
                    for f in base.rglob("*.htm*"):
                        files.append(f)
            except Exception as e:
                logger.error(f"Failed to download {filing_type} for {ticker}: {e}")
        return files
    
    def process_filing(self, file_path: Path, ticker: str) -> List[Chunk]:
        content = file_path.read_text(errors="ignore")
        # Clean HTML
        text = re.sub(r'<[^>]+>', ' ', content)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        
        doc_id = f"{ticker}-{file_path.parent.parent.name}-{file_path.parent.name}"
        
        return self.chunker.chunk_document(
            text=text,
            document_id=doc_id,
            metadata={"ticker": ticker, "filing_type": file_path.parent.parent.name}
        )
    
    def ingest(self, ticker: str, limit: int = 5) -> Dict[str, Any]:
        files = self.download_filings(ticker, limit=limit)
        all_chunks = []
        
        for f in files:
            chunks = self.process_filing(f, ticker)
            all_chunks.extend(chunks)
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
        
        return {"ticker": ticker, "files": len(files), "chunks": len(all_chunks)}
