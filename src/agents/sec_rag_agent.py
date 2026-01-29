import re
import asyncio
from typing import List, Dict, Tuple
from src.agents.base_agent import BaseAgent
from src.data.sec_loader import SECLoader
from src.data.vector_store import get_vector_store
from src.guardrails.schemas import RetrievedContext, Citation
from src.config.constants import AgentName, RERANK_TOP_K
from src.utils.logging import get_logger


class SECRAGAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name=AgentName.SEC_RAG, **kwargs)
        self.vector_store = get_vector_store(use_http=False)
        self.sec_loader = SECLoader()
        self._ingested_tickers = set()
        self._logger = get_logger(__name__)
    
    @property
    def system_prompt(self) -> str:
        return """You are an SEC filing analyst. Analyze regulatory filings and provide accurate, cited responses.

Rules:
1. Only use information from the provided sources
2. Cite sources using [Source N] format
3. Use exact figures from sources
4. If information is not in sources, say so clearly
5. Never make up information"""
    
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        search_filters = {}
        if filters.get("ticker"):
            search_filters["ticker"] = filters["ticker"]
        else:
            extracted = self._extract_ticker_from_query(query)
            if extracted:
                search_filters["ticker"] = extracted
        
        results = self.vector_store.search(
            query=query,
            top_k=RERANK_TOP_K,
            filters=search_filters if search_filters else None
        )

        if not results and search_filters.get("ticker"):
            ticker = str(search_filters["ticker"]).upper().strip()
            await self._ingest_on_demand(ticker)
            results = self.vector_store.search(
                query=query,
                top_k=RERANK_TOP_K,
                filters=search_filters
            )
        
        return [
            RetrievedContext(
                source_id=chunk.document_id,
                text=chunk.text,
                relevance_score=score,
                metadata=chunk.metadata
            )
            for chunk, score in results
        ]

    def _extract_ticker_from_query(self, query: str) -> str | None:
        match = re.search(r"\(([A-Z]{1,5})\)", query.upper())
        if match:
            return match.group(1)
        match = re.search(r"\$([A-Z]{1,5})\b", query.upper())
        if match:
            return match.group(1)
        return None

    async def _ingest_on_demand(self, ticker: str, limit: int = 3) -> None:
        if not ticker or ticker in self._ingested_tickers:
            return
        self._ingested_tickers.add(ticker)
        self._logger.info(f"On-demand SEC ingestion for {ticker} (limit={limit})")
        try:
            await asyncio.to_thread(self.sec_loader.ingest, ticker, limit)
        except Exception as exc:
            self._logger.error(f"On-demand SEC ingestion failed for {ticker}: {exc}")
    
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        if not contexts:
            return "No relevant SEC filing information found.", []
        
        context_text = self._format_contexts(contexts)
        
        prompt = f"""Question: {query}

Sources:
{context_text}

Answer the question using only the sources above. Cite using [Source N]."""

        response = await self.model.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3
        )
        
        citations = self._extract_citations(response.content, contexts)
        return response.content, citations
    
    def _extract_citations(
        self,
        response: str,
        contexts: List[RetrievedContext]
    ) -> List[Citation]:
        citations = []
        refs = set(re.findall(r'\[Source (\d+)\]', response))
        
        for ref in refs:
            idx = int(ref) - 1
            if 0 <= idx < len(contexts):
                ctx = contexts[idx]
                citations.append(Citation(
                    source_type="sec_filing",
                    source_id=ctx.source_id,
                    text_excerpt=ctx.text[:300],
                    relevance_score=ctx.relevance_score
                ))
        
        return citations
