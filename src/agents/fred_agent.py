from typing import List, Dict, Tuple
from fredapi import Fred
from src.agents.base_agent import BaseAgent
from src.guardrails.schemas import RetrievedContext, Citation
from src.config.settings import settings
from src.config.constants import AgentName, FRED_SERIES


class FREDAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name=AgentName.FRED, **kwargs)
        api_key = settings.fred_api_key.get_secret_value() if settings.fred_api_key else None
        self.fred = Fred(api_key=api_key) if api_key else None
    
    @property
    def system_prompt(self) -> str:
        return """You are a macroeconomic analyst. Analyze economic indicators from FRED.

Rules:
1. Report exact values and dates
2. Explain trends and implications
3. Compare to historical averages when relevant
4. Note data limitations"""
    
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        if not self.fred:
            return []
        
        contexts = []
        query_lower = query.lower()
        
        for keyword, series_id in FRED_SERIES.items():
            if keyword in query_lower:
                try:
                    data = self.fred.get_series(series_id, observation_start="2020-01-01")
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    
                    text = f"Indicator: {series_id}\nLatest Value: {latest:.2f}\nPrevious: {prev:.2f}\nDate: {data.index[-1].strftime('%Y-%m-%d')}"
                    
                    contexts.append(RetrievedContext(
                        source_id=f"FRED-{series_id}",
                        text=text,
                        relevance_score=0.9,
                        metadata={"series_id": series_id}
                    ))
                except Exception:
                    pass
        
        return contexts
    
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        if not contexts:
            return "No relevant economic data found. Try asking about GDP, unemployment, inflation, or interest rates.", []
        
        context_text = self._format_contexts(contexts)
        
        prompt = f"""Question: {query}

Economic Data:
{context_text}

Analyze this data to answer the question."""

        response = await self.model.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3
        )
        
        citations = [
            Citation(
                source_type="macro_data",
                source_id=ctx.source_id,
                text_excerpt=ctx.text[:300],
                relevance_score=ctx.relevance_score
            )
            for ctx in contexts
        ]
        
        return response.content, citations
