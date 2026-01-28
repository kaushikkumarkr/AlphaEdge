from typing import List, Dict, Tuple
from src.agents.base_agent import BaseAgent
from src.guardrails.schemas import RetrievedContext, Citation, AgentOutput
from src.config.constants import AgentName


class SynthesisAgent(BaseAgent):
    """Agent that synthesizes information from multiple sources."""
    
    def __init__(self, **kwargs):
        super().__init__(name=AgentName.SYNTHESIS, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are a synthesis analyst. Combine information from multiple data sources into a coherent analysis.

Rules:
1. Integrate SEC filing data, financial metrics, and macro indicators
2. Identify correlations and connections between data points
3. Cite all sources using [Source N] format
4. Highlight areas of uncertainty
5. Provide actionable insights"""
    
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        # Synthesis agent receives contexts from other agents
        return []
    
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        if not contexts:
            return "No data available for synthesis.", []
        
        context_text = self._format_contexts(contexts)
        
        prompt = f"""Question: {query}

Available Data:
{context_text}

Synthesize this information into a comprehensive analysis. Cite sources."""

        response = await self.model.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.4
        )
        
        citations = [
            Citation(
                source_type=ctx.metadata.get("type", "sec_filing"),
                source_id=ctx.source_id,
                text_excerpt=ctx.text[:300],
                relevance_score=ctx.relevance_score
            )
            for ctx in contexts
        ]
        
        return response.content, citations
    
    async def synthesize(
        self,
        query: str,
        agent_outputs: List[AgentOutput]
    ) -> AgentOutput:
        """Synthesize outputs from multiple agents."""
        import time
        start = time.time()
        
        # Combine all contexts from agent outputs
        all_contexts = []
        for output in agent_outputs:
            all_contexts.extend(output.retrieved_contexts)
        
        response_text, citations = await self._generate(query, all_contexts)
        
        # Calculate combined confidence
        if agent_outputs:
            avg_confidence = sum(o.confidence_score for o in agent_outputs) / len(agent_outputs)
        else:
            avg_confidence = 0.0
        
        return AgentOutput(
            agent_name=self.name,
            response_text=response_text,
            citations=citations,
            retrieved_contexts=all_contexts,
            confidence_score=avg_confidence,
            processing_time_ms=int((time.time() - start) * 1000)
        )
