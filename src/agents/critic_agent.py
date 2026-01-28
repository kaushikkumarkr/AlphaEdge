from typing import List, Dict, Tuple
from src.agents.base_agent import BaseAgent
from src.guardrails.schemas import (
    RetrievedContext, Citation, AgentOutput, CriticOutput
)
from src.config.constants import AgentName


class CriticAgent(BaseAgent):
    """Agent that validates responses for faithfulness and citation coverage."""
    
    def __init__(self, **kwargs):
        super().__init__(name=AgentName.CRITIC, **kwargs)
    
    @property
    def system_prompt(self) -> str:
        return """You are a validation agent. Your job is to check if a response is faithful to its sources.

For each claim in the response, verify:
1. Is it supported by the provided sources?
2. Are citations correct?
3. Are there any unsupported claims?

Respond in this exact format:
FAITHFULNESS: [0.0-1.0]
CITATION_COVERAGE: [0.0-1.0]
UNSUPPORTED_CLAIMS: [list of claims without source support]
PASSED: [true/false]"""
    
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        # Critic doesn't retrieve - it receives contexts
        return []
    
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        return "", []
    
    async def validate(
        self,
        response_text: str,
        contexts: List[RetrievedContext],
        citations: List[Citation]
    ) -> CriticOutput:
        """Validate a response against its sources."""
        
        if not contexts:
            return CriticOutput(
                faithfulness_score=0.0,
                citation_coverage=0.0,
                unsupported_claims=["No sources provided"],
                passed=False
            )
        
        # Build context text for validation
        context_text = "\n".join([
            f"[Source {i+1}]: {ctx.text[:500]}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Validate this response against its sources.

RESPONSE:
{response_text}

SOURCES:
{context_text}

Analyze faithfulness and citation coverage."""

        result = await self.model.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1
        )
        
        # Parse response
        return self._parse_critic_response(result.content)
    
    def _parse_critic_response(self, response: str) -> CriticOutput:
        """Parse the critic's structured response."""
        lines = response.strip().split("\n")
        
        faithfulness = 0.8
        coverage = 0.8
        unsupported = []
        passed = True
        
        for line in lines:
            line = line.strip()
            if line.startswith("FAITHFULNESS:"):
                try:
                    faithfulness = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CITATION_COVERAGE:"):
                try:
                    coverage = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("UNSUPPORTED_CLAIMS:"):
                claims_str = line.split(":", 1)[1].strip()
                if claims_str and claims_str != "[]":
                    unsupported = [c.strip() for c in claims_str.strip("[]").split(",")]
            elif line.startswith("PASSED:"):
                passed = "true" in line.lower()
        
        return CriticOutput(
            faithfulness_score=faithfulness,
            citation_coverage=coverage,
            unsupported_claims=unsupported,
            passed=passed
        )
