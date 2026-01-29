from typing import Optional
from src.guardrails.schemas import AgentOutput, CriticOutput, FinalResponse
from src.agents.critic_agent import CriticAgent
from src.config.settings import settings


class ResponseValidator:
    """Validates agent responses through multiple checks."""
    
    def __init__(self):
        self.critic = CriticAgent()
        self.min_faithfulness = settings.min_faithfulness_score
        self.min_confidence = settings.min_confidence_score
    
    async def validate(self, output: AgentOutput) -> tuple[bool, CriticOutput]:
        """
        Run validation pipeline on agent output.
        
        Returns:
            (passed, critic_output)
        """
        # 1. Schema validation (already done by Pydantic)
        
        # 2. Rule-based checks
        if not self._rule_checks(output):
            return False, CriticOutput(
                faithfulness_score=0.0,
                citation_coverage=0.0,
                unsupported_claims=["Failed rule-based checks"],
                passed=False
            )
        
        # 3. Critic agent validation
        critic_output = await self.critic.validate(
            response_text=output.response_text,
            contexts=output.retrieved_contexts,
            citations=output.citations
        )
        
        # 4. Confidence threshold check
        passed = (
            critic_output.faithfulness_score >= self.min_faithfulness and
            output.confidence_score >= self.min_confidence and
            critic_output.passed
        )
        
        return passed, critic_output
    
    def _rule_checks(self, output: AgentOutput) -> bool:
        """Apply rule-based validation checks."""
        
        # Check 1: Response must not be empty
        if not output.response_text or len(output.response_text) < 10:
            return False
        
        # Check 2: Must have citations if contexts exist
        if output.retrieved_contexts and not output.citations:
            # Warning but don't fail
            pass
        
        # Check 3: No fabricated sources
        valid_source_ids = {ctx.source_id for ctx in output.retrieved_contexts}
        for citation in output.citations:
            if citation.source_id not in valid_source_ids:
                return False
        
        return True
    
    def create_final_response(
        self,
        query: str,
        output: AgentOutput,
        critic_output: CriticOutput,
        needs_review: bool = False
    ) -> FinalResponse:
        """Create the final validated response."""
        
        return FinalResponse(
            query=query,
            response=output.response_text,
            citations=output.citations,
            agents_used=[output.agent_name],
            faithfulness_score=critic_output.faithfulness_score,
            confidence_score=output.confidence_score,
            needs_human_review=needs_review or not critic_output.passed
        )
