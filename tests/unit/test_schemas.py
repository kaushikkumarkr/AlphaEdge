import pytest
from pydantic import ValidationError
from src.guardrails.schemas import (
    Citation,
    RetrievedContext,
    AgentInput,
    AgentOutput,
    CriticOutput,
    FinalResponse,
)


class TestCitation:
    def test_valid_citation(self):
        citation = Citation(
            source_type="sec_filing",
            source_id="AAPL-10K-2023",
            text_excerpt="Apple reported revenue",
            relevance_score=0.95
        )
        assert citation.source_type == "sec_filing"
        assert citation.relevance_score == 0.95
    
    def test_invalid_source_type(self):
        with pytest.raises(ValidationError):
            Citation(
                source_type="invalid_type",
                source_id="test",
                text_excerpt="test",
                relevance_score=0.5
            )
    
    def test_invalid_relevance_score(self):
        with pytest.raises(ValidationError):
            Citation(
                source_type="sec_filing",
                source_id="test",
                text_excerpt="test",
                relevance_score=1.5  # > 1
            )


class TestRetrievedContext:
    def test_valid_context(self):
        ctx = RetrievedContext(
            source_id="AAPL-10K",
            text="Apple's revenue was $383 billion",
            relevance_score=0.9,
            metadata={"ticker": "AAPL"}
        )
        assert ctx.source_id == "AAPL-10K"
        assert ctx.metadata["ticker"] == "AAPL"
    
    def test_default_metadata(self):
        ctx = RetrievedContext(
            source_id="test",
            text="test",
            relevance_score=0.5
        )
        assert ctx.metadata == {}


class TestAgentInput:
    def test_valid_input(self):
        inp = AgentInput(
            query="What is Apple's revenue?",
            filters={"ticker": "AAPL"}
        )
        assert inp.query == "What is Apple's revenue?"
    
    def test_empty_query_fails(self):
        with pytest.raises(ValidationError):
            AgentInput(query="")
    
    def test_optional_fields(self):
        inp = AgentInput(query="test")
        assert inp.conversation_id is None
        assert inp.context is None
        assert inp.filters == {}


class TestAgentOutput:
    def test_valid_output(self):
        output = AgentOutput(
            agent_name="sec_rag_agent",
            response_text="Apple had revenue of $383 billion",
            citations=[],
            retrieved_contexts=[],
            confidence_score=0.85,
            processing_time_ms=150
        )
        assert output.agent_name == "sec_rag_agent"
        assert output.confidence_score == 0.85
    
    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                agent_name="test",
                response_text="test",
                confidence_score=-0.1,  # < 0
                processing_time_ms=100
            )


class TestCriticOutput:
    def test_valid_critic_output(self):
        output = CriticOutput(
            faithfulness_score=0.9,
            citation_coverage=0.95,
            unsupported_claims=[],
            passed=True
        )
        assert output.passed is True
        assert len(output.unsupported_claims) == 0
    
    def test_failed_validation(self):
        output = CriticOutput(
            faithfulness_score=0.5,
            citation_coverage=0.3,
            unsupported_claims=["Claim about revenue growth"],
            passed=False
        )
        assert output.passed is False
        assert len(output.unsupported_claims) == 1


class TestFinalResponse:
    def test_valid_final_response(self):
        response = FinalResponse(
            query="What is Apple's revenue?",
            response="Apple's revenue was $383 billion",
            citations=[],
            agents_used=["sec_rag_agent"],
            faithfulness_score=0.9,
            confidence_score=0.85,
            needs_human_review=False
        )
        assert response.needs_human_review is False
        assert "sec_rag_agent" in response.agents_used
