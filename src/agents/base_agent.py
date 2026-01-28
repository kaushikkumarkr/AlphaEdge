from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time
from src.models.base import BaseModelInterface
from src.models.openai_model import OpenAIModel
from src.guardrails.schemas import (
    AgentInput, AgentOutput, Citation, RetrievedContext
)
from src.utils.logging import get_logger
from src.utils.telemetry import get_tracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

logger = get_logger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str, model: Optional[BaseModelInterface] = None):
        self.name = name
        self.model = model or OpenAIModel()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass
    
    @abstractmethod
    async def _retrieve(self, query: str, filters: Dict) -> List[RetrievedContext]:
        pass
    
    @abstractmethod
    async def _generate(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, List[Citation]]:
        pass
    
    async def execute(self, input: AgentInput) -> AgentOutput:
        """Execute agent with full OpenInference tracing."""
        tracer = get_tracer()
        
        with tracer.start_as_current_span(
            f"agent.{self.name}",
            kind=trace.SpanKind.INTERNAL
        ) as span:
            # Set OpenInference agent attributes
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
            span.set_attribute("agent.name", self.name)
            span.set_attribute(SpanAttributes.INPUT_VALUE, input.query)
            span.set_attribute("agent.filters", str(input.filters))
            
            start = time.time()
            
            # Retrieval step with tracing
            with tracer.start_as_current_span(
                f"retriever.{self.name}",
                kind=trace.SpanKind.INTERNAL
            ) as retriever_span:
                retriever_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
                retriever_span.set_attribute(SpanAttributes.INPUT_VALUE, input.query)
                
                contexts = await self._retrieve(input.query, input.filters)
                
                retriever_span.set_attribute("retriever.document_count", len(contexts))
                if contexts:
                    # Add document details
                    for i, ctx in enumerate(contexts[:5]):  # Limit to first 5
                        retriever_span.set_attribute(f"retriever.document.{i}.id", ctx.source_id)
                        retriever_span.set_attribute(f"retriever.document.{i}.score", ctx.relevance_score)
            
            # Generation step (already traced in model)
            response_text, citations = await self._generate(input.query, contexts)
            
            confidence = self._calculate_confidence(contexts, citations)
            processing_time = int((time.time() - start) * 1000)
            
            # Set output attributes
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response_text[:2000])
            span.set_attribute("agent.confidence_score", confidence)
            span.set_attribute("agent.citation_count", len(citations))
            span.set_attribute("agent.context_count", len(contexts))
            span.set_attribute("agent.processing_time_ms", processing_time)
            
            return AgentOutput(
                agent_name=self.name,
                response_text=response_text,
                citations=citations,
                retrieved_contexts=contexts,
                confidence_score=confidence,
                processing_time_ms=processing_time
            )
    
    def _calculate_confidence(
        self,
        contexts: List[RetrievedContext],
        citations: List[Citation]
    ) -> float:
        if not contexts:
            return 0.0
        avg_relevance = sum(c.relevance_score for c in contexts) / len(contexts)
        citation_ratio = min(len(citations) / len(contexts), 1.0)
        return round(avg_relevance * 0.6 + citation_ratio * 0.4, 3)
    
    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant context found."
        parts = []
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"[Source {i}] ({ctx.source_id}):\n{ctx.text}\n")
        return "\n".join(parts)
