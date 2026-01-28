from src.orchestration.state import AlphaEdgeState
from src.agents.sec_rag_agent import SECRAGAgent
from src.agents.openbb_agent import OpenBBAgent
from src.agents.fred_agent import FREDAgent
from src.guardrails.schemas import AgentInput
from src.models.mlx_model import get_mlx_model
from src.config.constants import Intent
from src.utils.telemetry import get_tracer
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
import json


# Shared MLX model instance
_model = None
_tracer = get_tracer("orchestration.nodes")


def get_model():
    """Get or create shared MLX model."""
    global _model
    if _model is None:
        _model = get_mlx_model()
    return _model


async def classify_intent(state: AlphaEdgeState) -> AlphaEdgeState:
    """Classify the query intent with tracing."""
    with _tracer.start_as_current_span("orchestration.classify_intent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, state['query'])
        
        model = get_model()
        
        prompt = f"""Classify this query into one category:
- SEC_FILING: Questions about company filings, 10-K, 10-Q, risk factors
- FINANCIALS: Stock prices, ratios, earnings, metrics
- MACRO: GDP, unemployment, inflation, interest rates
- SYNTHESIS: Needs multiple data sources

Query: {state['query']}

Respond with just the category name."""

        response = await model.generate(prompt, temperature=0.1)
        intent = response.content.strip().upper()
        
        # Map to constants
        intent_map = {
            "SEC_FILING": Intent.SEC_FILING,
            "FINANCIALS": Intent.FINANCIALS,
            "MACRO": Intent.MACRO,
            "SYNTHESIS": Intent.SYNTHESIS,
        }
        
        classified_intent = intent_map.get(intent, Intent.SEC_FILING)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, classified_intent.value if hasattr(classified_intent, 'value') else str(classified_intent))
        span.set_attribute("intent.raw_response", intent)
        span.set_attribute("intent.classified", str(classified_intent))
        
        return {**state, "intent": classified_intent}


async def run_sec_agent(state: AlphaEdgeState) -> AlphaEdgeState:
    """Execute SEC RAG agent with tracing."""
    with _tracer.start_as_current_span("orchestration.run_sec_agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, state["query"])
        span.set_attribute("agent.type", "sec_rag")
        
        model = get_model()
        agent = SECRAGAgent(model=model)
        
        # Extract ticker from entities or query
        ticker = state.get("entities", {}).get("ticker") or state.get("filters", {}).get("ticker")
        span.set_attribute("query.ticker", ticker or "none")
        
        result = await agent.execute(AgentInput(
            query=state["query"],
            filters={"ticker": ticker} if ticker else {}
        ))
        
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.response_text[:500] if result.response_text else "")
        span.set_attribute("agent.confidence_score", result.confidence_score)
        span.set_attribute("agent.citation_count", len(result.citations))
        
        return {
            **state,
            "sec_results": [result.model_dump()],
            "final_response": result.response_text,
            "citations": [c.model_dump() for c in result.citations],
            "confidence_score": result.confidence_score,
        }


async def run_openbb_agent(state: AlphaEdgeState) -> AlphaEdgeState:
    """Execute OpenBB agent with tracing."""
    with _tracer.start_as_current_span("orchestration.run_openbb_agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, state["query"])
        span.set_attribute("agent.type", "openbb")
        
        model = get_model()
        agent = OpenBBAgent(model=model)
        ticker = state.get("entities", {}).get("ticker") or state.get("filters", {}).get("ticker")
        span.set_attribute("query.ticker", ticker or "none")
        
        result = await agent.execute(AgentInput(
            query=state["query"],
            filters={"ticker": ticker} if ticker else {}
        ))
        
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.response_text[:500] if result.response_text else "")
        span.set_attribute("agent.confidence_score", result.confidence_score)
        span.set_attribute("agent.citation_count", len(result.citations))
        
        return {
            **state,
            "openbb_results": [result.model_dump()],
            "final_response": result.response_text,
            "citations": [c.model_dump() for c in result.citations],
            "confidence_score": result.confidence_score,
        }


async def run_fred_agent(state: AlphaEdgeState) -> AlphaEdgeState:
    """Execute FRED agent with tracing."""
    with _tracer.start_as_current_span("orchestration.run_fred_agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, state["query"])
        span.set_attribute("agent.type", "fred")
        
        model = get_model()
        agent = FREDAgent(model=model)
        
        result = await agent.execute(AgentInput(query=state["query"]))
        
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.response_text[:500] if result.response_text else "")
        span.set_attribute("agent.confidence_score", result.confidence_score)
        span.set_attribute("agent.citation_count", len(result.citations))
        
        return {
            **state,
            "fred_results": [result.model_dump()],
            "final_response": result.response_text,
            "citations": [c.model_dump() for c in result.citations],
            "confidence_score": result.confidence_score,
        }


def route_by_intent(state: AlphaEdgeState) -> str:
    """Route to appropriate agent based on intent with tracing."""
    with _tracer.start_as_current_span("orchestration.route_by_intent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        
        intent = state.get("intent", Intent.SEC_FILING)
        
        routes = {
            Intent.SEC_FILING: "sec_agent",
            Intent.FINANCIALS: "openbb_agent",
            Intent.MACRO: "fred_agent",
            Intent.SYNTHESIS: "sec_agent",  # Start with SEC for synthesis
        }
        
        route = routes.get(intent, "sec_agent")
        span.set_attribute("routing.intent", str(intent))
        span.set_attribute("routing.destination", route)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, route)
        
        return route
