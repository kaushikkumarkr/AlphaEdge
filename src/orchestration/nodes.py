from src.orchestration.state import AlphaEdgeState
from src.agents.sec_rag_agent import SECRAGAgent
from src.agents.openbb_agent import OpenBBAgent
from src.agents.fred_agent import FREDAgent
from src.guardrails.schemas import AgentInput
from src.models.mlx_model import get_mlx_model
from src.config.constants import Intent
from src.orchestration.complexity import ComplexityDetector
from src.utils.telemetry import get_tracer
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
import json


# Shared MLX model instance
_model = None
_decomposer = None
_task_executor = None
_tracer = get_tracer("orchestration.nodes")


def get_model():
    """Get or create shared MLX model."""
    global _model
    if _model is None:
        _model = get_mlx_model()
    return _model


def get_decomposer():
    """Get or create shared QueryDecomposer instance."""
    global _decomposer
    if _decomposer is None:
        from src.orchestration.decomposer import QueryDecomposer
        _decomposer = QueryDecomposer(model=get_model())
    return _decomposer


def get_task_executor():
    """Get or create shared TaskExecutor instance."""
    global _task_executor
    if _task_executor is None:
        from src.orchestration.task_executor import TaskExecutor
        _task_executor = TaskExecutor()
    return _task_executor


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
        
        # Check query complexity
        is_complex = ComplexityDetector.is_complex(state['query'], classified_intent)
        complexity_score = ComplexityDetector.get_complexity_score(state['query'])
        
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, classified_intent.value if hasattr(classified_intent, 'value') else str(classified_intent))
        span.set_attribute("intent.raw_response", intent)
        span.set_attribute("intent.classified", str(classified_intent))
        span.set_attribute("complexity.is_complex", is_complex)
        span.set_attribute("complexity.score", complexity_score)
        
        return {**state, "intent": classified_intent, "is_complex": is_complex}


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
    """Route to appropriate agent or multi-task pipeline with tracing."""
    with _tracer.start_as_current_span("orchestration.route_by_intent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        
        is_complex = state.get("is_complex", False)
        intent = state.get("intent", Intent.SEC_FILING)
        
        # Complex queries go to multi-task pipeline
        if is_complex:
            span.set_attribute("routing.type", "multi_task")
            span.set_attribute("routing.destination", "decompose_query")
            return "decompose_query"
        
        # Simple queries use direct routing
        routes = {
            Intent.SEC_FILING: "sec_agent",
            Intent.FINANCIALS: "openbb_agent",
            Intent.MACRO: "fred_agent",
            Intent.SYNTHESIS: "decompose_query",  # Synthesis also goes to multi-task
        }
        
        route = routes.get(intent, "sec_agent")
        span.set_attribute("routing.type", "simple")
        span.set_attribute("routing.intent", str(intent))
        span.set_attribute("routing.destination", route)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, route)
        
        return route


async def decompose_query(state: AlphaEdgeState) -> AlphaEdgeState:
    """Decompose complex query into task plan."""
    with _tracer.start_as_current_span("orchestration.decompose_query") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, state["query"])
        
        decomposer = get_decomposer()
        task_plan = await decomposer.decompose(state["query"])
        
        span.set_attribute("task_plan.task_count", len(task_plan.tasks))
        span.set_attribute("task_plan.can_parallelize", task_plan.can_parallelize)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, task_plan.reasoning)
        
        return {
            **state,
            "task_plan": task_plan.model_dump(),
            "task_results": {},
            "completed_tasks": []
        }


async def execute_tasks(state: AlphaEdgeState) -> AlphaEdgeState:
    """Execute task plan with parallel execution."""
    with _tracer.start_as_current_span("orchestration.execute_tasks") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        
        from src.orchestration.decomposer import TaskPlan
        
        task_plan_data = state.get("task_plan")
        if not task_plan_data:
            return {**state, "error": "No task plan available"}
        
        task_plan = TaskPlan.model_validate(task_plan_data)
        
        executor = get_task_executor()
        task_results = await executor.execute_plan(task_plan, state["query"])
        
        completed_tasks = list(task_results.keys())
        
        span.set_attribute("execution.completed_count", len(completed_tasks))
        span.set_attribute("execution.failed_count", sum(
            1 for r in task_results.values() if r.get("status") == "failed"
        ))
        
        return {
            **state,
            "task_results": task_results,
            "completed_tasks": completed_tasks
        }


async def aggregate_results(state: AlphaEdgeState) -> AlphaEdgeState:
    """Aggregate multi-task results into final response."""
    with _tracer.start_as_current_span("orchestration.aggregate_results") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        
        task_results = state.get("task_results", {})
        
        if not task_results:
            return {
                **state,
                "final_response": "No results available.",
                "citations": [],
                "confidence_score": 0.0
            }
        
        # Find synthesis task result (if exists)
        synthesis_result = None
        for task_id, result in task_results.items():
            if result.get("type") == "synthesis":
                synthesis_result = result.get("result", {})
                break
        
        if synthesis_result:
            # Use synthesis result
            response = synthesis_result.get("response_text", "")
            citations = synthesis_result.get("citations", [])
            confidence = synthesis_result.get("confidence_score", 0.0)
        else:
            # Combine all results (fallback)
            responses = []
            all_citations = []
            confidences = []
            
            for task_id, result in task_results.items():
                task_result = result.get("result", {})
                responses.append(task_result.get("response_text", ""))
                all_citations.extend(task_result.get("citations", []))
                confidences.append(task_result.get("confidence_score", 0.0))
            
            response = "\n\n".join(responses)
            citations = all_citations
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, response[:500])
        span.set_attribute("aggregation.citation_count", len(citations))
        span.set_attribute("aggregation.confidence_score", confidence)
        
        return {
            **state,
            "final_response": response,
            "citations": citations,
            "confidence_score": confidence
        }
