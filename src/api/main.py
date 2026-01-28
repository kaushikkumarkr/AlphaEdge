from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import time

from src.orchestration.graph import get_graph
from src.guardrails.schemas import FinalResponse, Citation
from src.utils.telemetry import setup_telemetry, get_tracer
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

app = FastAPI(title="AlphaEdge API", version="1.0.0")

# Setup OpenTelemetry with Phoenix
setup_telemetry(app, service_name="alphaedge-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    filters: Optional[dict] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    citations: List[dict]
    confidence: float
    intent: str
    processing_time_ms: Optional[float] = None


# Metrics tracking
_metrics = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "avg_response_time_ms": 0,
}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Expose metrics for Prometheus scraping."""
    return _metrics


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    tracer = get_tracer("api.query")
    
    with tracer.start_as_current_span("api.query_handler") as span:
        # OpenInference semantic attributes
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, request.query)
        span.set_attribute("query.ticker", request.ticker or "none")
        span.set_attribute("session.id", request.conversation_id or "new")
        
        try:
            _metrics["total_queries"] += 1
            
            graph = get_graph()
            
            entities = {}
            if request.ticker:
                entities["ticker"] = request.ticker.upper()
            
            # Use conversation_id or generate new thread_id
            thread_id = request.conversation_id or str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            # Build filters from request
            filters = request.filters or {}
            if request.ticker:
                filters["ticker"] = request.ticker.upper()
            
            with tracer.start_as_current_span("api.graph_invoke") as graph_span:
                graph_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                result = await graph.ainvoke({
                    "query": request.query,
                    "conversation_id": thread_id,
                    "entities": entities,
                    "filters": filters,
                    "sec_results": [],
                    "openbb_results": [],
                    "fred_results": [],
                    "iteration_count": 0,
                }, config=config)
            
            processing_time = (time.time() - start_time) * 1000
            _metrics["successful_queries"] += 1
            _metrics["avg_response_time_ms"] = (
                _metrics["avg_response_time_ms"] * 0.9 + processing_time * 0.1
            )
            
            # Set output attributes
            response_text = result.get("final_response", "")
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response_text[:1000] if response_text else "")
            span.set_attribute("query.intent", str(result.get("intent", "unknown")))
            span.set_attribute("query.confidence", result.get("confidence_score", 0))
            span.set_attribute("query.processing_time_ms", processing_time)
            span.set_attribute("query.citation_count", len(result.get("citations", [])))
            
            return QueryResponse(
                response=response_text,
                citations=result.get("citations", []),
                confidence=result.get("confidence_score", 0),
                intent=str(result.get("intent", "unknown")),
                processing_time_ms=processing_time,
            )
        
        except Exception as e:
            _metrics["failed_queries"] += 1
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
