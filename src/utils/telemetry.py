"""OpenTelemetry + Arize Phoenix setup for AlphaEdge LLM Observability."""
import os
from typing import Optional
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# OpenInference for LLM-specific instrumentation
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

# Phoenix client for additional features
try:
    from phoenix.otel import register as phoenix_register
    HAS_PHOENIX = True
except ImportError:
    HAS_PHOENIX = False


_tracer_provider: Optional[TracerProvider] = None
_initialized = False


def setup_telemetry(app=None, service_name: str = "alphaedge"):
    """
    Configure OpenTelemetry with Arize Phoenix backend.
    
    This sets up:
    - OTLP exporter to Phoenix (localhost:4317)
    - FastAPI auto-instrumentation
    - LangChain instrumentation for agent traces
    - OpenAI instrumentation for LLM calls
    """
    global _tracer_provider, _initialized
    
    if _initialized:
        return _tracer_provider
    
    # Check if Phoenix/OTEL is enabled
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # Create resource with service info
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    # Set up tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter (sends to Phoenix)
    exporter = OTLPSpanExporter(
        endpoint=otel_endpoint,
        insecure=True
    )
    
    # Add span processor
    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Instrument LangChain (for agent orchestration traces)
    try:
        LangChainInstrumentor().instrument()
        print("✓ LangChain instrumented")
    except Exception as e:
        print(f"LangChain instrumentation skipped: {e}")
    
    # Instrument OpenAI (for LLM call traces)
    try:
        OpenAIInstrumentor().instrument()
        print("✓ OpenAI instrumented")
    except Exception as e:
        print(f"OpenAI instrumentation skipped: {e}")
    
    # Instrument FastAPI if app provided
    if app:
        FastAPIInstrumentor.instrument_app(app)
        print("✓ FastAPI instrumented")
    
    print(f"✓ Phoenix telemetry enabled: {otel_endpoint}")
    print(f"  View traces at: http://localhost:6006")
    
    _initialized = True
    return _tracer_provider


def get_tracer(name: str = "alphaedge"):
    """Get a tracer instance."""
    return trace.get_tracer(name)


def trace_llm_call(
    model_name: str = None,
    input_value: str = None,
    output_value: str = None,
):
    """
    Decorator to trace LLM calls with OpenInference semantic conventions.
    
    Usage:
        @trace_llm_call(model_name="gpt-4")
        async def my_llm_function(prompt):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"llm.{func.__name__}",
                kind=trace.SpanKind.CLIENT
            ) as span:
                # Set OpenInference LLM attributes
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                if model_name:
                    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                
                # Capture input
                if input_value or (args and len(args) > 0):
                    input_text = input_value or str(args[0])[:1000]
                    span.set_attribute(SpanAttributes.INPUT_VALUE, input_text)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if hasattr(result, 'content'):
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.content[:1000])
                    elif isinstance(result, str):
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result[:1000])
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"llm.{func.__name__}",
                kind=trace.SpanKind.CLIENT
            ) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                if model_name:
                    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_retriever(retriever_name: str = None):
    """
    Decorator to trace retrieval operations with OpenInference conventions.
    
    Usage:
        @trace_retriever(retriever_name="sec_vector_store")
        async def search_documents(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"retriever.{func.__name__}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
                if retriever_name:
                    span.set_attribute("retriever.name", retriever_name)
                
                # Capture query
                if args and len(args) > 0:
                    span.set_attribute(SpanAttributes.INPUT_VALUE, str(args[0])[:500])
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Capture document count
                    if isinstance(result, list):
                        span.set_attribute("retriever.document_count", len(result))
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"retriever.{func.__name__}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
                if retriever_name:
                    span.set_attribute("retriever.name", retriever_name)
                
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, list):
                        span.set_attribute("retriever.document_count", len(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_agent(agent_name: str = None):
    """
    Decorator to trace agent execution with OpenInference conventions.
    
    Usage:
        @trace_agent(agent_name="sec_rag_agent")
        async def execute(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(
                f"agent.{agent_name or func.__name__}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                span.set_attribute("agent.name", agent_name or func.__name__)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Capture agent output
                    if hasattr(result, 'response_text'):
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.response_text[:1000])
                        span.set_attribute("agent.confidence", result.confidence_score)
                    
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return func
    
    return decorator


def create_span(name: str, kind: str = "CHAIN", attributes: dict = None):
    """
    Context manager for creating custom spans.
    
    Usage:
        with create_span("my_operation", kind="LLM") as span:
            span.set_attribute("custom.attr", "value")
            # do work
    """
    tracer = get_tracer()
    span = tracer.start_span(name)
    
    # Set OpenInference span kind
    kind_map = {
        "LLM": "LLM",
        "RETRIEVER": "RETRIEVER", 
        "CHAIN": "CHAIN",
        "EMBEDDING": "EMBEDDING",
        "TOOL": "TOOL",
        "AGENT": "CHAIN",
    }
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind_map.get(kind, "CHAIN"))
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    return span


# Backwards compatibility
def traced(name: str = None):
    """Legacy decorator - use trace_llm_call, trace_retriever, or trace_agent instead."""
    return trace_agent(agent_name=name)
