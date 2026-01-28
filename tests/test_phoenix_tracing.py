#!/usr/bin/env python3
"""
Test script for Phoenix LLM Observability with OpenInference tracing.
This demonstrates the full tracing pipeline without starting the full API.
"""
import asyncio
import time

# Set up tracing first
from src.utils.telemetry import setup_telemetry, get_tracer
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

print("=" * 60)
print("AlphaEdge - Phoenix LLM Observability Test")
print("=" * 60)
print()

# Test 1: Basic trace
print("[1/4] Testing basic trace export...")
tracer = get_tracer("test.phoenix")

with tracer.start_as_current_span("test.basic_trace") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    span.set_attribute(SpanAttributes.INPUT_VALUE, "Test query: What is Apple's revenue?")
    time.sleep(0.1)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, "Apple's revenue was $383 billion in FY2023")

print("  ✓ Basic trace sent")

# Test 2: Simulated LLM trace
print("[2/4] Testing LLM trace with OpenInference semantics...")
with tracer.start_as_current_span("test.llm_call") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "mlx-community/Qwen2.5-3B-Instruct-4bit")
    span.set_attribute(SpanAttributes.INPUT_VALUE, "Summarize Apple's financial position")
    span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, '[{"role": "user", "content": "Summarize Apple..."}]')
    
    # Simulate LLM processing
    time.sleep(0.2)
    
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, "Apple maintains a strong financial position...")
    span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, '[{"role": "assistant", "content": "Apple maintains..."}]')
    span.set_attribute("llm.token_count.input", 15)
    span.set_attribute("llm.token_count.output", 50)
    span.set_attribute("llm.latency_ms", 200)

print("  ✓ LLM trace sent")

# Test 3: Simulated retriever trace
print("[3/4] Testing retriever trace...")
with tracer.start_as_current_span("test.retriever") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
    span.set_attribute(SpanAttributes.INPUT_VALUE, "Apple 10-K revenue breakdown")
    
    time.sleep(0.1)
    
    # Simulated retrieved documents
    span.set_attribute("retriever.top_k", 5)
    span.set_attribute("retriever.top_score", 0.92)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, '[{"doc": "SEC Filing 10-K...", "score": 0.92}]')

print("  ✓ Retriever trace sent")

# Test 4: Nested chain trace (like agent orchestration)
print("[4/4] Testing nested chain trace (agent orchestration)...")
with tracer.start_as_current_span("test.orchestration") as parent_span:
    parent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    parent_span.set_attribute(SpanAttributes.INPUT_VALUE, "What is AAPL's P/E ratio?")
    
    # Step 1: Intent classification
    with tracer.start_as_current_span("test.classify_intent") as intent_span:
        intent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        intent_span.set_attribute(SpanAttributes.INPUT_VALUE, "Classify: What is AAPL's P/E ratio?")
        time.sleep(0.05)
        intent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, "FINANCIALS")
    
    # Step 2: Agent execution
    with tracer.start_as_current_span("test.openbb_agent") as agent_span:
        agent_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        agent_span.set_attribute("agent.type", "openbb")
        agent_span.set_attribute("query.ticker", "AAPL")
        
        # Retrieval
        with tracer.start_as_current_span("test.fetch_metrics") as fetch_span:
            fetch_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
            time.sleep(0.05)
            fetch_span.set_attribute("metrics.pe_ratio", 28.5)
        
        # LLM response generation
        with tracer.start_as_current_span("test.generate_response") as llm_span:
            llm_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "Qwen2.5-3B")
            time.sleep(0.05)
            llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, "AAPL's P/E ratio is 28.5")
        
        agent_span.set_attribute("agent.confidence_score", 0.95)
    
    parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, "AAPL's P/E ratio is 28.5")

print("  ✓ Nested chain trace sent")

# Give time for traces to flush
print()
print("Flushing traces...")
time.sleep(2)

print()
print("=" * 60)
print("✅ All traces sent to Phoenix!")
print()
print("📊 View traces at: http://localhost:6006")
print()
print("You should see:")
print("  - 4 root traces")
print("  - LLM spans with model name, input/output, token counts")
print("  - Retriever spans with query and documents")
print("  - Nested chain showing agent orchestration flow")
print("=" * 60)
