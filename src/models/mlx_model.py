"""MLX-LM model interface for local inference on Apple Silicon."""

from typing import Optional
import time
from src.models.base import BaseModelInterface, ModelResponse
from src.utils.logging import get_logger
from src.utils.telemetry import get_tracer
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

logger = get_logger(__name__)


class MLXModel(BaseModelInterface):
    """Local LLM using MLX for Apple Silicon Macs."""
    
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"):
        """
        Initialize MLX model.
        
        Recommended models for 16GB RAM M1:
        - mlx-community/Qwen2.5-3B-Instruct-4bit (fast, good quality)
        - mlx-community/Llama-3.2-3B-Instruct-4bit (good for general tasks)
        - mlx-community/Phi-3-mini-4k-instruct-4bit (compact, efficient)
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from mlx_lm import load
            logger.info(f"Loading MLX model: {self.model_name}")
            self._model, self._tokenizer = load(self.model_name)
            logger.info("Model loaded successfully")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Generate response using MLX model with OpenInference tracing."""
        tracer = get_tracer()
        
        with tracer.start_as_current_span(
            "llm.mlx.generate",
            kind=trace.SpanKind.CLIENT
        ) as span:
            # Set OpenInference LLM attributes
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model_name)
            span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, 
                f'{{"temperature": {temperature}, "max_tokens": {max_tokens}}}')
            
            # Capture input
            span.set_attribute(SpanAttributes.INPUT_VALUE, prompt[:2000])
            if system_prompt:
                span.set_attribute("llm.system_prompt", system_prompt[:500])
            
            start_time = time.time()
            
            self._load_model()
            
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            
            # Format with chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Track token counts
            input_tokens = len(self._tokenizer.encode(formatted_prompt))
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens)
            
            # Create sampler with temperature
            sampler = make_sampler(temp=temperature)
            
            # Generate response
            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False
            )
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            output_tokens = len(self._tokenizer.encode(response))
            
            # Set output attributes
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response[:2000])
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, input_tokens + output_tokens)
            span.set_attribute("llm.latency_ms", latency_ms)
            
            return ModelResponse(
                content=response,
                model=self.model_name
            )


# Global model instance for reuse
_mlx_model = None


def get_mlx_model(model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit") -> MLXModel:
    """Get or create MLX model singleton."""
    global _mlx_model
    if _mlx_model is None:
        _mlx_model = MLXModel(model_name)
    return _mlx_model
