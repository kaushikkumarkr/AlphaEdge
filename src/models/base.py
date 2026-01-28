from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class ModelResponse(BaseModel):
    content: str
    model: str


class BaseModelInterface(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        pass
