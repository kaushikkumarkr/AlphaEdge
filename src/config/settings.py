from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # LLM
    openai_api_key: Optional[SecretStr] = Field(default=None)
    default_model: str = "gpt-4-turbo-preview"
    
    # Data Sources
    fred_api_key: Optional[SecretStr] = Field(default=None)
    sec_user_agent: str = "AlphaEdge research@example.com"
    
    # Vector DB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "alphaedge_sec"
    
    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    # Guardrails
    min_faithfulness_score: float = 0.8
    min_confidence_score: float = 0.7
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
