"""Central configuration, loaded from environment variables / .env (never hard-code secrets)."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # --- LLM (any OpenAI-compatible endpoint; default = Google Gemini) ---
    llm_api_key: str = Field(default="", description="API key for the LLM provider")
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash-lite"      # agent / answer generation (3.5-flash: only 20 free requests/day)
    llm_fast_model: str = "gemini-3.5-flash-lite"  # decomposition, judge
    # tried in order when a model is over its daily free quota or overloaded (each model has its own quota)
    llm_fallback_models: str = "gemini-3.5-flash-lite,gemini-3-flash-preview,gemma-4-31b-it,gemini-3.1-flash-lite"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2           # per model; then the next fallback model is tried (keeps the UI responsive)
    llm_temperature: float = 0.1

    # --- Vector store (Qdrant). Empty URL -> embedded local mode on disk ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = str(ROOT_DIR / "data" / "qdrant")
    chunks_collection: str = "chunks"
    documents_collection: str = "documents"

    # --- Embeddings: "local" (fastembed, no quota) or "gemini" (API; key defaults to llm_api_key) ---
    # changing provider/model changes the vector size -> use a fresh CHUNKS_COLLECTION and re-ingest
    embed_provider: str = "local"
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str = str(ROOT_DIR / "data" / "models")
    embed_api_key: str = ""
    embed_base_url: str = "https://generativelanguage.googleapis.com/v1beta/"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768               # Matryoshka-truncated from 3072
    embed_batch_size: int = 100        # API max per batch call
    embed_max_rpm: int = 0             # client-side throttle in texts/min; 0 = off (free tier: 100)

    # --- Chunking ---
    chunk_strategy: str = "sentence"  # "fixed" | "sentence"
    chunk_size: int = 800              # characters
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 8
    candidate_k: int = 50          # per retriever, before fusion
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    use_mmr: bool = True
    bm25_cache_path: str = str(ROOT_DIR / "data" / "bm25.pkl")

    # --- Agent ---
    agent_max_steps: int = 6

    # --- API security: required for write operations on the public demo ---
    admin_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
