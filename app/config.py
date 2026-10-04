from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Any value can be overridden in a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Upload safety (step 2) ---
    max_upload_mb: int = 10
    allowed_types: set[str] = {"pdf", "txt", "md"}
    upload_dir: str = "data/uploads"

    # --- Chunking (steps 4-5) ---
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    tokenizer_encoding: str = "cl100k_base"

    # --- Embeddings (step 6) ---
    embedding_provider: str = "local"                     # "local" or "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Vector store (step 7) ---
    chroma_dir: str = "chroma_db"
    collection_name: str = "documents"

    # --- LLM / generation (week 9) ---
    # Kept the name `openai_api_key` so nothing else in the codebase breaks,
    # but it now holds your OpenRouter key (sk-or-v1-...).
    openai_api_key: str | None = None
    openai_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    answer_top_k: int = 5
    min_rerank_score: float = 0.0   # below this, refuse rather than answer


settings = Settings()