from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    chat_model: str = Field("gpt-4o", env="CHAT_MODEL")
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    distill_model: str = Field("gpt-4o", env="DISTILL_MODEL")
    data_dir: str = Field("./data", env="DATA_DIR")
    max_context_turns: int = Field(20, env="MAX_CONTEXT_TURNS")
    cors_origins: list[str] = Field(["*"], env="CORS_ORIGINS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
