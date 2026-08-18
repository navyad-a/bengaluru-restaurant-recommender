from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Hybrid Restaurant Recommendation System (Indian Market)"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # PostgreSQL Database Settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "restaurant_recommender"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/restaurant_recommender"

    # Default Hybrid Recommendation Weights (must sum to 1.0)
    DEFAULT_WEIGHT_CF: float = Field(default=0.40, ge=0.0, le=1.0)
    DEFAULT_WEIGHT_CONTENT: float = Field(default=0.35, ge=0.0, le=1.0)
    DEFAULT_WEIGHT_LOCATION: float = Field(default=0.15, ge=0.0, le=1.0)
    DEFAULT_WEIGHT_QUALITY: float = Field(default=0.10, ge=0.0, le=1.0)

    # Spatial & Diversification Parameters
    DEFAULT_MMR_LAMBDA: float = Field(default=0.70, ge=0.0, le=1.0)
    DEFAULT_MAX_DISTANCE_KM: float = Field(default=10.0, gt=0.0)
    DEFAULT_TOP_K: int = Field(default=10, ge=1, le=50)

    # Model Storage Directory
    MODEL_STORAGE_DIR: str = "saved_models"

    # Performance & Concurrency Settings (Phase 12)
    THREAD_POOL_WORKERS: int = Field(default=8, ge=1, le=64)
    
    # Recommendation Cache Settings (Phase 12 & 15)
    CACHE_BACKEND: str = Field(default="memory", description="Cache backend: 'memory' or 'redis'")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    RECOMMENDATION_CACHE_ENABLED: bool = True
    RECOMMENDATION_CACHE_TTL_SECONDS: int = Field(default=300, ge=1)
    RECOMMENDATION_CACHE_MAX_SIZE: int = Field(default=1000, ge=10)

    # Rate Limiting Settings (Phase 12)
    RATE_LIMIT_ENABLED: bool = False  # Disabled by default for local/benchmark testing, enabled via config/env
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=120, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
