import os
from dotenv import load_dotenv
from functools import lru_cache
from typing import Any, Literal, Union
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.path_conf import BASE_PATH

load_dotenv()


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=f"{BASE_PATH}/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    ENVIRONMENT: Literal["dev", "pro"] = "dev"

    GEMINI_API_KEY: str

    # FastAPI
    FASTAPI_API_V1_PATH: str = "/api/v1"
    FASTAPI_TITLE: str = "FastAPI"
    FASTAPI_VERSION: str = "0.0.1"
    FASTAPI_DESCRIPTION: str = "FastAPI"
    FASTAPI_DOCS_URL: str = "/docs"
    FASTAPI_REDOC_URL: str = "/redoc"
    FASTAPI_OPENAPI_URL: Union[str, None] = "/openapi"
    FASTAPI_STATIC_FILES: bool = True

    DATABASE_ECHO: bool = False
    DATABASE_POOL_ECHO: bool = False
    DATABASE_SCHEMA: str = "fba"
    DATABASE_CHARSET: str = "utf8mb4"

    CORS_ALLOWED_ORIGINS: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:5173",
    ]
    CORS_EXPOSE_HEADERS: list[str] = [
        "X-Request-ID",
    ]

    MIDDLEWARE_CORS: bool = True
    MIDDLEWARE_ACCESS: bool = True

    DATETIME_TIMEZONE: str = "Asia/Ho_Chi_Minh"
    DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    UPLOAD_READ_SIZE: int = 1024
    UPLOAD_IMAGE_EXT_INCLUDE: list[str] = ["jpg", "jpeg", "png", "gif", "webp"]
    UPLOAD_IMAGE_SIZE_MAX: int = 5 * 1024 * 1024  # 5 MB
    UPLOAD_VIDEO_EXT_INCLUDE: list[str] = ["mp4", "mov", "avi", "flv"]
    UPLOAD_VIDEO_SIZE_MAX: int = 20 * 1024 * 1024  # 20 MB

    LOG_CID_DEFAULT_VALUE: str = "-"
    LOG_CID_UUID_LENGTH: int = 32  # 日志 correlation_id 长度，必须小于等于 32
    LOG_STD_LEVEL: str = "INFO"
    LOG_ACCESS_FILE_LEVEL: str = "INFO"
    LOG_ERROR_FILE_LEVEL: str = "ERROR"
    LOG_STD_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <lvl>{level: <8}</> | "
        "<cyan> {correlation_id} </> | <lvl>{message}</>"
    )
    LOG_FILE_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <lvl>{level: <8}</> | "
        "<cyan> {correlation_id} </> | <lvl>{message}</>"
    )

    TRACE_ID_REQUEST_HEADER_KEY: str = "X-Request-ID"

    OPERA_LOG_PATH_EXCLUDE: list[str] = [
        "/favicon.ico",
        "/docs",
        "/redoc",
        "/openapi",
    ]

    @model_validator(mode="before")
    @classmethod
    def check_env(cls, values: Any) -> Any:
        if values.get("ENVIRONMENT") == "pro":
            values["FASTAPI_OPENAPI_URL"] = None
            values["FASTAPI_STATIC_FILES"] = False
        return values


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
