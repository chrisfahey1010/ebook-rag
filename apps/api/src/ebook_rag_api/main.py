from fastapi import FastAPI

from ebook_rag_api.api.router import api_router
from ebook_rag_api.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    # Keeping construction in a factory makes later test setup and settings
    # overrides straightforward without introducing framework-heavy patterns.
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.include_router(api_router)
    return application


app = create_app()

