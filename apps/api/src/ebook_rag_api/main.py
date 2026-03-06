from contextlib import asynccontextmanager

from fastapi import FastAPI

from ebook_rag_api.api.router import api_router
from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db import Base, get_engine
from ebook_rag_api.models import Document, DocumentPage, IngestionJob


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # Importing model modules before metadata creation keeps table registration
    # explicit while avoiding implicit import side effects elsewhere.
    _ = (Document, DocumentPage, IngestionJob)
    Base.metadata.create_all(bind=get_engine())
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    # Keeping construction in a factory makes later test setup and settings
    # overrides straightforward without introducing framework-heavy patterns.
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.include_router(api_router)
    return application


app = create_app()
