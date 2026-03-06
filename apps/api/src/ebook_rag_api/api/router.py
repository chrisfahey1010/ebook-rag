from fastapi import APIRouter

from ebook_rag_api.api.routes.documents import router as documents_router
from ebook_rag_api.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(documents_router, prefix="/api/documents", tags=["documents"])
