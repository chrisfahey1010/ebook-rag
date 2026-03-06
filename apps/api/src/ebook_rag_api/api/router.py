from fastapi import APIRouter

from ebook_rag_api.api.routes.debug import router as debug_router
from ebook_rag_api.api.routes.documents import router as documents_router
from ebook_rag_api.api.routes.health import router as health_router
from ebook_rag_api.api.routes.qa import router as qa_router
from ebook_rag_api.api.routes.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(documents_router, prefix="/api/documents", tags=["documents"])
api_router.include_router(qa_router, prefix="/api/qa", tags=["qa"])
api_router.include_router(retrieval_router, prefix="/api/retrieval", tags=["retrieval"])
api_router.include_router(debug_router, prefix="/api/debug", tags=["debug"])
