from fastapi import APIRouter

from ebook_rag_api.core.config import get_settings

router = APIRouter()


@router.get("/health", summary="Health check")
def healthcheck() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }

