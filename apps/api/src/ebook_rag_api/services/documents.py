import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ebook_rag_api.models import Document, IngestionJob

FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class StoredUpload:
    original_filename: str
    stored_filename: str
    file_path: Path
    sha256: str
    size_bytes: int


def sanitize_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if not candidate:
        candidate = "document.pdf"

    sanitized = FILENAME_SANITIZER.sub("-", candidate)
    if not sanitized.lower().endswith(".pdf"):
        sanitized = f"{sanitized}.pdf"
    return sanitized


def store_pdf_upload(
    upload: UploadFile, uploads_dir: Path, max_upload_size_mb: int
) -> StoredUpload:
    original_filename = upload.filename or "document.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported.",
        )
    if upload.content_type not in {None, "application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a PDF content type.",
        )

    uploads_dir.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()
    size_limit = max_upload_size_mb * 1024 * 1024
    size_bytes = 0

    temp_name = f"upload-{uuid.uuid4().hex}.tmp"
    temp_path = uploads_dir / temp_name

    header = b""

    try:
        with temp_path.open("wb") as output:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                if not header:
                    header = chunk[:5]
                size_bytes += len(chunk)
                if size_bytes > size_limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"PDF exceeds {max_upload_size_mb} MB upload limit.",
                    )

                sha256.update(chunk)
                output.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        upload.file.close()

    if size_bytes == 0 or header != b"%PDF-":
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    digest = sha256.hexdigest()
    stored_filename = f"{digest[:12]}-{sanitize_filename(original_filename)}"
    final_path = uploads_dir / stored_filename
    shutil.move(temp_path, final_path)

    return StoredUpload(
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=final_path,
        sha256=digest,
        size_bytes=size_bytes,
    )


def create_document_record(
    session: Session, stored_upload: StoredUpload
) -> tuple[Document, IngestionJob]:
    existing_document = session.scalar(
        select(Document).where(Document.sha256 == stored_upload.sha256)
    )
    if existing_document is not None:
        stored_upload.file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This PDF has already been uploaded.",
        )

    document = Document(
        filename=stored_upload.stored_filename,
        original_filename=stored_upload.original_filename,
        title=Path(stored_upload.original_filename).stem or None,
        sha256=stored_upload.sha256,
        file_path=str(stored_upload.file_path),
        status="uploaded",
    )
    session.add(document)
    try:
        session.flush()

        ingestion_job = IngestionJob(document_id=document.id, status="queued")
        session.add(ingestion_job)
        session.commit()
        session.refresh(document)
        session.refresh(ingestion_job)
        return document, ingestion_job
    except IntegrityError:
        session.rollback()
        stored_upload.file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This PDF has already been uploaded.",
        ) from None


def get_document(session: Session, document_id: str) -> Document | None:
    return session.scalar(select(Document).where(Document.id == document_id))


def list_documents(session: Session) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list(session.scalars(statement))
