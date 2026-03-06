import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ebook_rag_api.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    ingestion_jobs = relationship("IngestionJob", back_populates="document")
    pages = relationship(
        "DocumentPage", back_populates="document", cascade="all, delete-orphan"
    )
