import uuid

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ebook_rag_api.db.base import Base


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page_number"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)

    document = relationship("Document", back_populates="pages")
