"""Make the pgvector embedding column follow configured dimensions."""

from __future__ import annotations

from alembic import op

from ebook_rag_api.core.config import get_settings

revision = "20260307_0002"
down_revision = "20260306_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    dimensions = get_settings().embedding_dimensions
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_vector_ivfflat")
    op.execute(
        f"""
        ALTER TABLE document_chunks
        ALTER COLUMN embedding_vector TYPE vector({dimensions})
        USING NULL::vector({dimensions})
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_vector_ivfflat
        ON document_chunks
        USING ivfflat (embedding_vector vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_vector_ivfflat")
    op.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN embedding_vector TYPE vector(128)
        USING NULL::vector(128)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_vector_ivfflat
        ON document_chunks
        USING ivfflat (embedding_vector vector_cosine_ops)
        WITH (lists = 100)
        """
    )
