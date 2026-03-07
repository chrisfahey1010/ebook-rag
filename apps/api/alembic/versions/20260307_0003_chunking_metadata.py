"""Persist chunking configuration and chunk provenance metadata."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260307_0003"
down_revision = "20260307_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("chunking_config", sa.JSON(), nullable=True))
    op.add_column("document_chunks", sa.Column("provenance", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("document_chunks", "provenance")
    op.drop_column("documents", "chunking_config")
