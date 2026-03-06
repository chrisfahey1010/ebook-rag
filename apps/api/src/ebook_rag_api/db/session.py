from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from ebook_rag_api.core.config import get_settings
from ebook_rag_api.db.vector import is_postgresql_dialect


def _connect_args(database_url: str) -> dict[str, bool]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


@lru_cache(maxsize=1)
def get_engine():
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        future=True,
        connect_args=_connect_args(settings.database_url),
    )
    if is_postgresql_dialect(engine.dialect.name):
        from pgvector.psycopg import register_vector

        @event.listens_for(engine, "connect")
        def register_pgvector(dbapi_connection, _connection_record) -> None:
            register_vector(dbapi_connection)

    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
