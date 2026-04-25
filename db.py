from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


engine = None
SessionLocal: sessionmaker[Session] | None = None


def _configure_engine() -> None:
    global engine, SessionLocal
    if engine is not None and SessionLocal is not None:
        return

    if not settings.database_url:
        return

    url = make_url(settings.database_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+pg8000")
    elif url.drivername == "postgresql+psycopg2":
        url = url.set(drivername="postgresql+pg8000")

    engine = create_engine(url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    _configure_engine()
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not configured")
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    if not settings.database_url:
        return

    _configure_engine()
    if engine is None:
        return

    from models import LessonProgress, LessonTemplate, User  # noqa: F401

    Base.metadata.create_all(bind=engine)
