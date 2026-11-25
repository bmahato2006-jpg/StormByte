from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from backend.app.core.config import settings
from backend.app.core.logging import logger

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL
        # SQLite compatibility for testing/in-memory environments
        if db_url.startswith("sqlite"):
            _engine = create_async_engine(
                db_url,
                echo=False,
                poolclass=NullPool,
                connect_args={"check_same_thread": False},
            )
        elif "asyncpg" in db_url:
            _engine = create_async_engine(
                db_url,
                echo=False,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        else:
            # Fallback to sqlite if driver unsupported
            _engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Safe transactional async session context manager with automatic rollback on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.error(f"Database transaction error: {exc}", exc_info=True)
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI route dependency yielding managed DB sessions."""
    async with get_db_session() as session:
        yield session
