# =============================================================================
# src/database/connection.py
# Database connection and session management
# =============================================================================
"""
Gestión de conexiones a PostgreSQL usando SQLAlchemy 2.0+
"""
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.config.settings import settings
from src.models import Base

# =============================================================================
# Sync Engine & Session
# =============================================================================

# Create sync engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    poolclass=QueuePool,
    pool_pre_ping=True,  # Verify connections before using
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# =============================================================================
# Async Engine & Session (Optional - for async endpoints)
# =============================================================================

# Convert sync URL to async (postgresql -> postgresql+asyncpg)
async_database_url = settings.DATABASE_URL.replace(
    "postgresql://",
    "postgresql+asyncpg://"
)

# Create async engine
async_engine = create_async_engine(
    async_database_url,
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


# =============================================================================
# Database Initialization
# =============================================================================

def sync_postgres_enums(engine):
    """
    Sincroniza dinámicamente los valores de los Enums de SQLAlchemy con PostgreSQL.
    Añade valores nuevos al ENUM sin necesidad de borrar y recrear la base de datos.
    """
    from sqlalchemy import text
    
    if engine.dialect.name != "postgresql":
        return

    try:
        with engine.execution_options(isolation_level="AUTOCOMMIT").connect() as conn:
            for table in Base.metadata.tables.values():
                for column in table.columns:
                    if hasattr(column.type, 'enums') and column.type.name:
                        enum_name = column.type.name
                        # Comprobar si existe el tipo en Postgres
                        result = conn.execute(
                            text("SELECT 1 FROM pg_type WHERE typname = :name"), 
                            {"name": enum_name}
                        ).scalar()
                        
                        if result:
                            # Añadir cada valor de forma segura
                            for value in column.type.enums:
                                stmt = text(f"ALTER TYPE {enum_name} ADD VALUE IF NOT EXISTS '{value}'")
                                try:
                                    conn.execute(stmt)
                                except Exception as e:
                                    # En caso de que el valor ya exista u otro error menor
                                    pass
    except Exception as e:
        print(f"⚠️  Error sincronizando enums con PostgreSQL: {e}")


def init_db():
    """Initialize database - create all tables"""
    print("🔧 Initializing database...")
    sync_postgres_enums(engine)
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized")


def drop_db():
    """Drop all tables - USE WITH CAUTION"""
    print("⚠️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped")


async def init_db_async():
    """Initialize database asynchronously"""
    print("🔧 Initializing database (async)...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized")


# =============================================================================
# Session Context Managers
# =============================================================================

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions (sync)

    Usage:
        with get_db_session() as db:
            result = db.query(User).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for async database sessions

    Usage:
        async with get_async_db_session() as db:
            result = await db.execute(select(User))
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# =============================================================================
# Dependency Injection (for FastAPI)
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints (sync)

    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI endpoints (async)

    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# =============================================================================
# Connection Events
# =============================================================================

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas if using SQLite"""
    # This is for SQLite compatibility, PostgreSQL ignores it
    pass


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Event fired when connection returns to pool"""
    # Can be used for logging/monitoring
    pass


# =============================================================================
# Health Check
# =============================================================================

def check_db_connection() -> bool:
    """Check if database connection is healthy"""
    try:
        with get_db_session() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def check_db_connection_async() -> bool:
    """Check if async database connection is healthy"""
    try:
        async with get_async_db_session() as db:
            await db.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"❌ Async database connection failed: {e}")
        return False


# =============================================================================
# Utility Functions
# =============================================================================

def get_engine():
    """Get sync engine instance"""
    return engine


def get_async_engine():
    """Get async engine instance"""
    return async_engine


def close_db_connections():
    """Close all database connections"""
    engine.dispose()
    print("✅ Database connections closed")


async def close_async_db_connections():
    """Close all async database connections"""
    await async_engine.dispose()
    print("✅ Async database connections closed")
