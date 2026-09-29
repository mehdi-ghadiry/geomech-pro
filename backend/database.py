"""
GeoMech Pro - Database Setup
==============================
SQLite via SQLAlchemy for now -- zero setup, works out of the box on
Windows with no extra server to install. Swapping to Postgres later
only means changing DATABASE_URL; the rest of the code doesn't change.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("GEOMECH_DATABASE_URL", "sqlite:///./geomech.db")

# check_same_thread=False is required for SQLite when accessed from
# FastAPI's threaded request handlers; harmless for other databases.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
