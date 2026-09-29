"""
GeoMech Pro - Database Models
===============================
Two tables:
  - users:        one row per registered account
  - well_results: one row per saved computation, owned by a user

Each user only ever sees their own well_results (enforced in main.py
by always filtering on user_id, never trusting a well ID alone).
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    wells = relationship("WellResult", back_populates="owner", cascade="all, delete-orphan")


class WellResult(Base):
    __tablename__ = "well_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    well_name = Column(String(255), nullable=False)
    params_json = Column(Text, nullable=False)    # compute parameters used (JSON string)
    results_json = Column(Text, nullable=False)   # full 1D MEM results (JSON string)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="wells")
