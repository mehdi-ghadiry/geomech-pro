"""
GeoMech Pro - Pydantic Schemas (auth + saved wells)
======================================================
Kept separate from the schemas already inline in main.py (ReportRequest)
to keep this new auth/wells surface easy to find and extend.
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class WellSaveRequest(BaseModel):
    well_name: str
    params: Dict[str, Any]
    results: List[Dict[str, Any]]
