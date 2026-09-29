"""
GeoMech Pro - Authentication
==============================
- Passwords are hashed with PBKDF2-HMAC-SHA256 (Python's stdlib
  `hashlib`) -- no compiled dependency like bcrypt, so it installs
  cleanly everywhere (no C/Rust compiler needed on Windows).
- Sessions are stateless JWT bearer tokens (PyJWT), sent by the
  frontend as `Authorization: Bearer <token>` on every request that
  needs to know who the user is.

SECRET KEY: if GEOMECH_SECRET_KEY is set as an environment variable,
that value is used (this is what you want for a real deployment,
especially if you ever run more than one backend instance -- they all
need the SAME key, or tokens issued by one won't be accepted by
another). Otherwise, a strong random key is generated automatically
the first time this runs and saved to `.secret_key` next to this file,
then reused on every future run. Either way there is never a shared,
guessable default key baked into the code.

Keep `.secret_key` out of version control (see .gitignore) and out of
anything you share publicly -- anyone who has it can forge a valid
login token for any account.
"""
import base64
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Header, HTTPException

_SECRET_KEY_FILE = Path(__file__).resolve().parent / ".secret_key"


def _load_or_create_secret_key() -> str:
    env_key = os.environ.get("GEOMECH_SECRET_KEY")
    if env_key:
        return env_key
    if _SECRET_KEY_FILE.exists():
        return _SECRET_KEY_FILE.read_text().strip()
    new_key = secrets.token_hex(32)  # 64 hex chars, cryptographically random
    _SECRET_KEY_FILE.write_text(new_key)
    return new_key


SECRET_KEY = _load_or_create_secret_key()
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 60 * 60 * 24 * 7  # 7 days

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    """Returns a self-contained string: iterations$salt_b64$hash_b64."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations_str, salt_b64, hash_b64 = stored.split("$")
        iterations = int(iterations_str)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def create_access_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": int(time.time()) + JWT_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user_id(authorization: Optional[str] = Header(None)) -> int:
    """
    FastAPI dependency for protected endpoints. Reads the bearer token
    from the Authorization header and returns the authenticated user's
    ID, or raises 401 if the token is missing/invalid/expired.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    return int(payload["sub"])
