"""
GeoMech Pro - Login Rate Limiting
====================================
Simple in-memory lockout for failed login attempts, keyed by username.
No new dependency (no Redis, no external rate-limiter package) -- just
a timestamp list per username, which is all this needs for a single
backend process.

LIMITATION (worth knowing, not a bug): this resets whenever the backend
restarts, and only works within ONE running backend process. That's
fine for how this is deployed today (one process, one SQLite file). If
this ever runs as multiple backend processes behind a load balancer,
this would need to move into the database or a shared store (e.g.
Redis) so all processes see the same attempt counts.
"""
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes

_failed_attempts: Dict[str, List[float]] = defaultdict(list)
_lock = Lock()


def _recent_attempts(username: str, now: float) -> List[float]:
    """Returns this username's failure timestamps still inside the lockout window."""
    return [t for t in _failed_attempts.get(username, []) if now - t < LOCKOUT_SECONDS]


def is_locked_out(username: str) -> Optional[int]:
    """Returns seconds remaining if locked out, else None."""
    now = time.time()
    with _lock:
        attempts = _recent_attempts(username, now)
        _failed_attempts[username] = attempts  # prune stale entries opportunistically

    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        oldest_counted = attempts[-MAX_FAILED_ATTEMPTS]
        remaining = int(LOCKOUT_SECONDS - (now - oldest_counted))
        return max(remaining, 1)
    return None


def record_failed_login(username: str) -> None:
    now = time.time()
    with _lock:
        attempts = _recent_attempts(username, now)
        attempts.append(now)
        _failed_attempts[username] = attempts


def clear_failed_attempts(username: str) -> None:
    with _lock:
        _failed_attempts.pop(username, None)
