"""
GeoMech Pro - Backend API Client
==================================
Thin wrapper around HTTP calls to the FastAPI backend, so app.py (the
Streamlit UI) never talks to `requests` directly and never imports the
computation engine. This keeps the frontend a pure presentation layer.

Set GEOMECH_BACKEND_URL as an environment variable (or edit the default
below) to point at wherever the backend is actually running --
localhost while developing, a real server/container URL in production.
"""
import math
import os
from typing import Any, Dict, List, Optional

import requests

BACKEND_URL = os.environ.get("GEOMECH_BACKEND_URL", "http://localhost:8000")

# Large LAS files (tens of thousands of rows) + a full MEM computation can
# take a little while -- keep the timeout generous rather than failing
# a legitimate slow request.
TIMEOUT_SECONDS = 120


class BackendError(Exception):
    """Raised when the backend API returns an error or can't be reached."""


def _raise_for_backend_error(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    raise BackendError(f"[HTTP {response.status_code}] {detail}")


def _json_safe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replaces NaN/Infinity floats with None so `requests` can actually send
    this as JSON. The compute engine legitimately produces NaN for
    physically-invalid or missing depths (that's correct -- see the
    RHOB/DT sanity guard in geomechanics_core.py); NaN just isn't valid
    JSON on its own, so it has to become `null` at the transport boundary,
    not be filtered out of the data.
    """
    clean = []
    for row in records:
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                clean_row[key] = None
            else:
                clean_row[key] = value
        clean.append(clean_row)
    return clean


def get_las_columns(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Returns {"columns": [...], "row_count": int} for the uploaded file."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/las/columns",
            files={"las_file": (filename, file_bytes)},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def compute_mem(file_bytes: bytes, filename: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full 1D MEM computation server-side and returns the JSON result."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/mem/compute",
            files={"las_file": (filename, file_bytes)},
            data=params,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def generate_report_pdf(results: List[Dict[str, Any]], well_name: str) -> bytes:
    """Requests the PDF report for an already-computed result set."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/report/pdf",
            json={"results": _json_safe_records(results), "well_name": well_name},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.content


# --------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------
def register(username: str, email: str, password: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/auth/register",
            json={"username": username, "email": email, "password": password},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def login(username: str, password: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------
# Saved wells (per-user, requires a token from login()/register())
# --------------------------------------------------------------------
def save_well(token: str, well_name: str, params: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/v1/wells",
            json={"well_name": well_name, "params": params, "results": _json_safe_records(results)},
            headers=_auth_headers(token),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def list_wells(token: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/v1/wells",
            headers=_auth_headers(token),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def get_well(token: str, well_id: int) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/api/v1/wells/{well_id}",
            headers=_auth_headers(token),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()


def delete_well(token: str, well_id: int) -> Dict[str, Any]:
    try:
        resp = requests.delete(
            f"{BACKEND_URL}/api/v1/wells/{well_id}",
            headers=_auth_headers(token),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}: {exc}")
    _raise_for_backend_error(resp)
    return resp.json()
