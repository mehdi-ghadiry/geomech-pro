"""
GeoMech Pro - Backend API
==========================
Standalone FastAPI service exposing the geomechanics computation engine
(1D MEM, Mud Weight Window, PDF reporting) over HTTP.

This service has NO UI. The Streamlit frontend (or any future client --
a mobile app, an Excel add-in, etc.) talks to it purely through these
endpoints. This is the split that lets the same computational core serve
more than one client in the future.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user_id, hash_password, verify_password
from rate_limit import clear_failed_attempts, is_locked_out, record_failed_login
from database import Base, engine, get_db
from geomechanics_core import GeomechanicsCore
from models import User, WellResult
from report_generator import generate_pdf_report
from schemas import TokenResponse, UserLogin, UserRegister, WellSaveRequest

app = FastAPI(
    title="GeoMech Pro API",
    description="1D Mechanical Earth Model (MEM) & Wellbore Stability computation engine.",
    version="1.0.0",
)

# Allows the Streamlit frontend (running on a different host/port) to call
# this API from the browser.
#
# Locally (default): allows any origin ("*") so nothing needs configuring
# while you're developing on localhost.
#
# When you publish this: set GEOMECH_ALLOWED_ORIGINS to your real frontend
# URL(s) (comma-separated for more than one), e.g.
#   set GEOMECH_ALLOWED_ORIGINS=https://your-frontend-domain.com
# No code change needed -- this reads it at startup.
_origins_env = os.environ.get("GEOMECH_ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warn_if_running_in_open_dev_mode() -> None:
    """
    Prints a loud, impossible-to-miss reminder in the console on every
    startup, for as long as GEOMECH_ALLOWED_ORIGINS hasn't been set.
    This is deliberately NOT a one-time note in a doc somewhere -- it's
    meant to catch you the day you actually deploy this publicly and
    forgot to set the env var, since by then this message will still be
    sitting right there in the terminal every time the server starts.
    """
    if _allowed_origins == ["*"]:
        print("\n" + "=" * 70)
        print("⚠️  GeoMech Pro backend is running in OPEN DEV MODE.")
        print("    CORS currently allows requests from ANY website (allow_origins=['*']).")
        print("    This is fine for local development (localhost).")
        print("    BEFORE you deploy this publicly, set an environment variable:")
        print("        GEOMECH_ALLOWED_ORIGINS=https://your-real-frontend-domain.com")
        print("    See PRE_LAUNCH_CHECKLIST.md in the project root for the full list.")
        print("=" * 70 + "\n")

# Creates users/well_results tables in the SQLite file on first run.
# Safe to call every startup -- it's a no-op if the tables already exist.
Base.metadata.create_all(bind=engine)

core = GeomechanicsCore()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple liveness check for uptime monitors / load balancers."""
    return {"status": "ok"}


@app.post("/api/v1/las/columns")
async def get_las_columns(las_file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Parses an uploaded LAS, CSV, TXT, or XLSX file and returns its curve names, so the
    client can build column-mapping controls (Depth / DT / DTS / RHOB)
    before requesting a full computation.
    """
    try:
        file_bytes = await las_file.read()
        df, columns = core.load_well_log(file_bytes, las_file.filename or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse well-log file: {exc}")

    return {
        "columns": columns,
        "row_count": int(len(df)),
    }


@app.post("/api/v1/mem/compute")
async def compute_mem(
    las_file: UploadFile = File(...),
    depth_col: str = Form(...),
    dt_col: str = Form(...),
    rhob_col: str = Form(...),
    dts_col: Optional[str] = Form(None),
    lithology_group: str = Form("unspecified"),
    depth_reference: str = Form(...),
    depth_unit: str = Form(...),
    biot_alpha: float = Form(1.0),
    dt_matrix: float = Form(55.5),
    dt_surface: float = Form(180.0),
    compaction_coefficient: float = Form(0.0003),
    normal_trend_calibrated: bool = Form(False),
    eaton_n: float = Form(3.0),
    tectonic_ex: float = Form(0.0005),
    tectonic_ey: float = Form(0.0002),
    stress_regime: str = Form("normal_faulting"),
    sonic_unit: str = Form("us/ft"),
    density_unit: str = Form("g/cm3"),
    assumed_shallow_density: float = Form(2.0),
    lot_depth: Optional[float] = Form(None),
    lot_pressure_mpa: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """
    Runs the full 1D MEM + Mud Weight Window computation on an uploaded
    LAS, CSV, TXT, or XLSX well-log file and returns the results as JSON records.

    If lot_depth and lot_pressure_mpa are both given, the model matches the
    nearest log sample within half the median sampling interval, then checks
    the attainable Shmin range. It applies the solved tectonic_ey and
    recomputes only when the target is achievable.
    The selected Andersonian stress regime is checked against the raw
    poroelastic stress calculation; unsupported samples are withheld rather
    than forced into an arbitrary stress range. The manually-entered
    `tectonic_ey` is used as-is when no LOT/FIT point is given.
    """
    try:
        if (lot_depth is None) != (lot_pressure_mpa is None):
            raise HTTPException(
                status_code=422,
                detail="LOT/FIT depth and pressure must be supplied together.",
            )
        file_bytes = await las_file.read()
        df, columns = core.load_well_log(file_bytes, las_file.filename or "")

        for col, label in [(depth_col, "depth_col"), (dt_col, "dt_col"), (rhob_col, "rhob_col")]:
            if col not in columns:
                raise HTTPException(status_code=422, detail=f"Column '{col}' ({label}) not found in file.")

        dts_actual = dts_col if (dts_col and dts_col in columns) else None

        common_params = dict(
            df=df,
            depth_col=depth_col,
            dt_col=dt_col,
            rhob_col=rhob_col,
            dts_col=dts_actual,
            lithology_group=lithology_group,
            depth_reference=depth_reference,
            depth_unit=depth_unit,
            biot_alpha=biot_alpha,
            dt_matrix=dt_matrix,
            dt_surface=dt_surface,
            compaction_coefficient=compaction_coefficient,
            normal_trend_calibrated=normal_trend_calibrated,
            eaton_n=eaton_n,
            tectonic_ex=tectonic_ex,
            stress_regime=stress_regime,
            sonic_unit=sonic_unit,
            density_unit=density_unit,
            assumed_shallow_density=assumed_shallow_density,
        )

        results_df = core.compute_1d_mem(tectonic_ey=tectonic_ey, **common_params)

        calibration_info: Optional[Dict[str, Any]] = None
        if lot_depth is not None and lot_pressure_mpa is not None:
            calib = core.solve_tectonic_ey_for_lot(
                results_df,
                lot_depth,
                lot_pressure_mpa,
                biot_alpha,
                tectonic_ex=tectonic_ex,
                initial_tectonic_ey=tectonic_ey,
                stress_regime=stress_regime,
            )
            calibration_info = calib
            # Do not silently clip an impossible LOT/FIT target and call it calibrated.
            if calib["calibration_applied"]:
                tectonic_ey = calib["tectonic_ey"]
                results_df = core.compute_1d_mem(tectonic_ey=tectonic_ey, **common_params)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Computation failed: {exc}")

    # NaN is not valid JSON -- convert to None so the client gets clean nulls
    # instead of a malformed response.
    results_df = results_df.replace({np.nan: None})

    return {
        "columns": list(results_df.columns),
        "row_count": int(len(results_df)),
        "results": results_df.to_dict(orient="records"),
        "calibration": calibration_info,
    }
# ==========================================================
# Deviated / Horizontal Wellbore Stability
# ==========================================================
class DeviatedStabilityRequest(BaseModel):
    results: List[Dict[str, Any]]
    depth_m: float
    well_inclination_deg: float
    well_azimuth_deg: float
    shmax_azimuth_deg: float
    friction_angle_deg: float = 30.0
    mud_weight_sg_min: float = 0.90
    mud_weight_sg_max: float = 2.50
    mud_weight_sg_step: float = 0.01
    n_theta: int = 181
    biot_alpha: Optional[float] = None


@app.post("/api/v1/mem/deviated_stability")
def deviated_stability(payload: DeviatedStabilityRequest) -> Dict[str, Any]:
    if not payload.results:
        raise HTTPException(status_code=422, detail="No results provided (payload.results is empty).")

    mem_df = pd.DataFrame(payload.results)

    try:
        out = core.compute_deviated_wellbore_stability(
            mem_df=mem_df,
            depth_m=float(payload.depth_m),
            well_inclination_deg=float(payload.well_inclination_deg),
            well_azimuth_deg=float(payload.well_azimuth_deg),
            shmax_azimuth_deg=float(payload.shmax_azimuth_deg),
            friction_angle_deg=float(payload.friction_angle_deg),
            mud_weight_sg_min=float(payload.mud_weight_sg_min),
            mud_weight_sg_max=float(payload.mud_weight_sg_max),
            mud_weight_sg_step=float(payload.mud_weight_sg_step),
            n_theta=int(payload.n_theta),
            biot_alpha=payload.biot_alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Deviated stability computation failed: {exc}")

    return {
        "collapse_emw_sg": (
            float(out["collapse_emw_sg"]) if out["collapse_emw_sg"] is not None else None
        ),
        "solution_found": bool(out["solution_found"]),
        "status": out["status"],
        "max_tested_mud_weight_sg": float(out["max_tested_mud_weight_sg"]),
        "biot_alpha_used": float(out["biot_alpha_used"]),
        "theta_rad": np.asarray(out["theta_rad"], dtype=float).tolist(),
        "sigma_tt_eff": np.asarray(out["sigma_tt_eff"], dtype=float).tolist(),
    }



class ReportRequest(BaseModel):
    results: List[Dict[str, Any]]
    well_name: str = "Well-01"


@app.post("/api/v1/report/pdf")
def generate_report(payload: ReportRequest) -> StreamingResponse:
    """
    Builds the PDF engineering report from a previously computed result
    set (as returned by /api/v1/mem/compute) and streams it back.
    """
    if not payload.results:
        raise HTTPException(status_code=422, detail="No results provided.")

    results_df = pd.DataFrame(payload.results)
    try:
        pdf_bytes = generate_pdf_report(results_df, well_name=payload.well_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Report generation failed: {exc}")

    # HTTP headers must be Latin-1/ASCII -- c.isalnum() alone is Unicode-aware
    # and would happily pass Persian/Arabic/etc. characters through, which
    # then crashes header encoding. Restrict to ASCII alnum explicitly.
    safe_name = "".join(c for c in payload.well_name if (c.isalnum() and c.isascii()) or c in ("-", "_")) or "Well"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="GeoMech_Report_{safe_name}.pdf"'},
    )


# ==========================================================
# Auth: register / login
# ==========================================================
@app.post("/api/v1/auth/register", response_model=TokenResponse)
def register(payload: UserRegister, db: Session = Depends(get_db)) -> TokenResponse:
    if "@" not in payload.email:
        raise HTTPException(status_code=422, detail="Please provide a valid email address.")

    existing = (
        db.query(User)
        .filter((User.username == payload.username) | (User.email == payload.email))
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="That username or email is already registered.")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, username=user.username)


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    locked_seconds = is_locked_out(payload.username)
    if locked_seconds is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed login attempts. Try again in {locked_seconds} seconds.",
        )

    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        record_failed_login(payload.username)
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    clear_failed_attempts(payload.username)
    token = create_access_token(user.id, user.username)
    return TokenResponse(access_token=token, username=user.username)


@app.get("/api/v1/auth/me")
def get_me(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)) -> Dict[str, Any]:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"id": user.id, "username": user.username, "email": user.email}


# ==========================================================
# Saved wells (per-user, requires login)
# ==========================================================
@app.post("/api/v1/wells")
def save_well(
    payload: WellSaveRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    well = WellResult(
        user_id=user_id,
        well_name=payload.well_name,
        params_json=json.dumps(payload.params),
        results_json=json.dumps(payload.results),
    )
    db.add(well)
    db.commit()
    db.refresh(well)
    return {"id": well.id, "well_name": well.well_name, "created_at": well.created_at.isoformat()}


@app.get("/api/v1/wells")
def list_wells(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    wells = (
        db.query(WellResult)
        .filter(WellResult.user_id == user_id)
        .order_by(WellResult.created_at.desc())
        .all()
    )
    return [{"id": w.id, "well_name": w.well_name, "created_at": w.created_at.isoformat()} for w in wells]


@app.get("/api/v1/wells/{well_id}")
def get_well(
    well_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # Always filter by BOTH well_id and user_id -- this is what stops
    # user A from being able to read user B's saved wells by guessing IDs.
    well = db.query(WellResult).filter(WellResult.id == well_id, WellResult.user_id == user_id).first()
    if not well:
        raise HTTPException(status_code=404, detail="Well not found.")
    return {
        "id": well.id,
        "well_name": well.well_name,
        "params": json.loads(well.params_json),
        "results": json.loads(well.results_json),
        "created_at": well.created_at.isoformat(),
    }


@app.delete("/api/v1/wells/{well_id}")
def delete_well(
    well_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    well = db.query(WellResult).filter(WellResult.id == well_id, WellResult.user_id == user_id).first()
    if not well:
        raise HTTPException(status_code=404, detail="Well not found.")
    db.delete(well)
    db.commit()
    return {"status": "deleted"}
