import numpy as np
import pandas as pd
import plotly.graph_objects as go
import subprocess
import sys
import time
import socket
import os

def is_backend_running(host="127.0.0.1", port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_backend():
    if not is_backend_running():
        # Set environment to include backend in PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath("backend") + os.pathsep + env.get("PYTHONPATH", "")
        
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000"],
            env=env
        )
        for _ in range(15):
            if is_backend_running():
                break
            time.sleep(0.5)

start_backend()

import streamlit as st
from plotly.subplots import make_subplots

import api_client
from api_client import BackendError
from depth_units import depth_from_meters, depth_to_meters
from config import (
    DEVELOPER_EMAIL,
    DEVELOPER_NAME,
    DEVELOPER_TITLE,
    DEVELOPER_WHATSAPP,
    PLATFORM_NAME,
)

# Page Configuration
st.set_page_config(
    page_title="GeoMech Pro | 1D MEM & Deviated Wellbore Stability",
    page_icon="⚒️",
    layout="wide",
)

# Header Section
st.title("⚒️ GeoMech Pro: 1D Mechanical Earth Model & Wellbore Stability")
st.caption(f"Advanced Subsurface Geomechanics & 3D Deviated Wellbore Simulator | Lead Developer: **{DEVELOPER_NAME}**")
st.markdown("---")

# --- Session state defaults (auth + saved-well viewing) ---
st.session_state.setdefault("auth_token", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("loaded_well_id", None)
st.session_state.setdefault("loaded_well_name", None)

# Sidebar: Account (login / register / saved wells)
with st.sidebar:
    st.header("🔐 Account")

    if st.session_state["auth_token"]:
        st.success(f"Logged in as **{st.session_state['username']}**")
        if st.button("Log out", use_container_width=True):
            st.session_state["auth_token"] = None
            st.session_state["username"] = None
            st.session_state["loaded_well_id"] = None
            st.rerun()

        # --- Saved wells for this user ---
        st.subheader("📚 My Saved Wells")
        try:
            saved_wells = api_client.list_wells(st.session_state["auth_token"])
        except BackendError as e:
            saved_wells = []
            st.error(f"Could not load saved wells: {e}")

        if not saved_wells:
            st.caption("No saved wells yet. Compute a well below, then save it here.")
        else:
            for w in saved_wells:
                wcol1, wcol2 = st.columns([3, 1])
                wcol1.write(f"**{w['well_name']}**\n\n{w['created_at'][:10]}")
                if wcol2.button("Open", key=f"open_well_{w['id']}", use_container_width=True):
                    st.session_state["loaded_well_id"] = w["id"]
                    st.rerun()
                if wcol2.button("🗑️", key=f"delete_well_{w['id']}", use_container_width=True):
                    try:
                        api_client.delete_well(st.session_state["auth_token"], w["id"])
                        if st.session_state["loaded_well_id"] == w["id"]:
                            st.session_state["loaded_well_id"] = None
                        st.rerun()
                    except BackendError as e:
                        st.error(f"Could not delete: {e}")
    else:
        auth_tab_login, auth_tab_register = st.tabs(["Log in", "Register"])
        with auth_tab_login:
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    try:
                        result = api_client.login(login_username, login_password)
                        st.session_state["auth_token"] = result["access_token"]
                        st.session_state["username"] = result["username"]
                        st.rerun()
                    except BackendError as e:
                        st.error(str(e))
        with auth_tab_register:
            with st.form("register_form"):
                reg_username = st.text_input("Choose a username")
                reg_email = st.text_input("Email")
                reg_password = st.text_input("Choose a password", type="password")
                if st.form_submit_button("Create account", use_container_width=True):
                    try:
                        result = api_client.register(reg_username, reg_email, reg_password)
                        st.session_state["auth_token"] = result["access_token"]
                        st.session_state["username"] = result["username"]
                        st.rerun()
                    except BackendError as e:
                        st.error(str(e))
        st.caption("Guest mode works without an account -- you just can't save results.")

    st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("📂 Well Data & Calibration")
    st.info(
        f"👨‍💻 **Developer:** {DEVELOPER_NAME}\n\n"
        f"🎓 {DEVELOPER_TITLE}\n\n"
        f"✉ **Email:** {DEVELOPER_EMAIL}\n\n"
        f"**WhatsApp:** {DEVELOPER_WHATSAPP}\n\n"
    )

    uploaded_file = st.file_uploader(
        "Upload Well Log Data (LAS, CSV, TXT, XLSX)",
        type=["las", "csv", "txt", "xlsx"],
        help="Upload a LAS log or a tabular export with a header row. Required curves: depth, DT, and RHOB; DTS is optional.",
    )

    st.subheader("⚙️ Log Units & In-Situ Calibration")
    depth_reference_label = st.selectbox(
        "Selected depth curve represents",
        ["Choose depth reference...", "TVD (true vertical depth)", "MD (measured depth)"],
        index=0,
        help="Pore-pressure and stress calculations require TVD. MD needs a measured well trajectory for conversion.",
    )
    depth_reference = {
        "TVD (true vertical depth)": "TVD",
        "MD (measured depth)": "MD",
    }.get(depth_reference_label, "")
    depth_unit = st.selectbox("Depth Unit", ["m", "ft"], index=0)
    sonic_unit = st.selectbox("Sonic Unit", ["us/ft", "us/m"], index=0)
    density_unit = st.selectbox("Density Unit", ["g/cm3", "kg/m3"], index=0)

    # 🧭 Wellbore Trajectory Inputs for Deviated/Horizontal Wells
    st.subheader("🧭 Well Trajectory & In-Situ Azimuth")
    well_inclination = st.slider(
        "Well Inclination θ (°)",
        min_value=0.0,
        max_value=90.0,
        value=0.0,
        step=5.0,
        help="0° = Vertical Well, 90° = Horizontal Well."
    )
    well_azimuth = st.slider(
        "Well Azimuth φ (°)",
        min_value=0.0,
        max_value=360.0,
        value=0.0,
        step=5.0,
        help="Well trajectory direction relative to True North (0°=N, 90°=E)."
    )
    shmax_azimuth = st.slider(
        "SHmax Azimuth (°)",
        min_value=0.0,
        max_value=180.0,
        value=45.0,
        step=5.0,
        help="Maximum horizontal stress orientation relative to True North."
    )

    with st.expander("🛠️ Advanced Geomechanics Parameters", expanded=False):
        biot_alpha = st.slider("Biot's Coefficient (α)", 0.5, 1.0, 1.0, 0.05)
        dt_surface = st.number_input("Surface sonic for normal trend (μs/ft)", value=180.0, min_value=1.0, step=5.0)
        dt_normal = st.number_input("Compacted-end sonic for normal trend (μs/ft)", value=100.0, min_value=1.0, step=5.0)
        compaction_coefficient = st.number_input(
            "Normal-trend compaction coefficient (1/m)", value=0.0003, min_value=0.0,
            step=0.0001, format="%.6f",
            help="Calibrate these trend settings with normally compacted shale from this field. Generic defaults are not field validation.",
        )
        normal_trend_calibrated = st.checkbox(
            "I calibrated this normal sonic trend for the formation",
            value=False,
            help="Enable only after calibrating the trend against normally compacted local shale and/or field pressure references.",
        )
        st.caption("Negative estimates and estimates above the overburden screening limit are excluded from stresses and mud-weight calculations. Generic defaults are not field calibration.")
        eaton_exp = st.slider("Eaton's Exponent", 1.0, 5.0, 3.0, 0.1)
        assumed_shallow_density = st.number_input(
            "Assumed Shallow Density Above Log Top (g/cm3)",
            value=2.0, step=0.05, format="%.2f",
            help="The log usually doesn't start at the surface. This estimates the "
                 "overburden weight of everything above the top of the log using a "
                 "typical near-surface sediment density.",
        )
        tectonic_ex = st.number_input("Tectonic Strain εx (SHmax direction)", value=0.0005, format="%.5f")

        use_lot_calibration = st.checkbox(
            "📏 Calibrate εy with a LOT/FIT measurement",
            help="If you have a real Leak-Off Test or Formation Integrity Test result, "
                 "enter it below instead of guessing εy. The model matches the nearest "
                 "log sample (within its depth tolerance) and applies calibration only "
                 "when the pressure is within the modeled Shmin range.",
        )
        if use_lot_calibration:
            lot_depth_input = st.number_input("LOT/FIT Depth (m)", value=2000.0, step=10.0)
            lot_pressure_input = st.number_input(
                "LOT/FIT Pressure (MPa)", value=30.0, step=0.5,
                help="Convert from EMW (SG) if needed: pressure_MPa = EMW_SG × depth_m × 0.00980665",
            )
            tectonic_ey = None  # solved by the backend, not user-entered
        else:
            tectonic_ey = st.number_input("Tectonic Strain εy (Shmin direction)", value=0.0002, format="%.5f")
            lot_depth_input = None
            lot_pressure_input = None

results_df = None
active_well_name = None
compute_params_used = None
data_source = None  # "upload" or "saved"

if st.session_state["loaded_well_id"] is not None:
    # --- Viewing a previously saved well from the user's account ---
    try:
        well_detail = api_client.get_well(st.session_state["auth_token"], st.session_state["loaded_well_id"])
        results_df = pd.DataFrame(well_detail["results"]).apply(pd.to_numeric, errors="coerce")
        active_well_name = well_detail["well_name"]
        compute_params_used = well_detail.get("params", {})
        data_source = "saved"
        st.info(f"📚 Viewing saved well **{active_well_name}** from your account.")
        if st.button("🔙 Close saved well and upload a new file instead"):
            st.session_state["loaded_well_id"] = None
            st.rerun()
    except BackendError as e:
        st.error(f"❌ Could not load saved well: {e}")
        st.session_state["loaded_well_id"] = None

elif uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()

        # --- Step 1: ask the backend which curves this file contains ---
        with st.spinner("Reading well-log file on server..."):
            col_info = api_client.get_las_columns(file_bytes, uploaded_file.name)
        columns = col_info["columns"]
        st.sidebar.success(f"✅ Well-log data loaded ({col_info['row_count']} rows)!")

        def find_default(candidates, cols):
            for c in candidates:
                for col in cols:
                    if c.lower() in col.lower():
                        return col
            return cols[0] if cols else ""

        dept_col = find_default(["dept", "depth"], columns)
        dt_col = find_default(["dt", "dtco"], columns)
        dts_col = find_default(["dts", "dtsm"], columns)
        rhob_col = find_default(["rhob", "den"], columns)

        st.sidebar.subheader("🎯 Curve Mapping")
        sel_dept = st.sidebar.selectbox("Depth curve (select a TVD curve)", columns, index=columns.index(dept_col) if dept_col in columns else 0)
        sel_dt = st.sidebar.selectbox("Compressional Sonic (DT)", columns, index=columns.index(dt_col) if dt_col in columns else 0)
        sel_dts = st.sidebar.selectbox("Shear Sonic (DTS)", ["None"] + columns, index=(columns.index(dts_col) + 1) if dts_col in columns else 0)
        sel_rhob = st.sidebar.selectbox("Bulk Density (RHOB)", columns, index=columns.index(rhob_col) if rhob_col in columns else 0)

        dts_actual = None if sel_dts == "None" else sel_dts
        if dts_actual is None:
            vs_lithology_label = st.sidebar.selectbox(
                "Lithology group (only if DTS is missing)",
                [
                    "Not specified / mixed — measured DTS required",
                    "Water-saturated clastic shale/sandstone — Castagna mudrock-line estimate",
                    "Limestone — empirical limestone Vp–Vs estimate",
                    "Dolomite — empirical dolomite Vp–Vs estimate",
                    "Mixed/other carbonate — measured DTS or local calibration required",
                ],
                index=0,
                help=(
                    "The clastic, limestone, and dolomite relations are separate empirical fits. "
                    "They are estimates, not substitutes for measured DTS or formation calibration. "
                    "Select one lithology only when it represents the full analyzed interval."
                ),
            )
            lithology_options = {
                "Water-saturated clastic": "water_saturated_clastic",
                "Limestone": "limestone",
                "Dolomite": "dolomite",
                "Mixed/other carbonate": "carbonate",
            }
            lithology_group = next(
                (value for label, value in lithology_options.items() if vs_lithology_label.startswith(label)),
                "unspecified",
            )
            if lithology_group == "water_saturated_clastic":
                st.warning(
                    "Using the empirical Castagna mudrock line for the selected water-saturated "
                    "clastic interval. This is an estimate, not measured DTS."
                )
            elif lithology_group in {"limestone", "dolomite"}:
                st.warning(
                    f"Using an empirical {lithology_group} Vp–Vs relation. It is not locally calibrated; "
                    "inspect the estimated DTS/Vs and use measured DTS or a local fit when available."
                )
            else:
                st.info(
                    "For mixed/unknown lithology, provide measured DTS or a locally calibrated Vp–Vs relation."
                )
                st.stop()
        else:
            lithology_group = "measured_dts"

        if depth_reference != "TVD":
            if depth_reference == "MD":
                st.warning(
                    "This version cannot convert MD to TVD without a measured well trajectory. "
                    "Choose a TVD depth curve, or provide trajectory conversion support before computing pressures."
                )
            else:
                st.info("Select whether the chosen depth curve is TVD or MD before computing.")
            st.stop()

        if not normal_trend_calibrated:
            st.warning(
                "Pressure, stress, and mud-weight calculations are paused. First calibrate the normal sonic trend "
                "for this formation; the generic starting values are not safe to treat as field data."
            )
            st.stop()

        # --- Step 2: send the file + column mapping + parameters to the
        #     backend and get the computed 1D MEM back as JSON ---
        compute_params = {
            "depth_col": sel_dept,
            "depth_reference": depth_reference,
            "depth_unit": depth_unit,
            "dt_col": sel_dt,
            "rhob_col": sel_rhob,
            "dts_col": dts_actual or "",
            "lithology_group": lithology_group,
            "biot_alpha": biot_alpha,
            "dt_matrix": dt_normal,
            "dt_surface": dt_surface,
            "compaction_coefficient": compaction_coefficient,
            "normal_trend_calibrated": normal_trend_calibrated,
            "eaton_n": eaton_exp,
            "tectonic_ex": tectonic_ex,
            "sonic_unit": sonic_unit,
            "density_unit": density_unit,
            "assumed_shallow_density": assumed_shallow_density,
            "well_inclination": well_inclination,
            "well_azimuth": well_azimuth,
            "shmax_azimuth": shmax_azimuth,
        }
        if use_lot_calibration:
            compute_params["lot_depth"] = lot_depth_input
            compute_params["lot_pressure_mpa"] = lot_pressure_input
        else:
            compute_params["tectonic_ey"] = tectonic_ey

        with st.spinner("Computing Geomechanics Engine on server..."):
            compute_result = api_client.compute_mem(file_bytes, uploaded_file.name, compute_params)

        results_df = pd.DataFrame(compute_result["results"])
        results_df = results_df.apply(pd.to_numeric, errors="coerce")
        lower_hits = int(results_df.get("Pore_Pressure_Lower_Bound_Hit", pd.Series(0, index=results_df.index)).fillna(0).sum())
        upper_hits = int(results_df.get("Pore_Pressure_Upper_Bound_Hit", pd.Series(0, index=results_df.index)).fillna(0).sum())
        invalid_pp = int((1 - results_df.get("Pore_Pressure_Estimate_Valid", pd.Series(0, index=results_df.index)).fillna(0)).sum())
        if lower_hits or upper_hits or invalid_pp:
            negative_pp = int(results_df.get("Pore_Pressure_Negative_Flag", pd.Series(0, index=results_df.index)).fillna(0).sum())
            st.warning(
                f"Eaton pressure failed quality checks at {invalid_pp} samples, including {negative_pp} negative estimates. "
                "Invalid pressure values are excluded from the stress and mud-weight calculations; gaps in those outputs are intentional. "
                "Recalibrate the normal sonic trend against normally compacted local shale and field measurements before operational use."
            )
        if "Elastic_Properties_Valid" in results_df.columns:
            invalid_elastic = int(
                (1 - results_df["Elastic_Properties_Valid"].fillna(0)).sum()
            )
            if invalid_elastic:
                st.warning(
                    f"Elastic-property QC failed at {invalid_elastic} samples because the density or Vp/Vs pair "
                    "does not support positive elastic moduli. Modulus-, strength-, and stress-dependent values "
                    "are masked at those samples; inspect the logs before use."
                )
        active_well_name = uploaded_file.name
        compute_params_used = compute_params
        data_source = "upload"

        calib_info = compute_result.get("calibration")
        if calib_info:
            requested_depth_display = float(depth_from_meters(calib_info["requested_depth"], depth_unit))
            matched_depth_display = float(depth_from_meters(calib_info["matched_depth"], depth_unit))
            depth_offset_display = float(depth_from_meters(calib_info["depth_offset_m"], depth_unit))
            if calib_info.get("calibration_applied", calib_info["within_range"]):
                st.success(
                    f"📏 LOT/FIT requested at {requested_depth_display:.1f} {depth_unit}; calibrated at the nearest log sample "
                    f"({matched_depth_display:.1f} {depth_unit}, offset {depth_offset_display:.1f} {depth_unit}). "
                    f"Solved tectonic εy = {calib_info['tectonic_ey']:.6f}; Shmin before calibration was "
                    f"{calib_info['uncalibrated_shmin_mpa']:.1f} MPa."
                )
            else:
                st.warning(
                    f"⚠️ LOT/FIT calibration was not applied. At the matched log sample ({matched_depth_display:.1f} {depth_unit}; "
                    f"requested {requested_depth_display:.1f} {depth_unit}), the modeled Shmin range is "
                    f"{calib_info['achievable_min_mpa']:.1f}–{calib_info['achievable_max_mpa']:.1f} MPa. "
                    "The original tectonic strain was retained."
                )

    except BackendError as e:
        st.error(f"❌ Backend Connection/Computation Error: {e}")
        st.info(f"The backend at `{api_client.BACKEND_URL}` is unreachable or returned an error. Make sure the FastAPI service is running.")
    except Exception as e:
        st.error(f"❌ Execution Error: {str(e)}")

if results_df is not None:
    display_depth = depth_from_meters(results_df["Depth"], depth_unit)
    depth_label = f"Depth ({depth_unit})"
    try:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Depth Interval", f"{display_depth.min():.0f} - {display_depth.max():.0f} {depth_unit}")
        c2.metric("Mean Sv", f"{results_df['Overburden_Stress_Sv_MPa'].mean():.1f} MPa" if "Overburden_Stress_Sv_MPa" in results_df else "N/A")
        c3.metric("Mean Pore Press", f"{results_df['Pore_Pressure_Pp_MPa'].mean():.1f} MPa" if "Pore_Pressure_Pp_MPa" in results_df else "N/A")
        
        # Display Deviated collapse MW if available, else standard collapse MW
        disp_col_mw = "Deviated_Collapse_EMW_SG" if ("Deviated_Collapse_EMW_SG" in results_df and well_inclination > 0) else "Collapse_EMW_SG"
        c4.metric(f"Min MW ({well_inclination:.0f}° Incl)", f"{results_df[disp_col_mw].mean():.2f} SG" if disp_col_mw in results_df else "N/A")
        c5.metric("Mean Shmin Screen", f"{results_df['Shmin_EMW_SG'].mean():.2f} SG" if "Shmin_EMW_SG" in results_df else "N/A")

        # Tab Structure
        tab1, tab2, tab3 = st.tabs([
            "📊 1D MEM & Mud Weight Window Logs",
            "🎯 2D Wellbore Stress & Failure Simulator (Kirsch & Mohr-Coulomb)",
            "🧭 Deviated & Horizontal Wellbore Stability (3D Trajectory)"
        ])

        # ==========================================
        # TAB 1: Continuous 1D Logs
        # ==========================================
        with tab1:
            st.markdown("### 📊 4-Track 1D MEM & Safe Mud Weight Window")
            fig = make_subplots(
                rows=1,
                cols=4,
                shared_yaxes=True,
                horizontal_spacing=0.04,
                subplot_titles=(
                    "Track 1: Elastic Moduli",
                    "Track 2: Rock Strength (UCS)",
                    "Track 3: In-Situ Stresses (MPa)",
                    "Track 4: Mud Weight Window (SG)",
                ),
            )

            depth = display_depth

            # Track 1: Elastic Moduli
            if "Youngs_Modulus_GPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Youngs_Modulus_GPa"], y=depth, name="Young's (E)", line=dict(color="#00E5FF", width=1.5)), row=1, col=1)
            if "Bulk_Modulus_GPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Bulk_Modulus_GPa"], y=depth, name="Bulk (K)", line=dict(color="#FFD700", width=1.2, dash="dot")), row=1, col=1)
            if "Shear_Modulus_GPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Shear_Modulus_GPa"], y=depth, name="Shear (G)", line=dict(color="#FF9100", width=1.2, dash="dash")), row=1, col=1)

            # Track 2: Rock Strength
            if "UCS_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["UCS_MPa"], y=depth, name="UCS (MPa)", line=dict(color="#AB47BC", width=2)), row=1, col=2)
            if "Tensile_Strength_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Tensile_Strength_MPa"], y=depth, name="Tensile (To)", line=dict(color="#BA68C8", width=1.2, dash="dot")), row=1, col=2)

            # Track 3: Stresses
            if "Pore_Pressure_Pp_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Pore_Pressure_Pp_MPa"], y=depth, name="Pore Press (Pp)", line=dict(color="#00E676", width=2)), row=1, col=3)
            if "Shmin_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Shmin_MPa"], y=depth, name="Shmin (σh)", line=dict(color="#2979FF", width=1.5, dash="dash")), row=1, col=3)
            if "SHmax_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["SHmax_MPa"], y=depth, name="SHmax (σH)", line=dict(color="#FF6E40", width=1.5, dash="dash")), row=1, col=3)
            if "Overburden_Stress_Sv_MPa" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Overburden_Stress_Sv_MPa"], y=depth, name="Overburden (Sv)", line=dict(color="#D50000", width=2)), row=1, col=3)

            # Track 4: Mud Weight Window
            if "Pore_Pressure_EMW_SG" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Pore_Pressure_EMW_SG"], y=depth, name="Pore Press EMW", line=dict(color="#00E676", width=1.5, dash="dot")), row=1, col=4)
            if "Collapse_EMW_SG" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Collapse_EMW_SG"], y=depth, name="Vertical Collapse (0°)", line=dict(color="#FF1744", width=2)), row=1, col=4)
            if "Deviated_Collapse_EMW_SG" in results_df and well_inclination > 0:
                fig.add_trace(go.Scatter(x=results_df["Deviated_Collapse_EMW_SG"], y=depth, name=f"Deviated Collapse ({well_inclination:.0f}°)", line=dict(color="#FF5252", width=2.5, dash="dash")), row=1, col=4)
            if "Shmin_EMW_SG" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Shmin_EMW_SG"], y=depth, name="Shmin Opening/Propagation Screen", line=dict(color="#FF9100", width=1.8, dash="dash")), row=1, col=4)
            if "Tensile_Breakdown_EMW_SG" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Tensile_Breakdown_EMW_SG"], y=depth, name="Tensile Initiation Estimate (Vertical)", line=dict(color="#2979FF", width=1.5, dash="dot")), row=1, col=4)
            if "Fracture_EMW_SG" in results_df:
                fig.add_trace(go.Scatter(x=results_df["Fracture_EMW_SG"], y=depth, name="Conservative Upper Screen", line=dict(color="#651FFF", width=2)), row=1, col=4)

            fig.update_yaxes(title_text=depth_label, autorange="reversed", row=1, col=1)
            fig.update_xaxes(title_text="Moduli (GPa)", row=1, col=1)
            fig.update_xaxes(title_text="Strength (MPa)", row=1, col=2)
            fig.update_xaxes(title_text="Stress (MPa)", row=1, col=3)
            fig.update_xaxes(title_text="EMW (SG)", row=1, col=4)

            fig.update_layout(
                height=800,
                hovermode="y unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                ),
                margin=dict(l=50, r=40, t=120, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # TAB 2: 2D Kirsch & Mohr-Coulomb Simulator
        # ==========================================
        with tab2:
            st.markdown("### 🔬 2D Near-Wellbore Elastic Stress & Failure Field")
            st.write(
                "This interactive tool simulates the analytical **Kirsch stress distribution** and evaluates "
                "the **Mohr-Coulomb Failure Index (MCI)** around the borehole."
            )

            # UI Controls
            col_depth, col_mud, col_r = st.columns(3)
            with col_depth:
                target_depth_display = st.slider(
                    f"Select Depth ({depth_unit})",
                    float(display_depth.min()),
                    float(display_depth.max()),
                    float(display_depth.mean()),
                    step=0.5 if depth_unit == "m" else 1.0,
                )

            # UI uses the selected unit; all calculations remain in metres.
            target_depth_m = depth_to_meters(target_depth_display, depth_unit)

            # Locate nearest row using the displayed depth values.
            nearest_idx = (display_depth - target_depth_display).abs().idxmin()
            row_data = results_df.loc[nearest_idx]

            def get_val(key_list, default_val):
                for k in key_list:
                    if k in row_data.index:
                        val = row_data[k]
                        if not pd.isna(val) and not np.isinf(val):
                            return float(val)
                return float(default_val)

            rec_collapse = get_val(["Deviated_Collapse_EMW_SG", "Collapse_EMW_SG"], 1.15)
            rec_frac = get_val(["Fracture_EMW_SG", "Shmin_EMW_SG"], 1.85)

            with col_mud:
                sim_mud_sg = st.slider(
                    "Test Drilling Mud Weight (SG)",
                    0.80,
                    2.50,
                    float(np.clip(rec_collapse + 0.05, 0.90, 2.30)),
                    0.02,
                )

            with col_r:
                well_dia_inch = st.selectbox("Wellbore Diameter (inches)", [6.0, 8.5, 12.25, 17.5], index=1)
                Rw = (well_dia_inch * 0.0254) / 2.0

            # Extract Formation Parameters safely
            sv = get_val(["Overburden_Stress_Sv_MPa", "Sv_MPa"], 50.0)
            shmin = get_val(["Shmin_MPa"], 32.0)
            shmax = get_val(["SHmax_MPa"], 42.0)
            pp = get_val(["Pore_Pressure_Pp_MPa", "Pp_MPa"], 22.0)
            row_biot_alpha = get_val(["Biot_Coefficient"], biot_alpha)
            ucs = get_val(["UCS_MPa"], 45.0)
            friction_ang = get_val(["Friction_Angle_deg", "Phi_deg", "Internal_Friction_deg"], 30.0)

            # Mud pressure calculation: Pw (MPa)
            pw = (sim_mud_sg * 1000.0) * 9.80665 * target_depth_m / 1e6

            # Diagnostic Status
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Local UCS", f"{ucs:.1f} MPa")
            m2.metric("Pore Pressure (Pp)", f"{pp:.1f} MPa")
            m3.metric("Current Mud Press (Pw)", f"{pw:.1f} MPa")

            if sim_mud_sg < rec_collapse:
                stability_status = "⚠️ Shear Breakout (MW Too Low)"
            elif sim_mud_sg > rec_frac:
                stability_status = "⚠️ Conservative fracture/propagation screen exceeded"
            else:
                stability_status = "✅ Borehole Stable"
            m4.metric("Stability State", stability_status)

            # Cartesian Grid Formulation for Kirsch Solution
            max_r = Rw * 3.0
            grid_points = 80
            x_axis = np.linspace(-max_r, max_r, grid_points)
            y_axis = np.linspace(-max_r, max_r, grid_points)
            X, Y = np.meshgrid(x_axis, y_axis)
            R = np.sqrt(X**2 + Y**2)
            THETA = np.arctan2(Y, X)

            mask_hole = R < Rw
            safe_R = np.where(mask_hole, Rw, R)
            eta = (Rw / safe_R) ** 2

            s_mean = (shmax + shmin) / 2.0
            s_diff = (shmax - shmin) / 2.0

            # Kirsch Equations
            sig_r = s_mean * (1.0 - eta) + s_diff * (1.0 - 4.0 * eta + 3.0 * (eta**2)) * np.cos(2.0 * THETA) + pw * eta
            sig_th = s_mean * (1.0 + eta) - s_diff * (1.0 + 3.0 * (eta**2)) * np.cos(2.0 * THETA) - pw * eta
            tau_rth = -s_diff * (1.0 + 2.0 * eta - 3.0 * (eta**2)) * np.sin(2.0 * THETA)

            # Effective Stresses
            sig_r_eff = sig_r - row_biot_alpha * pp
            sig_th_eff = sig_th - row_biot_alpha * pp

            # Principal In-Plane Stresses
            c_stress = (sig_r_eff + sig_th_eff) / 2.0
            rad_diff = np.sqrt(((sig_r_eff - sig_th_eff) / 2.0) ** 2 + tau_rth**2)
            sig1_eff = c_stress + rad_diff
            sig3_eff = c_stress - rad_diff

            # Mohr-Coulomb Failure Index
            phi_rad = np.radians(friction_ang)
            q_mc = np.tan(np.pi / 4.0 + phi_rad / 2.0) ** 2
            mc_strength = q_mc * np.maximum(sig3_eff, 0.0) + ucs
            mci = sig1_eff / np.maximum(mc_strength, 1e-4)

            sig_th_eff[mask_hole] = np.nan
            mci[mask_hole] = np.nan

            # 2D Visuals
            fig_sim = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Effective Hoop Stress σ'θ (MPa)",
                    "Mohr-Coulomb Failure Index (MCI ≥ 1.0 = Failure)",
                ),
                horizontal_spacing=0.12,
            )

            fig_sim.add_trace(
                go.Contour(
                    z=sig_th_eff,
                    x=x_axis,
                    y=y_axis,
                    colorscale="Viridis",
                    contours=dict(showlines=False),
                    colorbar=dict(title="σ'θ (MPa)", x=0.44),
                ),
                row=1,
                col=1,
            )

            circle_theta = np.linspace(0, 2 * np.pi, 100)
            fig_sim.add_trace(
                go.Scatter(
                    x=Rw * np.cos(circle_theta),
                    y=Rw * np.sin(circle_theta),
                    fill="toself",
                    fillcolor="rgba(30, 30, 30, 0.9)",
                    line=dict(color="#FFFFFF", width=2),
                    name="Wellbore Wall",
                ),
                row=1,
                col=1,
            )

            fig_sim.add_trace(
                go.Contour(
                    z=mci,
                    x=x_axis,
                    y=y_axis,
                    colorscale="Turbo",
                    contours=dict(start=0.5, end=1.5, size=0.1, showlines=True),
                    colorbar=dict(title="MCI", x=1.0),
                ),
                row=1,
                col=2,
            )

            fig_sim.add_trace(
                go.Scatter(
                    x=Rw * np.cos(circle_theta),
                    y=Rw * np.sin(circle_theta),
                    fill="toself",
                    fillcolor="rgba(30, 30, 30, 0.9)",
                    line=dict(color="#FFFFFF", width=2),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            fig_sim.update_layout(
                height=550,
                margin=dict(l=30, r=30, t=60, b=40),
            )
            fig_sim.update_xaxes(title_text="X (m) [SHmax Direction →]", scaleanchor="y", row=1, col=1)
            fig_sim.update_yaxes(title_text="Y (m) [Shmin Direction ↑]", row=1, col=1)
            fig_sim.update_xaxes(title_text="X (m) [SHmax Direction →]", scaleanchor="y", row=1, col=2)
            fig_sim.update_yaxes(title_text="Y (m) [Shmin Direction ↑]", row=1, col=2)

            st.plotly_chart(fig_sim, use_container_width=True)

        # ==========================================
        # TAB 3: Deviated & Horizontal Wellbore Stability
        # ==========================================
        with tab3:
            st.markdown("### 🧭 3D Wellbore Stability & Trajectory Optimization")
            st.write(
                "Analyze how borehole stability changes with **well inclination** and **azimuth** "
                "relative to the regional in-situ stress field ($S_{Hmax}$, $S_{hmin}$, $S_v$)."
            )

            col_dev_ctl1, col_dev_ctl2 = st.columns([2, 1])

            with col_dev_ctl1:
                target_depth_dev_display = st.slider(
                    f"Evaluation Depth for Trajectory Analysis ({depth_unit})",
                    float(display_depth.min()),
                    float(display_depth.max()),
                    float(display_depth.mean()),
                    step=1.0,
                    key="dev_depth_slider",
                )

            target_depth_dev_m = depth_to_meters(target_depth_dev_display, depth_unit)
            dev_row_idx = (display_depth - target_depth_dev_display).abs().idxmin()
            d_row = results_df.loc[dev_row_idx]

            d_sv = float(d_row.get("Overburden_Stress_Sv_MPa", 50.0))
            d_shmin = float(d_row.get("Shmin_MPa", 32.0))
            d_shmax = float(d_row.get("SHmax_MPa", 42.0))
            d_pp = float(d_row.get("Pore_Pressure_Pp_MPa", 22.0))
            d_ucs = float(d_row.get("UCS_MPa", 45.0))
            d_fric = float(d_row.get("Friction_Angle_deg", 30.0))

            with col_dev_ctl2:
                friction_angle_input = st.number_input(
                    "Internal Friction Angle (°)",
                    min_value=15.0,
                    max_value=50.0,
                    value=float(d_fric if not np.isnan(d_fric) else 30.0),
                    step=1.0,
                )

            # Prepare Payload for Backend
            dev_payload = {
                "results": results_df.to_dict(orient="records"),
                "depth_m": float(target_depth_dev_m),
                "well_inclination_deg": float(well_inclination),
                "well_azimuth_deg": float(well_azimuth),
                "shmax_azimuth_deg": float(shmax_azimuth),
                "friction_angle_deg": float(friction_angle_input),
                "biot_alpha": float(row_biot_alpha),
            }

            dev_result = None
            with st.spinner("Simulating 3D Wellbore Stresses on Server..."):
                try:
                    import requests
                    backend_endpoint = f"{api_client.BACKEND_URL}/api/v1/mem/deviated_stability"
                    resp = requests.post(backend_endpoint, json=dev_payload, timeout=20)
                    if resp.status_code == 200:
                        dev_result = resp.json()
                    else:
                        st.error(f"Backend calculation error ({resp.status_code}): {resp.text}")
                except Exception as exc:
                    st.error(f"Failed to communicate with calculation service: {exc}")

            if dev_result is not None and not dev_result.get("solution_found", True):
                max_tested = float(dev_result["max_tested_mud_weight_sg"])
                st.error(
                    "No safe mud weight was found in the tested range. "
                    f"The highest tested value was {max_tested:.2f} SG; it must not be treated as safe. "
                    "Review the inputs or widen the search range."
                )
            elif dev_result is not None:
                collapse_emw = float(dev_result["collapse_emw_sg"])
                thetas_deg = np.degrees(np.array(dev_result["theta_rad"]))
                sigma_tt = np.array(dev_result["sigma_tt_eff"])

                # Comparison with vertical well collapse
                vert_collapse = float(d_row.get("Collapse_EMW_SG", collapse_emw))
                delta_emw = collapse_emw - vert_collapse

                # Metrics summary
                dm1, dm2, dm3, dm4 = st.columns(4)
                dm1.metric("Selected Inclination", f"{well_inclination:.0f}°")
                dm2.metric("Selected Azimuth", f"{well_azimuth:.0f}°")
                dm3.metric("Collapse EMW", f"{collapse_emw:.2f} SG", delta=f"{delta_emw:+.2f} vs Vertical")
                dm4.metric("Max Wall Hoop Stress", f"{np.max(sigma_tt):.1f} MPa")

                # Plots: Wall Hoop Stress distribution around well circumference
                fig_dev = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "xy"}, {"type": "polar"}]],
                    subplot_titles=(
                        "Hoop Stress Distribution vs Wellbore Angle θ",
                        "Polar Stress Profile around Wellbore Wall",
                    ),
                    horizontal_spacing=0.15,
                )

                # Cartesian Track
                fig_dev.add_trace(
                    go.Scatter(
                        x=thetas_deg,
                        y=sigma_tt,
                        mode="lines",
                        line=dict(color="#FF3D00", width=2.5),
                        name="σ'θθ (Hoop Stress)",
                    ),
                    row=1,
                    col=1,
                )
                fig_dev.add_hline(
                    y=d_ucs,
                    line_dash="dot",
                    line_color="#D50000",
                    annotation_text="UCS Limit",
                    row=1,
                    col=1,
                )

                # Polar Track
                fig_dev.add_trace(
                    go.Scatterpolar(
                        r=np.maximum(sigma_tt, 0.0),
                        theta=thetas_deg,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(255, 61, 0, 0.15)",
                        line=dict(color="#FF3D00", width=2),
                        name="Polar σ'θθ",
                    ),
                    row=1,
                    col=2,
                )

                fig_dev.update_layout(
                    height=500,
                    margin=dict(l=40, r=40, t=60, b=40),
                    showlegend=False,
                )
                fig_dev.update_xaxes(title_text="Wellbore Wall Angle θ (°)", row=1, col=1)
                fig_dev.update_yaxes(title_text="Effective Hoop Stress σ'θθ (MPa)", row=1, col=1)

                st.plotly_chart(fig_dev, use_container_width=True)

                st.success(
                    f"✅ **Trajectory Evaluation at {target_depth_dev_display:.1f} {depth_unit}:** "
                    f"Minimum required mud weight to prevent borehole breakout is **{collapse_emw:.2f} SG**."
                )

    except Exception as e:
        st.error(f"Visualization rendering error: {str(e)}")
# ==========================================
# Geomechanical Reporting & Data Export
# ==========================================
st.markdown("---")
st.subheader("📑 Geomechanical Reporting & Data Export")

col_exp1, col_exp2 = st.columns([1, 1])

# پیدا کردن دیتافریم فعال حاصل از محاسبات
active_df = None
if "mem_df" in st.session_state and st.session_state["mem_df"] is not None:
    active_df = st.session_state["mem_df"]
elif "results_df" in locals() and locals()["results_df"] is not None:
    active_df = locals()["results_df"]
elif "df" in locals() and locals()["df"] is not None:
    active_df = locals()["df"]

if active_df is not None:
    # 1. دانلود گزارش رسمی PDF با تابع موجود در ریپازیتوری شما
    with col_exp1:
        try:
            import sys
            import os
            sys.path.append(os.path.abspath("backend"))
            from report_generator import generate_pdf_report
            
            well_id = "TEST_WELL_A-1"
            if uploaded_file is not None:
                well_id = uploaded_file.name.rsplit(".", 1)[0]
                
            pdf_bytes = generate_pdf_report(active_df, well_name=well_id, depth_unit=depth_unit)
            
            st.download_button(
                label="📄 Download 1D MEM Report (PDF)",
                data=pdf_bytes,
                file_name=f"{well_id}_MEM_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

    # 2. دانلود فایل CSV داده‌ها
    with col_exp2:
        export_df = active_df.copy()
        if "Depth" in export_df.columns:
            export_df["Depth"] = depth_from_meters(export_df["Depth"], depth_unit)
            export_df.rename(columns={"Depth": f"Depth ({depth_unit})"}, inplace=True)
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Processed Log Data (CSV)",
            data=csv_data,
            file_name=f"{well_id}_Log_Data.csv",
            mime="text/csv",
            use_container_width=True
        )
