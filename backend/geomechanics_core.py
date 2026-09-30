import io
import math
import numpy as np
import pandas as pd
import lasio


class GeomechanicsCore:
    """
    GeoMech Pro Computational Core:
    1D Mechanical Earth Model (MEM) + Mud Weight Window (MWW) engine.
    Fully calibrated, noise-resilient, and physically constrained.
    """

    # ---------- Well-log file loaders ----------
    def load_well_log(self, file_bytes, filename=""):
        """Load a LAS, CSV, TXT, or XLSX well-log export."""
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension == "xlsx":
            try:
                df = pd.read_excel(io.BytesIO(file_bytes))
            except ImportError as exc:
                raise ValueError(
                    "Excel support requires openpyxl. Install backend requirements first."
                ) from exc
            except Exception as exc:
                raise ValueError(f"Could not read Excel workbook: {exc}") from exc
            return self._clean_tabular_dataframe(df, "Excel workbook")

        text = file_bytes.decode("utf-8-sig", errors="ignore")

        if extension == "csv":
            return self._load_delimited_table(text, "CSV")

        try:
            return self._load_las_text(text)
        except Exception:
            return self._load_ascii_fallback(text)

    def load_las(self, file_bytes):
        return self.load_well_log(file_bytes)

    def _load_las_text(self, text):
        las = lasio.read(io.StringIO(text))
        df = las.df()
        df.reset_index(inplace=True)

        if not df.columns[0] or str(df.columns[0]).lower() == "index":
            df.rename(columns={df.columns[0]: "DEPT"}, inplace=True)

        df = df.apply(pd.to_numeric, errors="coerce")

        null_candidates = {-999.25, -9999.0, -999.0, 999.25, 9999.0}
        try:
            header_null = las.well["NULL"].value
            if header_null is not None:
                null_candidates.add(float(header_null))
        except Exception:
            pass
        df.replace(list(null_candidates), np.nan, inplace=True)

        if df.empty or df.shape[1] < 2:
            raise ValueError("Parsed LAS file produced no usable curves.")

        return df, list(df.columns)

    def _load_delimited_table(self, text, source_name):
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:
            raise ValueError(f"Could not parse {source_name} data: {exc}") from exc
        return self._clean_tabular_dataframe(df, source_name)

    def _clean_tabular_dataframe(self, df, source_name):
        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]
        df = df.loc[:, [column for column in df.columns if column]]
        if df.empty or df.shape[1] < 2:
            raise ValueError(f"{source_name} produced no usable curves.")
        df = df.apply(pd.to_numeric, errors="coerce")
        df.replace([-999.25, -9999.0, -999.0, 999.25, 9999.0], np.nan, inplace=True)
        if df.dropna(how="all").empty:
            raise ValueError(f"{source_name} contains no numeric log data.")
        return df, list(df.columns)

    def _load_ascii_fallback(self, text):
        lines = text.splitlines()
        curve_names = []
        data_lines = []
        in_data = False
        in_curve_section = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("~A"):
                in_data = True
                continue
            if stripped.startswith("~C"):
                in_curve_section = True
                continue
            if in_curve_section:
                if stripped.startswith("~") or stripped == "":
                    in_curve_section = False
                    continue
                if not stripped.startswith("#"):
                    name = stripped.split(".")[0].strip()
                    if name:
                        curve_names.append(name)
                continue
            if in_data:
                if stripped and not stripped.startswith("#"):
                    data_lines.append(stripped)

        rows = []
        for dl in data_lines:
            parts = dl.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

        if not rows:
            candidate_lines = [l for l in lines if l.strip()]
            if not candidate_lines:
                raise ValueError("No numeric data could be parsed from this file.")
            curve_names = candidate_lines[0].replace(",", " ").split()
            for dl in candidate_lines[1:]:
                parts = dl.replace(",", " ").split()
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue

        n_cols = len(rows[0]) if rows else 0
        if not curve_names or len(curve_names) != n_cols:
            curve_names = [f"Curve_{i+1}" for i in range(n_cols)]

        df = pd.DataFrame(rows, columns=curve_names)
        return self._clean_tabular_dataframe(df, "text export")

    # ---------- 1D MEM Engine ----------
    def compute_1d_mem(
        self,
        df,
        depth_col,
        dt_col,
        rhob_col,
        dts_col=None,
        biot_alpha=1.0,
        dt_matrix=55.5,
        dt_fluid=189.0,
        eaton_n=3.0,
        tectonic_ex=0.0,      # Corrected default to zero-strain baseline
        tectonic_ey=0.0,      # Corrected default to zero-strain baseline
        sonic_unit="us/ft",
        density_unit="g/cm3",
        gassi_poisson=0.40,
        gassi_min_poisson=0.20,
        friction_angle=30.0,
        normal_pressure_grad=9.80665e-3,  # MPa/m (fresh water)
        assumed_shallow_density=2.0,       # g/cm3
    ):
        """
        Full 1D MEM computation with calibrated poroelastic horizontal stresses
        and robust Eaton pore pressure lower-bounding.
        """
        out = pd.DataFrame()
        out["Depth"] = pd.to_numeric(df[depth_col], errors="coerce")
        depth_m = out["Depth"].values

        # --- Unit normalization ---
        dt = pd.to_numeric(df[dt_col], errors="coerce").values.copy()
        if sonic_unit == "us/m":
            dt = dt / 3.28084

        rhob = pd.to_numeric(df[rhob_col], errors="coerce").values.copy()
        if density_unit == "kg/m3":
            rhob = rhob / 1000.0

        # --- Outlier rejection & Quality Control (QC) ---
        rhob = np.where((rhob < 1.0) | (rhob > 3.6), np.nan, rhob)
        dt = np.where((dt < 35.0) | (dt > 250.0), np.nan, dt)

        # Interpolate small internal NaNs if available to preserve continuity
        s_rhob = pd.Series(rhob).interpolate(method="linear", limit=5).bfill().ffill()
        s_dt = pd.Series(dt).interpolate(method="linear", limit=5).bfill().ffill()
        rhob = s_rhob.values
        dt = s_dt.values

        # --- Shear sonic (or Castagna estimation) ---
        if dts_col is not None:
            dts = pd.to_numeric(df[dts_col], errors="coerce").values.copy()
            if sonic_unit == "us/m":
                dts = dts / 3.28084
            dts = np.where((dts < 35.0) | (dts > 500.0), np.nan, dts)
            dts = pd.Series(dts).interpolate(method="linear", limit=5).bfill().ffill().values
        else:
            dts = None

        def castagna_dts(dt_v):
            vp_ft_s = 1e6 / np.where(dt_v > 0, dt_v, np.nan)
            vs_ft_s = 0.8621 * vp_ft_s - 1172.4
            vs_ft_s = np.maximum(vs_ft_s, 100.0)
            return 1e6 / vs_ft_s

        if dts is None:
            dts = castagna_dts(dt)

        # --- Dynamic Elastic Moduli (GPa) ---
        rho_kg = rhob * 1000.0
        vp_m = (1e6 / dt) * 0.3048
        vs_m = (1e6 / dts) * 0.3048
        vp_m[vp_m <= 0] = np.nan
        vs_m[vs_m <= 0] = np.nan

        mu_dyn = rho_kg * vs_m**2 / 1e9          # GPa (Shear)
        lam = rho_kg * (vp_m**2 - 2.0 * vs_m**2) / 1e9
        nu_dyn = lam / (2.0 * (lam + mu_dyn))
        e_dyn = mu_dyn * (3.0 * lam + 2.0 * mu_dyn) / (lam + mu_dyn)
        k_dyn = lam + (2.0 / 3.0) * mu_dyn

        # Static correction
        out["Youngs_Modulus_GPa"] = np.clip(0.7 * e_dyn, 0.5, 120.0)
        out["Shear_Modulus_GPa"] = np.clip(0.7 * mu_dyn, 0.2, 50.0)
        out["Bulk_Modulus_GPa"] = np.clip(0.7 * k_dyn, 0.5, 150.0)
        out["Poisson_Ratio"] = np.clip(nu_dyn, 0.10, 0.45)

        # --- Rock Strength ---
        out["UCS_MPa"] = np.clip(0.77 * (out["Youngs_Modulus_GPa"].values * 1000.0) ** 0.91 / 100.0, 1.0, 350.0)
        out["Tensile_Strength_MPa"] = np.clip(out["UCS_MPa"].values / 12.0, 0.1, 30.0)
        out["Friction_Angle_deg"] = float(friction_angle)

        # --- Overburden Stress Sv (MPa) ---
        valid = ~np.isnan(rhob)
        sv = np.full_like(depth_m, np.nan, dtype=float)
        if valid.any():
            first_logged_depth = depth_m[valid][0]
            surface_offset_mpa = (
                (assumed_shallow_density * 1000.0) * 9.80665 * max(first_logged_depth, 0.0) / 1e6
            )
            rho_si = rhob[valid] * 1000.0
            dz = np.diff(depth_m[valid], prepend=depth_m[valid][0])
            sv_valid = surface_offset_mpa + np.cumsum(rho_si * 9.80665 * dz) / 1e6
            sv[valid] = sv_valid
            sv[~valid] = np.interp(depth_m[~valid], depth_m[valid], sv_valid)
        out["Overburden_Stress_Sv_MPa"] = sv

        # --- Hydrostatic Pressure Baseline ---
        p_hydro = normal_pressure_grad * np.maximum(depth_m, 0.0)

        # --- Pore Pressure (Eaton's Sonic Method) ---
        # dt_normal represents standard compaction trend line
        c_compaction = 0.0003  # 1/m empirical compaction factor
        dt_surface = 180.0     # us/ft typical surface compaction value
        dt_normal = dt_matrix + (dt_surface - dt_matrix) * np.exp(-c_compaction * depth_m)

        # Eaton acoustic ratio (observed vs normal trend)
        ratio = np.maximum(dt_normal / np.maximum(dt, 1e-2), 0.1)
        pp_eaton = sv - (sv - p_hydro) * (ratio ** eaton_n)

        # CRITICAL SAFEGUARD: Pp must strictly be between Hydrostatic and 0.95*Sv
        pp = np.clip(pp_eaton, p_hydro, sv * 0.95)
        out["Pore_Pressure_Pp_MPa"] = pp

        # --- Effective Stresses & Poroelastic Horizontal Stresses ---
        e_pa = out["Youngs_Modulus_GPa"].values * 1e9
        nu = out["Poisson_Ratio"].values
        sv_pa = out["Overburden_Stress_Sv_MPa"].values * 1e6
        pp_pa = pp * 1e6
        sig_v_eff = np.maximum(sv_pa - biot_alpha * pp_pa, 1e4)

        nu_eff = np.clip(nu, gassi_min_poisson, gassi_poisson)

        # Generalized Poroelastic Plane Strain Formulation:
        # Shmin_eff = (nu/(1-nu))*Sig_v + (E/(1-nu^2))*(ey + nu*ex)
        # SHmax_eff = (nu/(1-nu))*Sig_v + (E/(1-nu^2))*(ex + nu*ey)
        one_minus_nu2 = np.maximum(1.0 - nu_eff**2, 1e-4)
        term_ey = (e_pa / one_minus_nu2) * (tectonic_ey + nu_eff * tectonic_ex)
        term_ex = (e_pa / one_minus_nu2) * (tectonic_ex + nu_eff * tectonic_ey)

        shmin_eff = (nu_eff / (1.0 - nu_eff)) * sig_v_eff + term_ey
        shmax_eff = (nu_eff / (1.0 - nu_eff)) * sig_v_eff + term_ex

        shmin_calc = (shmin_eff + biot_alpha * pp_pa) / 1e6
        shmax_calc = (shmax_eff + biot_alpha * pp_pa) / 1e6

        # Enforce physical stress ordering: Pp < Shmin <= SHmax
        # In normal faulting regime: Pp < Shmin < SHmax < Sv
        out["Shmin_MPa"] = np.clip(shmin_calc, pp + 0.5, sv * 0.98)
        out["SHmax_MPa"] = np.clip(shmax_calc, out["Shmin_MPa"], sv * 1.15)

        out["Nu_Eff"] = nu_eff
        out["Sig_V_Eff_MPa"] = sig_v_eff / 1e6

        # --- Mud Weight Window (EMW in SG) ---
        def mw_sg(pressure_mpa, d):
            return pressure_mpa / (np.maximum(d, 1.0) * 9.80665e-3)

        out["Pore_Pressure_EMW_SG"] = mw_sg(pp, depth_m)

        # Shear Failure Collapse Pressure (Mohr-Coulomb around wellbore)
        phi = np.radians(float(friction_angle))
        q_mc = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
        shmin_v = out["Shmin_MPa"].values
        shmax_v = out["SHmax_MPa"].values
        p_frac_upper = shmin_v  # Losses threshold at wellbore wall

        pw_collapse = (
            (3.0 * shmax_v - shmin_v - out["UCS_MPa"].values) / (q_mc + 1.0)
        ) + (pp * (q_mc - 1.0) / (q_mc + 1.0))

        pw_collapse = np.maximum(pw_collapse, pp)
        out["Collapse_EMW_SG"] = np.clip(mw_sg(pw_collapse, depth_m), 0.8, 2.5)
        out["Shmin_EMW_SG"] = mw_sg(shmin_v, depth_m)
        out["Fracture_EMW_SG"] = np.clip(mw_sg(p_frac_upper, depth_m), out["Pore_Pressure_EMW_SG"] + 0.05, 3.0)

        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out

    # ---------- LOT/FIT Calibration ----------
    def solve_tectonic_ey_for_lot(
        self,
        results_df: pd.DataFrame,
        lot_depth: float,
        lot_pressure_mpa: float,
        biot_alpha: float = 1.0,
    ) -> dict:
        """Solves algebraically for tectonic_ey using generalized plane strain."""
        if results_df.empty:
            raise ValueError("No computed results to calibrate against.")

        idx = (results_df["Depth"] - lot_depth).abs().idxmin()
        row = results_df.loc[idx]

        nu_eff = row.get("Nu_Eff")
        e_gpa = row.get("Youngs_Modulus_GPa")
        sig_v_eff_mpa = row.get("Sig_V_Eff_MPa")
        pp_mpa = row.get("Pore_Pressure_Pp_MPa")

        if any(pd.isna(v) for v in (nu_eff, e_gpa, sig_v_eff_mpa, pp_mpa)) or e_gpa == 0:
            raise ValueError(
                f"Cannot calibrate: missing or invalid log data at the nearest available "
                f"depth ({row['Depth']:.1f} m) to the requested LOT/FIT depth ({lot_depth:.1f} m)."
            )

        e_pa = float(e_gpa) * 1e9
        sig_v_eff_pa = float(sig_v_eff_mpa) * 1e6
        pp_pa = float(pp_mpa) * 1e6
        nu_eff = float(nu_eff)

        target_shmin_eff_pa = (lot_pressure_mpa * 1e6) - biot_alpha * pp_pa
        baseline_shmin_eff_pa = (nu_eff / (1.0 - nu_eff)) * sig_v_eff_pa
        
        # Consistent with (1 - nu^2) in generalized plane strain
        tectonic_ey_solved = (target_shmin_eff_pa - baseline_shmin_eff_pa) * (1.0 - nu_eff**2) / e_pa

        achievable_min_mpa = float(pp_mpa) + 0.5
        achievable_max_mpa = float(row["Overburden_Stress_Sv_MPa"])
        within_range = achievable_min_mpa <= lot_pressure_mpa <= achievable_max_mpa

        return {
            "tectonic_ey": float(tectonic_ey_solved),
            "matched_depth": float(row["Depth"]),
            "uncalibrated_shmin_mpa": float((baseline_shmin_eff_pa + biot_alpha * pp_pa) / 1e6),
            "achievable_min_mpa": achievable_min_mpa,
            "achievable_max_mpa": achievable_max_mpa,
            "within_range": within_range,
        }
    # =========================================================
    # Deviated/Horizontal Wells Stability Patch
    # =========================================================

    @staticmethod
    def _deg2rad(x: float) -> float:
        return float(x) * math.pi / 180.0

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < eps:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return (v / n).astype(float)

    @staticmethod
    def _build_in_situ_stress_tensor(
        sv_mpa: float,
        shmin_mpa: float,
        shmax_mpa: float,
        shmax_azimuth_deg: float,
    ) -> np.ndarray:
        Sh = float(shmin_mpa)
        SH = float(shmax_mpa)
        Sv = float(sv_mpa)
        az = GeomechanicsCore._deg2rad(shmax_azimuth_deg)
        e_SH = np.array([math.sin(az), math.cos(az), 0.0], dtype=float)
        e_Sh = np.array([math.sin(az + math.pi / 2.0), math.cos(az + math.pi / 2.0), 0.0], dtype=float)
        e_z = np.array([0.0, 0.0, 1.0], dtype=float)
        sigma = (
            SH * np.outer(e_SH, e_SH)
            + Sh * np.outer(e_Sh, e_Sh)
            + Sv * np.outer(e_z, e_z)
        )
        return sigma

    @staticmethod
    def _wellbore_rotation_matrix(
        well_inclination_deg: float,
        well_azimuth_deg: float,
    ) -> np.ndarray:
        inc = GeomechanicsCore._deg2rad(well_inclination_deg)
        az = GeomechanicsCore._deg2rad(well_azimuth_deg)
        z_w = np.array(
            [math.sin(az) * math.sin(inc), math.cos(az) * math.sin(inc), math.cos(inc)],
            dtype=float
        )
        z_w = GeomechanicsCore._unit(z_w)
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, z_w))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
        x_w = np.cross(ref, z_w)
        x_w = GeomechanicsCore._unit(x_w)
        y_w = np.cross(z_w, x_w)
        y_w = GeomechanicsCore._unit(y_w)
        return np.vstack([x_w, y_w, z_w])

    @staticmethod
    def _rotate_tensor(R: np.ndarray, sigma_enu: np.ndarray) -> np.ndarray:
        return R @ sigma_enu @ R.T

    @staticmethod
    def _kirsch_wall_stresses(
        sigma_well: np.ndarray,
        pp_mpa: float,
        pw_mpa: float,
        nu: float,
        theta_rad: np.ndarray,
    ) -> dict:
        sxx, syy, szz = float(sigma_well[0, 0]), float(sigma_well[1, 1]), float(sigma_well[2, 2])
        txy, txz, tyz = float(sigma_well[0, 1]), float(sigma_well[0, 2]), float(sigma_well[1, 2])
        pp, pw = float(pp_mpa), float(pw_mpa)
        nu = float(np.clip(nu, 0.05, 0.49))
        c2, s2 = np.cos(2.0 * theta_rad), np.sin(2.0 * theta_rad)
        c1, s1 = np.cos(theta_rad), np.sin(theta_rad)

        s_rr_eff = (pw - pp) * np.ones_like(theta_rad)
        s_tt_total = (sxx + syy) - 2.0 * (sxx - syy) * c2 - 4.0 * txy * s2 - pw
        s_tt_eff = s_tt_total - pp
        s_zz_total = szz - 2.0 * nu * (sxx - syy) * c2 - 4.0 * nu * txy * s2
        tau_tz_total = 2.0 * (tyz * c1 - txz * s1)
        s_zz_eff = s_zz_total - pp

        return {
            "sigma_rr_eff": s_rr_eff,
            "sigma_tt_eff": s_tt_eff,
            "sigma_zz_eff": s_zz_eff,
            "tau_tz_total": tau_tz_total,
        }

    @staticmethod
    def _mohr_coulomb_collapse_check(
        sigma1_eff: np.ndarray,
        sigma3_eff: np.ndarray,
        ucs_mpa: float,
        friction_angle_deg: float,
    ) -> np.ndarray:
        phi = GeomechanicsCore._deg2rad(friction_angle_deg)
        q = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi) + 1e-12)
        return sigma1_eff >= (q * sigma3_eff + float(ucs_mpa))

    def compute_deviated_wellbore_stability(
        self,
        mem_df: pd.DataFrame,
        depth_m: float,
        well_inclination_deg: float,
        well_azimuth_deg: float,
        shmax_azimuth_deg: float,
        friction_angle_deg: float = 30.0,
        mud_weight_sg_min: float = 0.90,
        mud_weight_sg_max: float = 2.50,
        mud_weight_sg_step: float = 0.01,
        n_theta: int = 181,
    ) -> dict:
        if mem_df is None or mem_df.empty:
            raise ValueError("mem_df is empty.")

        idx = (mem_df["Depth"] - float(depth_m)).abs().idxmin()
        row = mem_df.loc[idx]

        sigma_enu = self._build_in_situ_stress_tensor(
            float(row["Overburden_Stress_Sv_MPa"]),
            float(row["Shmin_MPa"]),
            float(row["SHmax_MPa"]),
            shmax_azimuth_deg,
        )
        R = self._wellbore_rotation_matrix(well_inclination_deg, well_azimuth_deg)
        sigma_well = self._rotate_tensor(R, sigma_enu)

        theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), dtype=float)
        mws = np.arange(mud_weight_sg_min, mud_weight_sg_max + 0.5 * mud_weight_sg_step, mud_weight_sg_step)
        depth_use = max(float(row["Depth"]), 1.0)
        mpam_per_m = 9.80665e-3

        collapse_mw = float(mud_weight_sg_max)
        nu = float(row.get("Poisson_Ratio", 0.25))
        pp = float(row["Pore_Pressure_Pp_MPa"])
        ucs = float(row.get("UCS_MPa", 20.0))

        for mw in mws:
            Pw = float(mw) * mpam_per_m * depth_use
            kir = self._kirsch_wall_stresses(sigma_well, pp, Pw, nu, theta)
            mean = 0.5 * (kir["sigma_tt_eff"] + kir["sigma_zz_eff"])
            rad = np.sqrt((0.5 * (kir["sigma_tt_eff"] - kir["sigma_zz_eff"])) ** 2 + (kir["tau_tz_total"]) ** 2)
            s1 = mean + rad
            s3 = mean - rad
            if not np.any(self._mohr_coulomb_collapse_check(s1, s3, ucs, friction_angle_deg)):
                collapse_mw = float(mw)
                break

        # Calculate final state with the determined collapse mud weight
        final_pw = collapse_mw * mpam_per_m * depth_use
        final_kir = self._kirsch_wall_stresses(sigma_well, pp, final_pw, nu, theta)

        # Calculate final state with the determined collapse mud weight
        final_pw = collapse_mw * mpam_per_m * depth_use
        final_kir = self._kirsch_wall_stresses(sigma_well, pp, final_pw, nu, theta)

        return {
            "collapse_emw_sg": float(collapse_mw),
            "theta_rad": theta.tolist() if hasattr(theta, "tolist") else list(theta),
            "sigma_tt_eff": final_kir["sigma_tt_eff"].tolist() if hasattr(final_kir["sigma_tt_eff"], "tolist") else list(final_kir["sigma_tt_eff"]),
        }
