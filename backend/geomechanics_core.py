import io
import numpy as np
import pandas as pd
import lasio


class GeomechanicsCore:
    """
    GeoMech Pro Computational Core:
    1D Mechanical Earth Model (MEM) + Mud Weight Window (MWW) engine.
    """

    # ---------- Well-log file loaders ----------
    def load_well_log(self, file_bytes, filename=""):
        """Load a LAS, CSV, TXT, or XLSX well-log export.

        The file extension is used only to select the most suitable parser;
        the returned value is always a numeric DataFrame plus its curve names.
        """
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

        # LAS is attempted first for .las files and extension-less uploads.
        # TXT then falls back to a lenient ASCII/CSV parser below.
        try:
            return self._load_las_text(text)
        except Exception:
            return self._load_ascii_fallback(text)

    # Kept for callers of the earlier backend API.
    def load_las(self, file_bytes):
        return self.load_well_log(file_bytes)

    def _load_las_text(self, text):
        """
        Loads a LAS well-log file (v1.2 / 2.0 / 3.0) using the `lasio`
        library, which correctly parses the standard header sections
        (~V, ~W, ~C, ~P, ~A), handles multi-line/irregular headers, and
        automatically converts the file's declared NULL value (from the
        ~W section, e.g. -999.25) to NaN -- something the previous
        hand-rolled parser only did for two hardcoded values.

        Falls back to a lenient whitespace/CSV parser for plain ASCII
        .txt exports that are not valid LAS files.

        Returns: (DataFrame, list_of_column_names)
        """
        las = lasio.read(io.StringIO(text))
        df = las.df()
        df.reset_index(inplace=True)

        # lasio names the index column after the index curve mnemonic
        # (normally DEPT/DEPTH); guard against it coming back unnamed.
        if not df.columns[0] or str(df.columns[0]).lower() == "index":
            df.rename(columns={df.columns[0]: "DEPT"}, inplace=True)

        df = df.apply(pd.to_numeric, errors="coerce")

        # --- Explicit NULL substitution (do NOT rely solely on lasio) ---
        # The file's own declared NULL value (~W section, e.g. -999.25)
        # is read directly from the header and swapped for NaN here.
        # Verified necessary: on real-world files lasio's own null handling
        # can be inconsistent, and a missed -999.25 silently propagates into
        # density/sonic values and corrupts every downstream calculation.
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
        """Read a CSV/TSV-style export with a header row."""
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:
            raise ValueError(f"Could not parse {source_name} data: {exc}") from exc
        return self._clean_tabular_dataframe(df, source_name)

    def _clean_tabular_dataframe(self, df, source_name):
        """Normalise tabular exports and reject files without usable curves."""
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

    # ---------- Fallback ASCII/whitespace parser ----------
    def _load_ascii_fallback(self, text):
        """
        Lenient parser for whitespace- or comma-delimited ASCII log data
        that doesn't carry a proper LAS header (e.g. exported .txt files).
        Still honors a LAS-style ~A / ~C section if present; otherwise
        treats the first non-empty line as a column header row.
        """
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
            # No ~A section found at all -> treat the file as a plain
            # delimited table with the first non-empty line as a header.
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
        tectonic_ex=0.0005,
        tectonic_ey=0.0002,
        sonic_unit="us/ft",
        density_unit="g/cm3",
        gassi_poisson=0.40,
        gassi_min_poisson=0.20,
        friction_angle=30.0,
        normal_pressure_grad=9.80665e-3,  # MPa/m (fresh water)
        assumed_shallow_density=2.0,  # g/cm3, used only above the top of the log
    ):
        """
        Full 1D MEM computation. Returns a DataFrame with elastic moduli,
        strength, in-situ stresses, and the mud weight window in SG.
        """
        out = pd.DataFrame()
        out["Depth"] = pd.to_numeric(df[depth_col], errors="coerce")

        depth_m = out["Depth"].values

        # --- Unit normalization ---
        dt = pd.to_numeric(df[dt_col], errors="coerce").values.copy()
        if sonic_unit == "us/m":
            dt = dt / 3.28084  # convert to us/ft

        rhob = pd.to_numeric(df[rhob_col], errors="coerce").values.copy()
        if density_unit == "kg/m3":
            rhob = rhob / 1000.0  # convert to g/cm3

        # --- Physical sanity guard (defense-in-depth) ---
        # Never trust the loader alone to have stripped every NULL/bad
        # value. A leftover sensor-error or unconverted NULL marker (e.g.
        # RHOB = -999.25 g/cm3) would otherwise silently blow up every
        # downstream elastic-modulus and stress calculation for that
        # depth (and, through auto-scaled chart axes, distort the whole
        # log display). Values outside a physically plausible range are
        # rejected here as NaN instead of being computed with.
        rhob = np.where((rhob < 1.0) | (rhob > 3.6), np.nan, rhob)  # g/cm3
        dt = np.where((dt < 30.0) | (dt > 300.0), np.nan, dt)  # us/ft

        # --- Shear sonic (or Castagna estimation) ---
        if dts_col is not None:
            dts = pd.to_numeric(df[dts_col], errors="coerce").values.copy()
            if sonic_unit == "us/m":
                dts = dts / 3.28084
            dts = np.where((dts < 30.0) | (dts > 600.0), np.nan, dts)  # us/ft
        else:
            dts = None  # will be estimated

        # Castagna: Vs = 0.8621*Vp - 1172  =>  DTS_est
        def castagna_dts(dt_v):
            vp_ft_s = 1e6 / np.where(dt_v > 0, dt_v, np.nan)
            vs_ft_s = 0.8621 * vp_ft_s - 1172.4
            vs_ft_s = np.maximum(vs_ft_s, 1.0)
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

        # Static correction (common empirical: E_static ~ 0.7 * E_dynamic)
        out["Youngs_Modulus_GPa"] = 0.7 * e_dyn
        out["Shear_Modulus_GPa"] = 0.7 * mu_dyn
        out["Bulk_Modulus_GPa"] = 0.7 * k_dyn
        out["Poisson_Ratio"] = np.clip(nu_dyn, 0.05, 0.49)

        # --- Rock Strength (empirical correlations) ---
        # UCS from E (MPa): Modifyd Hassanabad-style correlation
        out["UCS_MPa"] = np.clip(0.77 * (out["Youngs_Modulus_GPa"].values * 1000.0) ** 0.91 / 100.0, 1.0, 500.0)
        out["Tensile_Strength_MPa"] = np.clip(out["UCS_MPa"].values / 12.0, 0.1, 25.0)
        out["Friction_Angle_deg"] = float(friction_angle)

        # --- Overburden Stress Sv (MPa) via density integration ---
        valid = ~np.isnan(rhob)
        sv = np.full_like(depth_m, np.nan, dtype=float)
        if valid.any():
            first_logged_depth = depth_m[valid][0]
            # The log almost never starts at the surface (this file starts
            # at 1000 m). Rock above that depth still weighs on everything
            # below it -- ignoring it (the previous behavior) understates
            # Sv by tens of MPa, which then artificially caps how high
            # Shmin/SHmax are allowed to go. Since there's no density log
            # for that shallow interval, estimate it with a typical
            # near-surface sediment density (default 2.0 g/cm3, adjustable).
            surface_offset_mpa = (
                (assumed_shallow_density * 1000.0) * 9.80665 * max(first_logged_depth, 0.0) / 1e6
            )
            rho_si = rhob[valid] * 1000.0
            dz = np.diff(depth_m[valid], prepend=depth_m[valid][0])
            sv_valid = surface_offset_mpa + np.cumsum(rho_si * 9.80665 * dz) / 1e6
            sv[valid] = sv_valid
            # extrapolate to top
            sv[~valid] = np.interp(depth_m[~valid], depth_m[valid], sv_valid)
        out["Overburden_Stress_Sv_MPa"] = sv

        # --- Pore Pressure (Eaton's Sonic Method) ---
        if dt_matrix >= 100.0:
            dt_matrix = 55.5  # sanity guard: input expected in us/ft
        ratio = np.maximum((dt - dt_matrix) / np.maximum(dt - dt_fluid, 1e-3), 0.01)
        pp_eaton = sv - (sv - normal_pressure_grad * depth_m) * (ratio ** (-eaton_n))
        pp = np.clip(pp_eaton, 0.1 * np.maximum(normal_pressure_grad * depth_m, 0.1), sv * 0.95)
        out["Pore_Pressure_Pp_MPa"] = pp

        # --- Effective Stresses & Horizontal Stresses (Por-Eaton + Tectonic Strains) ---
        e_pa = out["Youngs_Modulus_GPa"].values * 1e9
        nu = out["Poisson_Ratio"].values
        sv_pa = out["Overburden_Stress_Sv_MPa"].values * 1e6
        pp_pa = pp * 1e6
        sig_v_eff = np.maximum(sv_pa - biot_alpha * pp_pa, 1e4)

        nu_eff = np.clip(nu, gassi_min_poisson, gassi_poisson)
        shmin_eff = (nu_eff / (1.0 - nu_eff)) * sig_v_eff + (e_pa / (1.0 - nu_eff)) * tectonic_ey
        shmax_eff = (nu_eff / (1.0 - nu_eff)) * sig_v_eff + (e_pa / (1.0 - nu_eff)) * tectonic_ex

        out["Shmin_MPa"] = (shmin_eff + biot_alpha * pp_pa) / 1e6
        out["SHmax_MPa"] = (shmax_eff + biot_alpha * pp_pa) / 1e6

        # Exposed so LOT/FIT calibration (see solve_tectonic_ey_for_lot) can
        # solve for tectonic_ey algebraically without recomputing everything
        # from raw logs. Neither depends on tectonic_ex/ey, so they're valid
        # regardless of what those two were set to for this particular run.
        out["Nu_Eff"] = nu_eff
        out["Sig_V_Eff_MPa"] = sig_v_eff / 1e6

        # Order sanity: Pp < Shmin < SHmax < Sv
        out["Shmin_MPa"] = np.clip(out["Shmin_MPa"], out["Pore_Pressure_Pp_MPa"] + 0.5, out["Overburden_Stress_Sv_MPa"])
        out["SHmax_MPa"] = np.clip(out["SHmax_MPa"], out["Shmin_MPa"], out["Overburden_Stress_Sv_MPa"] * 1.05)

        # --- Mud Weight Window (EMW in SG) ---
        def mw_sg(pressure_mpa, d):
            return pressure_mpa / (np.maximum(d, 1.0) * 9.80665e-3)

        out["Pore_Pressure_EMW_SG"] = mw_sg(pp, depth_m)

        # Shear Collapse (Mohr-Coulomb around the wellbore, simplified)
        phi = np.radians(float(friction_angle))
        q_mc = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
        shmin_v = out["Shmin_MPa"].values
        shmax_v = out["SHmax_MPa"].values
        p_frac_upper = shmin_v  # losses threshold
        # Collapse pressure: min mud pressure preventing breakout
        pw_collapse = (
            (2.0 * shmin_v - shmax_v - out["UCS_MPa"].values * (q_mc - 1.0))
            + (q_mc + 1.0) * pp
        ) / (q_mc + 1.0)
        out["Collapse_EMW_SG"] = np.clip(mw_sg(np.maximum(pw_collapse, pp), depth_m), 0.5, 3.5)
        out["Shmin_EMW_SG"] = mw_sg(shmin_v, depth_m)
        out["Fracture_EMW_SG"] = np.clip(mw_sg(p_frac_upper, depth_m), out["Pore_Pressure_EMW_SG"] + 0.05, 4.0)

        # Clean non-finite values
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
        """
        Solves for the tectonic_ey (minimum-horizontal-stress tectonic
        strain) that makes the model's Shmin at `lot_depth` equal a
        field-measured LOT/FIT pressure.

        Why this works without iterating: Shmin_eff is LINEAR in
        tectonic_ey --

            Shmin_eff = (nu_eff / (1-nu_eff)) * Sig_V_Eff
                        + (E / (1-nu_eff)) * tectonic_ey

        -- and every other term (nu_eff, E, Sig_V_Eff, Pp) comes straight
        from the well logs and doesn't depend on tectonic_ey/ex at all.
        So this is a direct algebraic solve, not a numerical search: run
        compute_1d_mem() once (any tectonic_ey), then call this with the
        LOT/FIT depth and pressure to get the ey that reproduces it.

        Returns a dict with the solved tectonic_ey, the matched depth
        actually used (nearest available to lot_depth), and the
        uncalibrated (tectonic-term-free) Shmin at that depth for context.
        Raises ValueError if the calibration depth has no usable log data.
        """
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
        baseline_shmin_eff_pa = (nu_eff / (1.0 - nu_eff)) * sig_v_eff_pa  # tectonic-free part
        tectonic_ey_solved = (target_shmin_eff_pa - baseline_shmin_eff_pa) * (1.0 - nu_eff) / e_pa

        # The model clips final Shmin to [Pp + 0.5, Sv] (see compute_1d_mem)
        # to keep the stress ordering physically sane. If the entered
        # LOT/FIT pressure falls outside that band at this depth, solving
        # for tectonic_ey still "succeeds" mathematically, but the actual
        # recomputed Shmin will be clamped and NOT equal the target -- so
        # the caller needs to know that up front rather than silently
        # getting a mismatched result.
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
