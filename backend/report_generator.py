from fpdf import FPDF
import numpy as np
import pandas as pd

from config import PLATFORM_NAME, DEVELOPER_EMAIL, DEVELOPER_WHATSAPP


def _format_stat(value, decimals: int) -> str:
    """Format a finite report statistic, or mark unavailable QC values clearly."""
    if pd.isna(value) or not np.isfinite(float(value)):
        return "N/A (QC)"
    return f"{float(value):.{decimals}f}"


def _column_stat(results_df: pd.DataFrame, column: str, operation: str):
    if column not in results_df:
        return np.nan
    values = pd.to_numeric(results_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return getattr(values, operation)()


def _pdf_safe(text: str) -> str:
    """
    fpdf's built-in core fonts (Helvetica, etc.) only support the Latin-1
    character set. Any character outside it -- Persian/Arabic, CJK,
    emoji, curly quotes typed by some editors -- crashes PDF generation
    with a UnicodeEncodeError instead of just looking wrong. This keeps
    report generation from failing: unsupported characters become '?'.

    Native rendering of non-Latin scripts (e.g. an actual Persian well
    name shown in Persian) would need a bundled Unicode TrueType font,
    which isn't included here.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class PDFReport(FPDF):
    def header(self):
        # Header banner
        self.set_fill_color(11, 19, 43)  # Navy Blue #0B132B
        self.rect(0, 0, 210, 26, "F")

        self.set_font("helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 8, "GEO-MECHANICAL ENGINEERING REPORT (1D MEM)", 0, 1, "L")

        self.set_font("helvetica", "I", 9)
        self.set_text_color(200, 220, 255)
        self.set_xy(10, 14)
        self.cell(0, 6, f"{PLATFORM_NAME} | Email : {DEVELOPER_EMAIL} | WhatsApp : {DEVELOPER_WHATSAPP} ", 0, 1, "L")
        self.ln(15)

    def footer(self):
        self.set_y(-18)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Developer gmail : {DEVELOPER_EMAIL} ")



def generate_pdf_report(
    results_df: pd.DataFrame,
    well_name: str = "Well-01",
    depth_unit: str = "m",
) -> bytes:
    """
    Generates a professional engineering PDF report including 1D MEM,
    Rock Strength, In-Situ Stresses, and Wellbore Stability MWW summary.
    """
    well_name_safe = _pdf_safe(well_name)
    if depth_unit not in {"m", "ft"}:
        raise ValueError("depth_unit must be 'm' or 'ft'")
    feet_per_metre = 3.280839895013123
    depth_factor = feet_per_metre if depth_unit == "ft" else 1.0
    depth_min = results_df["Depth"].min() * depth_factor
    depth_max = results_df["Depth"].max() * depth_factor

    pdf = PDFReport()
    pdf.well_name = well_name_safe
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title & Metadata
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(11, 19, 43)
    pdf.cell(0, 8, f"Well Identifier: {well_name_safe}", 0, 1, "L")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0,
        6,
        f"Analyzed Depth Interval: {depth_min:.1f} {depth_unit} to {depth_max:.1f} {depth_unit} (Total Samples: {len(results_df)})",
        0,
        1,
        "L",
    )
    pdf.ln(4)

    # Section 1: Executive Summary
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(11, 19, 43)
    pdf.cell(0, 7, "1. Executive Summary & Geomechanical Properties", 0, 1, "L")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    mean_ucs = _column_stat(results_df, "UCS_MPa", "mean")
    max_pp = _column_stat(results_df, "Pore_Pressure_Pp_MPa", "max")
    mean_shmin = _column_stat(results_df, "Shmin_MPa", "mean")
    mean_shmax = _column_stat(results_df, "SHmax_MPa", "mean")
    mean_sv = _column_stat(results_df, "Overburden_Stress_Sv_MPa", "mean")
    mean_collapse_emw = _column_stat(results_df, "Collapse_EMW_SG", "mean")
    mean_frac_emw = _column_stat(results_df, "Fracture_EMW_SG", "mean")

    pressure_usable = results_df.get(
        "Pore_Pressure_Estimate_Usable", pd.Series(0, index=results_df.index)
    ).fillna(0).astype(bool)
    flagged_pressure_samples = int((~pressure_usable).sum())
    pressure_qc_note = (
        f"Pressure QC flagged {flagged_pressure_samples} of {len(results_df)} samples. "
        "Raw Eaton estimates are retained for diagnosis; pressure-derived outputs are withheld at flagged depths. "
        "Calibrate the sonic trend against well-specific measurements before operational use."
        if flagged_pressure_samples
        else "Pressure QC found no samples outside the configured screening bounds; field calibration is still required."
    )

    summary_text = (
        f"This automated technical report presents the 1D Mechanical Earth Model (MEM) and Wellbore Stability analysis "
        f"for {well_name_safe}. Across the evaluated interval, the formation exhibits an average Unconfined Compressive Strength (UCS) "
        f"of {_format_stat(mean_ucs, 1)} MPa. Pore pressure reaches a maximum of {_format_stat(max_pp, 1)} MPa. "
        f"In-situ stress diagnostics indicate an average Overburden Stress (Sv) of {_format_stat(mean_sv, 1)} MPa, "
        f"Minimum Horizontal Stress (Shmin) of {_format_stat(mean_shmin, 1)} MPa, and Maximum Horizontal Stress (SHmax) of {_format_stat(mean_shmax, 1)} MPa.\n"
        f"{pressure_qc_note}"
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(6)

    # Section 2: Key Statistics Table
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 7, "2. Geomechanical Parameter Statistics", 0, 1, "L")
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(240, 243, 246)
    pdf.set_text_color(11, 19, 43)

    # Table Header
    pdf.cell(70, 7, "Parameter", 1, 0, "L", True)
    pdf.cell(40, 7, "Minimum", 1, 0, "C", True)
    pdf.cell(40, 7, "Average", 1, 0, "C", True)
    pdf.cell(40, 7, "Maximum", 1, 1, "C", True)

    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)

    params = [
        ("Young's Modulus (GPa)", "Youngs_Modulus_GPa"),
        ("Poisson's Ratio (-)", "Poisson_Ratio"),
        ("UCS (MPa)", "UCS_MPa"),
        ("Pore Pressure (MPa)", "Pore_Pressure_Pp_MPa"),
        ("Minimum Horizontal Stress (MPa)", "Shmin_MPa"),
        ("Maximum Horizontal Stress (MPa)", "SHmax_MPa"),
        ("Overburden Stress (MPa)", "Overburden_Stress_Sv_MPa"),
        ("Shear Collapse EMW (SG)", "Collapse_EMW_SG"),
        ("Fracture Breakdown EMW (SG)", "Fracture_EMW_SG"),
    ]

    for label, col in params:
        if col in results_df.columns:
            vmin = _column_stat(results_df, col, "min")
            vmean = _column_stat(results_df, col, "mean")
            vmax = _column_stat(results_df, col, "max")
            pdf.cell(70, 6, label, 1, 0, "L")
            pdf.cell(40, 6, _format_stat(vmin, 2), 1, 0, "C")
            pdf.cell(40, 6, _format_stat(vmean, 2), 1, 0, "C")
            pdf.cell(40, 6, _format_stat(vmax, 2), 1, 1, "C")

    pdf.ln(6)

    # Section 3: Wellbore Stability & Drilling Window
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(11, 19, 43)
    pdf.cell(0, 7, "3. Wellbore Stability & Mud Weight Window (MWW)", 0, 1, "L")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    if np.isfinite(mean_collapse_emw) and np.isfinite(mean_frac_emw):
        mww_text = (
            "Based on the Mohr-Coulomb failure criterion and elastic stress distribution around a vertical wellbore:\n"
            f"- Shear Failure (Collapse) Gradient: averages {_format_stat(mean_collapse_emw, 2)} SG.\n"
            f"- Tensile Failure (Fracture) Gradient: averages {_format_stat(mean_frac_emw, 2)} SG.\n"
            f"- Screening window from valid samples only: [{_format_stat(mean_collapse_emw, 2)} SG - "
            f"{_format_stat(mean_frac_emw, 2)} SG]. {pressure_qc_note}"
        )
    else:
        mww_text = (
            "Operational mud-weight window and recommendation are withheld because no usable pressure-dependent "
            "stability results are available. Review the pressure QC flags and calibrate the sonic trend before "
            "recomputing. " + pressure_qc_note
        )
    pdf.multi_cell(0, 5, mww_text)
    pdf.ln(8)

    # Signature & Signoff
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Report generated autonomously via {PLATFORM_NAME} ", 0, 1, "L")

    # Return bytes
    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        # errors="replace" is a last line of defense -- every well_name
        # insertion above already goes through _pdf_safe(), but this
        # guarantees the function never crashes even if something else
        # (a future edit, a new field) slips an unsupported character in.
        pdf_output = pdf_output.encode("latin-1", errors="replace")
    return bytes(pdf_output)
