"""Input-only plotting and persistent educational chart labels."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EDUCATIONAL_WARNING = "EDUCATIONAL ONLY - NOT FOR ENGINEERING DECISIONS"
TEXT_METADATA = {"Calculation_Mode", "Result_Use_Warning", "Depth_Reference_Used", "Pore_Pressure_Source"}


def find_shear_curve(columns):
    """Never substitute depth or compressional sonic for an absent shear log."""
    for candidate in ("dts", "dtsm", "dt_s", "dtshear"):
        for column in columns:
            if str(column).strip().lower() == candidate:
                return column
    return None


def is_educational(results):
    return ("Calculation_Mode" in results and
            results["Calculation_Mode"].eq("educational").any())


def label_educational_chart(fig):
    fig.add_annotation(
        text=EDUCATIONAL_WARNING + "<br>Calibration not independently validated", x=0.5, y=0.01, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color="#B91C1C"),
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#B91C1C", borderpad=6,
    )
    return fig


def raw_log_figure(records, depth_column, curve_columns):
    """Plot in file row order with gaps; units and depth reference remain unverified."""
    if not curve_columns:
        raise ValueError("Select at least one input curve.")
    df = pd.DataFrame(records)
    depth = pd.to_numeric(df[depth_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fig = make_subplots(rows=1, cols=len(curve_columns), shared_yaxes=True,
                        subplot_titles=curve_columns)
    for i, column in enumerate(curve_columns, 1):
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fig.add_trace(go.Scatter(x=values, y=depth, name=column, connectgaps=False), row=1, col=i)
        fig.update_xaxes(title_text=f"{column} (file values)", row=1, col=i)
    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(title_text=f"{depth_column} (file values)", row=1, col=1)
    fig.update_layout(height=650, title="INPUT LOG PREVIEW - units / depth reference unverified",
                      margin=dict(t=100, b=65), showlegend=False)
    fig.add_annotation(text="Parsed input only - no MEM calculation or field validation",
                       x=0.5, y=-0.1, xref="paper", yref="paper", showarrow=False)
    return fig
