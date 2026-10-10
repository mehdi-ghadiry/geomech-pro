# GeoMech Pro: 1D Mechanical Earth Model & Wellbore Stability

Advanced Subsurface Geomechanics & 2D Kirsch Stress Simulation Platform.

## Overview
GeoMech Pro is an end-to-end cloud-native geomechanical software suite designed for automated 1D Mechanical Earth Model (MEM) generation, pore pressure prediction, in-situ stress profiling, and safe mud weight window (MWW) determination from wireline logs (.las, .csv, .xlsx).

## Project Structure
```text
GeoMechanics_SaaS/
├── backend/
│   ├── main.py
│   ├── geomechanics_core.py
│   ├── report_generator.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── requirements.txt
├── requirements.txt
└── README.md
```

## Input preview and calculation modes
- **Input logs only** (default): plots parsed file values before calibration,
  lithology or TVD gates. Missing samples remain gaps; no pressure/stress model
  is run. Preview units and depth reference are explicitly unverified.
- **Engineering (calibration required)**: retains the normal-sonic-trend
  calibration declaration and all existing QC checks. A declaration is not
  independent field validation.
- **Educational / experimental**: requires an educational-use acknowledgement
  and verified input units. It bypasses only the normal-trend declaration, not
  depth-reference, lithology, physical-property or pressure/stress QC. Generic
  defaults may still fail QC; they are not automatically adjusted to make plots.

Computed records retain `Calculation_Mode`, `Normal_Trend_Calibrated` and
`Result_Use_Warning`, including CSV, stored records and derived stability API
responses. Educational plots and every PDF page carry a non-engineering-use
warning. Numerical formulas are unchanged.

Run regression tests from the repository root:
```sh
python -m unittest discover -s tests -v
```
Synthetic test fixtures demonstrate software behavior, not engineering validity.
## Quick experimental testing vs. engineering mode

The UI now starts in **Educational / experimental** mode. Upload and map depth,
DT and RHOB to try the workflow without declaring field calibration. In this
mode the log range filter, pressure upper screen, stress-regime screen and
property bounds are optional. Internal-gap interpolation can be disabled or
limited by sample count. Engineering mode retains the existing screening and
normal-trend calibration requirement; declaring calibration alone does not
guarantee accurate results or operational suitability.

Experimental defaults explicitly assume a vertical well when depth is MD or
unspecified, assume Vs/Vp = 0.5 only when DTS is missing (not an empirical
lithology inference), and substitute hydrostatic pressure where the Eaton
estimate cannot be used. Each assumption can be switched off. Depth is **not**
trajectory-converted; hydrostatic substitution is **not** pressure calibration.
Selected log units must still be supported; until verified they are assumptions.

Raw Eaton results and QC flags are retained even when a screen is disabled or
a hydrostatic fallback is used. Filter settings, assumed depth/pressure/Vs and
calibration status survive CSV/saved records and are summarized in PDF reports.
Charts retain the educational-only warning. Non-finite/non-positive input logs,
unstable elastic pairs, unsupported units and unusable depth data still cannot
be converted into valid data by switching filters off. No-window results are
retained rather than made to look safe.

These defaults are for software testing, **not engineering decisions**. For
field analysis, select engineering mode, verify TVD and units, use measured
logs, calibrate the normal-compaction trend and validate against local pressure,
stress and rock-property data. LOT/FIT calibration remains optional and must
use a genuine, applicable measurement.
