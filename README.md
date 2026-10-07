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
