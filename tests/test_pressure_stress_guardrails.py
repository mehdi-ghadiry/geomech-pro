import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class PressureStressGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()

    def test_vertical_collapse_formula_includes_biot_alpha(self):
        phi_deg = 30.0
        alpha = 0.6
        shmax, shmin, ucs, pp = 75.0, 45.0, 20.0, 30.0
        q = (1.0 + np.sin(np.radians(phi_deg))) / (1.0 - np.sin(np.radians(phi_deg)))
        expected = (3.0 * shmax - shmin - ucs + alpha * pp * (q - 1.0)) / (q + 1.0)
        actual = self.core._vertical_collapse_pressure(shmax, shmin, ucs, pp, phi_deg, alpha)
        self.assertAlmostEqual(float(actual), expected)

    def test_alpha_one_reduces_to_previous_formula(self):
        phi_deg = 30.0
        shmax, shmin, ucs, pp = 75.0, 45.0, 20.0, 30.0
        q = (1.0 + np.sin(np.radians(phi_deg))) / (1.0 - np.sin(np.radians(phi_deg)))
        previous = (3.0 * shmax - shmin - ucs + pp * (q - 1.0)) / (q + 1.0)
        actual = self.core._vertical_collapse_pressure(shmax, shmin, ucs, pp, phi_deg, 1.0)
        self.assertAlmostEqual(float(actual), previous)

    def test_1d_solver_passes_biot_alpha_to_collapse_relation(self):
        logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0, 1020.0],
            "DT": [90.0, 90.0, 90.0],
            "RHOB": [2.20, 2.21, 2.22],
            "DTS": [180.0, 181.0, 182.0],
        })
        with patch.object(
            self.core,
            "_vertical_collapse_pressure",
            wraps=self.core._vertical_collapse_pressure,
        ) as collapse:
            self.core.compute_1d_mem(
                df=logs,
                depth_col="TVD",
                dt_col="DT",
                rhob_col="RHOB",
                dts_col="DTS",
                depth_reference="TVD",
                biot_alpha=0.6,
                dt_surface=80.0,
                dt_matrix=55.5,
                compaction_coefficient=0.0003,
                normal_trend_calibrated=True,
            )
        self.assertEqual(collapse.call_count, 1)
        self.assertEqual(collapse.call_args.args[-1], 0.6)

    def _stability_row(self, **overrides):
        row = {
            "Depth": 2000.0,
            "Biot_Coefficient": 0.65,
            "Overburden_Stress_Sv_MPa": 50.0,
            "Shmin_MPa": 30.0,
            "SHmax_MPa": 40.0,
            "Pore_Pressure_Pp_MPa": 20.0,
            "Poisson_Ratio": 0.25,
            "UCS_MPa": 25.0,
        }
        row.update(overrides)
        return row

    def test_deviated_solver_rejects_qc_failed_pressure_before_search(self):
        row = self._stability_row(
            Pore_Pressure_Pp_MPa=np.nan,
            Pore_Pressure_Estimate_Valid=0,
            Pore_Pressure_QC_Flag=1,
        )
        with self.assertRaisesRegex(ValueError, "QC failed"):
            self.core.compute_deviated_wellbore_stability(
                pd.DataFrame([row]), 2000.0, 0.0, 0.0, 0.0,
                mud_weight_sg_min=0.9, mud_weight_sg_max=1.0, mud_weight_sg_step=0.05,
            )

    def test_deviated_solver_rejects_nonfinite_pressure_without_qc_columns(self):
        row = self._stability_row(Pore_Pressure_Pp_MPa=np.nan)
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            self.core.compute_deviated_wellbore_stability(
                pd.DataFrame([row]), 2000.0, 0.0, 0.0, 0.0,
                mud_weight_sg_min=0.9, mud_weight_sg_max=1.0, mud_weight_sg_step=0.05,
            )

    def test_deviated_solver_rejects_nonzero_qc_flag_even_if_pressure_is_finite(self):
        row = self._stability_row(Pore_Pressure_QC_Flag=1)
        with self.assertRaisesRegex(ValueError, "QC failed"):
            self.core.compute_deviated_wellbore_stability(
                pd.DataFrame([row]), 2000.0, 0.0, 0.0, 0.0,
                mud_weight_sg_min=0.9, mud_weight_sg_max=1.0, mud_weight_sg_step=0.05,
            )


if __name__ == "__main__":
    unittest.main()
