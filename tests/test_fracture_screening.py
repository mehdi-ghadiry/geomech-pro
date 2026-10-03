import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.geomechanics_core import GeomechanicsCore
from report_generator import _full_interval_screening_window


class FractureScreeningTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()

    def test_breakdown_pressure_matches_current_kirsch_wall_criterion(self):
        shmin, shmax, t0, pp, alpha = 40.0, 60.0, 4.0, 25.0, 0.7
        expected = 3.0 * shmin - shmax + t0 - alpha * pp
        actual = self.core._vertical_tensile_breakdown_pressure(shmin, shmax, t0, pp, alpha)
        self.assertAlmostEqual(float(actual), expected)
        sigma_well = np.diag([shmin, shmax, 80.0])
        theta = np.array([np.pi / 2.0])
        wall = self.core._kirsch_wall_stresses(sigma_well, pp, float(actual), 0.25, alpha, theta)
        self.assertAlmostEqual(float(wall["sigma_tt_eff"][0]), -t0)

    def test_alpha_one_reduces_to_classic_hubbert_willis_form(self):
        shmin, shmax, t0, pp = 40.0, 60.0, 4.0, 25.0
        expected = 3.0 * shmin - shmax + t0 - pp
        actual = self.core._vertical_tensile_breakdown_pressure(shmin, shmax, t0, pp, 1.0)
        self.assertAlmostEqual(float(actual), expected)

    def test_tensile_breakdown_raw_pressure_is_not_clipped(self):
        actual = self.core._vertical_tensile_breakdown_pressure(30.0, 100.0, 1.0, 50.0, 1.0)
        self.assertLess(float(actual), 0.0)

    def test_1d_output_separates_breakdown_and_conservative_upper_screen(self):
        logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0, 1020.0],
            "DT": [90.0, 90.0, 90.0],
            "RHOB": [2.20, 2.21, 2.22],
            "DTS": [180.0, 181.0, 182.0],
        })
        out = self.core.compute_1d_mem(
            df=logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference="TVD", biot_alpha=0.6, dt_surface=80.0, dt_matrix=55.5,
            compaction_coefficient=0.0003, normal_trend_calibrated=True,
        )
        self.assertIn("Tensile_Breakdown_Pressure_MPa", out)
        self.assertTrue(np.isfinite(out["Tensile_Breakdown_EMW_SG"]).all())
        expected_upper = np.minimum(out["Tensile_Breakdown_EMW_SG"], out["Shmin_EMW_SG"])
        np.testing.assert_allclose(out["Fracture_EMW_SG"], expected_upper)
        expected_exists = np.maximum(out["Pore_Pressure_EMW_SG"], out["Collapse_EMW_SG"]) <= expected_upper
        np.testing.assert_array_equal(out["Mud_Window_Exists"], expected_exists)

    def test_full_interval_window_uses_worst_depths_not_means(self):
        frame = pd.DataFrame({
            "Pore_Pressure_EMW_SG": [1.1, 1.2],
            "Collapse_EMW_SG": [1.2, 2.0],
            "Fracture_EMW_SG": [1.8, 1.9],
        })
        result = _full_interval_screening_window(frame)
        self.assertTrue(result["valid"])
        self.assertFalse(result["exists"])
        self.assertAlmostEqual(result["lower_sg"], 2.0)
        self.assertAlmostEqual(result["upper_sg"], 1.8)

    def test_full_interval_window_fails_closed_on_missing_sample(self):
        frame = pd.DataFrame({
            "Pore_Pressure_EMW_SG": [1.1, np.nan],
            "Collapse_EMW_SG": [1.2, 1.4],
            "Fracture_EMW_SG": [1.8, 1.9],
        })
        result = _full_interval_screening_window(frame)
        self.assertFalse(result["valid"])
        self.assertFalse(result["exists"])


if __name__ == "__main__":
    unittest.main()
