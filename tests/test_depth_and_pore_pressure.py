import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class DepthAndPorePressureTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0, 1020.0],
            "DT": [90.0, 90.0, 90.0],
            "RHOB": [2.20, 2.21, 2.22],
            "DTS": [180.0, 181.0, 182.0],
        })

    def compute(self, depth_unit="m", depth_reference="TVD", **trend_params):
        params = {
            "dt_surface": 80.0,
            "dt_matrix": 55.5,
            "compaction_coefficient": 0.0003,
            "normal_trend_calibrated": True,
        }
        params.update(trend_params)
        return self.core.compute_1d_mem(
            df=self.logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference=depth_reference, depth_unit=depth_unit, **params,
        )

    def test_uncalibrated_normal_trend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disabled until the normal sonic compaction trend"):
            self.core.compute_1d_mem(
                df=self.logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
                depth_reference="TVD",
            )

    def test_md_is_rejected_without_trajectory_conversion(self):
        with self.assertRaisesRegex(ValueError, "requires a TVD depth curve"):
            self.compute(depth_reference="MD")

    def test_feet_are_normalized_to_metres(self):
        result = self.compute(depth_unit="ft")
        self.assertAlmostEqual(result["Depth"].iloc[0], 304.8, places=6)
        self.assertAlmostEqual(result["Depth"].iloc[-1], 310.896, places=6)

    def test_negative_eaton_estimates_are_rejected_when_all_samples_fail(self):
        with self.assertRaisesRegex(ValueError, "No physically admissible pore-pressure estimates"):
            self.compute(dt_surface=180.0, dt_matrix=100.0, compaction_coefficient=0.0003)

    def test_invalid_negative_pressure_is_masked_before_stress_and_mud_window(self):
        logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0],
            "DT": [90.0, 165.0],
            "RHOB": [2.20, 2.21],
            "DTS": [180.0, 181.0],
        })
        result = self.core.compute_1d_mem(
            df=logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference="TVD", dt_surface=180.0, dt_matrix=100.0,
            compaction_coefficient=0.0003, normal_trend_calibrated=True,
        )
        self.assertLess(result["Pore_Pressure_Pp_Eaton_Raw_MPa"].iloc[0], 0.0)
        self.assertTrue(pd.isna(result["Pore_Pressure_Pp_MPa"].iloc[0]))
        self.assertEqual(result["Pore_Pressure_Negative_Flag"].iloc[0], 1)
        for col in ("Shmin_MPa", "SHmax_MPa", "Pore_Pressure_EMW_SG", "Collapse_EMW_SG", "Fracture_EMW_SG"):
            self.assertTrue(pd.isna(result[col].iloc[0]), col)
        self.assertTrue(pd.notna(result["Pore_Pressure_Pp_MPa"].iloc[1]))

    def test_calibrated_trend_parameters_are_used(self):
        base = self.compute()
        changed = self.compute(dt_surface=100.0)
        self.assertFalse(np.allclose(
            base["Pore_Pressure_Pp_Eaton_Raw_MPa"],
            changed["Pore_Pressure_Pp_Eaton_Raw_MPa"],
        ))

    def test_non_monotonic_tvd_is_rejected(self):
        self.logs.loc[1, "TVD"] = 990.0
        with self.assertRaisesRegex(ValueError, "ordered from shallow to deep"):
            self.compute()


if __name__ == "__main__":
    unittest.main()
