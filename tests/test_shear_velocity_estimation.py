import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class ShearVelocityEstimationTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0, 1020.0],
            "DT": [90.0, 90.0, 90.0],
            "RHOB": [2.20, 2.21, 2.22],
        })
        self.trend = {
            "dt_surface": 80.0,
            "dt_matrix": 55.5,
            "compaction_coefficient": 0.0003,
            "normal_trend_calibrated": True,
        }

    def test_castagna_ft_s_intercept_is_converted_from_km_s(self):
        # At Vp=10,000 ft/s, Vs=0.8621*Vp-3846.4=4774.6 ft/s.
        dts = self.core._castagna_clastic_dts_us_ft(np.array([100.0]))
        self.assertAlmostEqual(float(dts[0]), 1e6 / 4774.6, places=6)

    def test_absent_dts_is_not_assumed_to_be_clastic(self):
        with self.assertRaisesRegex(ValueError, "explicitly identifies the interval"):
            self.core.compute_1d_mem(
                self.logs, "TVD", "DT", "RHOB", depth_reference="TVD", **self.trend
            )

    def test_castagna_estimator_rejects_carbonate_without_measured_dts(self):
        with self.assertRaisesRegex(ValueError, "not a carbonate relation"):
            self.core.compute_1d_mem(
                self.logs, "TVD", "DT", "RHOB", depth_reference="TVD",
                lithology_group="carbonate", **self.trend
            )

    def test_clastic_estimate_requires_explicit_lithology_and_is_recorded(self):
        result = self.core.compute_1d_mem(
            self.logs, "TVD", "DT", "RHOB", depth_reference="TVD",
            lithology_group="water_saturated_clastic", **self.trend
        )
        self.assertEqual(
            result["Vs_Estimation_Method"].iloc[0],
            "Castagna mudrock line (water-saturated clastic only)",
        )
        self.assertTrue(np.isfinite(result["Shear_Modulus_GPa"]).all())

    def test_measured_dts_is_accepted_for_carbonate(self):
        logs = self.logs.assign(DTS=[180.0, 181.0, 182.0])
        result = self.core.compute_1d_mem(
            logs, "TVD", "DT", "RHOB", dts_col="DTS", lithology_group="carbonate",
            depth_reference="TVD", **self.trend
        )
        self.assertEqual(result["Vs_Estimation_Method"].iloc[0], "Measured DTS")

    def test_limestone_estimator_uses_user_formula_and_unit_conversion(self):
        dt = np.array([100.0])  # us/ft
        vp_kms = 304.8 / dt
        vs_kms = -0.05508 * vp_kms**2 + 1.01677 * vp_kms - 1.03049
        dts, method = self.core._carbonate_empirical_dts_us_ft(dt, "limestone")
        self.assertAlmostEqual(float(dts[0]), float(304.8 / vs_kms[0]), places=9)
        self.assertIn("limestone", method)

    def test_dolomite_estimator_uses_user_formula_and_unit_conversion(self):
        dt = np.array([100.0])  # us/ft
        vp_kms = 304.8 / dt
        vs_kms = 0.58321 * vp_kms - 0.07775
        dts, method = self.core._carbonate_empirical_dts_us_ft(dt, "dolomite")
        self.assertAlmostEqual(float(dts[0]), float(304.8 / vs_kms[0]), places=9)
        self.assertIn("dolomite", method)

    def test_mixed_carbonate_cannot_select_single_lithology_fit(self):
        with self.assertRaisesRegex(ValueError, "explicit limestone or dolomite"):
            self.core.compute_1d_mem(
                self.logs, "TVD", "DT", "RHOB", depth_reference="TVD",
                lithology_group="carbonate", **self.trend
            )

    def test_carbonate_fit_masks_nonphysical_rows_without_clamping(self):
        dts, _ = self.core._carbonate_empirical_dts_us_ft(np.array([500.0, 100.0]), "limestone")
        self.assertTrue(np.isnan(dts[0]))
        self.assertTrue(np.isfinite(dts[1]))


if __name__ == "__main__":
    unittest.main()
