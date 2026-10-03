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


if __name__ == "__main__":
    unittest.main()
