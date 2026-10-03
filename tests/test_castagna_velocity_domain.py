import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class CastagnaVelocityDomainTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()

    def test_nonpositive_castagna_vs_is_missing_not_floored(self):
        estimates = self.core._castagna_clastic_dts_us_ft(np.array([250.0, 221.3]))
        self.assertTrue(np.isnan(estimates[0]))

        vp = 1e6 / 221.3
        vs = 0.8621 * vp - 3846.4
        expected_dts = 1e6 / vs
        self.assertGreater(vs, 0.0)
        self.assertLess(vs, 100.0)
        self.assertAlmostEqual(estimates[1], expected_dts)
        self.assertGreater(estimates[1], 10000.0)

    def test_nonphysical_estimate_is_flagged_in_mem_outputs(self):
        logs = pd.DataFrame({
            "DEPT": [1000.0, 1010.0],
            "DT": [250.0, 100.0],
            "RHOB": [2.3, 2.3],
        })
        result = self.core.compute_1d_mem(
            logs,
            "DEPT",
            "DT",
            "RHOB",
            lithology_group="water_saturated_clastic",
            depth_reference="TVD",
            dt_matrix=180.0,
            dt_surface=180.0,
            compaction_coefficient=0.0,
            normal_trend_calibrated=True,
        )
        self.assertEqual(result.loc[0, "Vs_Estimation_Valid"], 0)
        self.assertEqual(result.loc[0, "Vs_Estimation_QC_Flag"], 1)
        self.assertEqual(result.loc[0, "Elastic_Properties_Valid"], 0)
        self.assertEqual(result.loc[1, "Vs_Estimation_Valid"], 1)

    def test_all_nonpositive_clastic_estimates_fail_with_clear_message(self):
        logs = pd.DataFrame({
            "DEPT": [1000.0, 1010.0],
            "DT": [250.0, 245.0],
            "RHOB": [2.3, 2.3],
        })
        with self.assertRaisesRegex(ValueError, "non-positive across the interval"):
            self.core.compute_1d_mem(
                logs,
                "DEPT",
                "DT",
                "RHOB",
                lithology_group="water_saturated_clastic",
                depth_reference="TVD",
                dt_matrix=180.0,
                dt_surface=180.0,
                compaction_coefficient=0.0,
                normal_trend_calibrated=True,
            )


if __name__ == "__main__":
    unittest.main()
