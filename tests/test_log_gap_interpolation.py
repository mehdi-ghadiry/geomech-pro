import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class ShortInternalGapInterpolationTests(unittest.TestCase):
    def test_only_short_bounded_gaps_are_interpolated(self):
        values = np.array([np.nan, 10.0, np.nan, np.nan, 40.0,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           100.0, np.nan])

        result = GeomechanicsCore._interpolate_short_internal_gaps(values, max_gap=5)

        self.assertTrue(np.isnan(result[0]))  # leading edge remains missing
        np.testing.assert_allclose(result[1:5], [10.0, 20.0, 30.0, 40.0])
        self.assertTrue(np.isnan(result[5:11]).all())  # six-sample outage stays missing
        self.assertEqual(result[11], 100.0)
        self.assertTrue(np.isnan(result[12]))  # trailing edge remains missing

    def test_gap_exactly_at_limit_is_filled(self):
        values = np.array([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0])
        result = GeomechanicsCore._interpolate_short_internal_gaps(values, max_gap=5)
        np.testing.assert_allclose(result, np.arange(7, dtype=float))

    def test_gap_over_limit_is_not_partially_filled(self):
        values = np.array([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0])
        result = GeomechanicsCore._interpolate_short_internal_gaps(values, max_gap=5)
        self.assertTrue(np.isnan(result[1:7]).all())

    def test_long_density_gap_remains_invalid_in_pressure_outputs(self):
        depth = np.arange(1000.0, 1140.0, 10.0)
        rhob = np.full(depth.size, 2.20)
        rhob[3:9] = np.nan
        logs = pd.DataFrame({
            "TVD": depth,
            "DT": np.full(depth.size, 90.0),
            "RHOB": rhob,
            "DTS": np.full(depth.size, 180.0),
        })

        result = GeomechanicsCore().compute_1d_mem(
            df=logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference="TVD", dt_surface=80.0, dt_matrix=55.5,
            compaction_coefficient=0.0003, normal_trend_calibrated=True,
        )

        self.assertTrue(result["Overburden_Stress_Sv_MPa"].iloc[5:8].isna().all())
        self.assertTrue(result["Pore_Pressure_Pp_MPa"].iloc[5:8].isna().all())
        self.assertTrue((result["Pore_Pressure_Estimate_Valid"].iloc[5:8] == 0).all())


if __name__ == "__main__":
    unittest.main()
