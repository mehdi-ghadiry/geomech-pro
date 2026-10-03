import unittest

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

    def compute(self, depth_unit="m", depth_reference="TVD"):
        return self.core.compute_1d_mem(
            df=self.logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference=depth_reference, depth_unit=depth_unit,
        )

    def test_md_is_rejected_without_trajectory_conversion(self):
        with self.assertRaisesRegex(ValueError, "requires a TVD depth curve"):
            self.compute(depth_reference="MD")

    def test_feet_are_normalized_to_metres(self):
        result = self.compute(depth_unit="ft")
        self.assertAlmostEqual(result["Depth"].iloc[0], 304.8, places=6)
        self.assertAlmostEqual(result["Depth"].iloc[-1], 310.896, places=6)

    def test_unbounded_eaton_estimate_is_reported_and_flagged(self):
        result = self.compute()
        hydrostatic = result["Depth"] * 9.80665e-3
        self.assertTrue((result["Pore_Pressure_Pp_MPa"] < hydrostatic).all())
        self.assertTrue((result["Pore_Pressure_Lower_Bound_Hit"] == 1).all())
        self.assertTrue((result["Pore_Pressure_Pp_MPa"] == result["Pore_Pressure_Pp_Eaton_Raw_MPa"]).all())
        self.assertTrue((result["Pore_Pressure_Hydrostatic_Reference_MPa"] == hydrostatic).all())

    def test_non_monotonic_tvd_is_rejected(self):
        self.logs.loc[1, "TVD"] = 990.0
        with self.assertRaisesRegex(ValueError, "ordered from shallow to deep"):
            self.compute()


if __name__ == "__main__":
    unittest.main()
