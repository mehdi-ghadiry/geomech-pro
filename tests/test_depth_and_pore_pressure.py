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

    def test_unbounded_eaton_estimate_is_reported_but_not_used_downstream(self):
        result = self.compute()
        hydrostatic = result["Depth"] * 9.80665e-3
        self.assertTrue((result["Pore_Pressure_Pp_Eaton_Raw_MPa"] < hydrostatic).all())
        self.assertTrue((result["Pore_Pressure_Lower_Bound_Hit"] == 1).all())
        self.assertTrue(result["Pore_Pressure_Pp_MPa"].isna().all())
        self.assertTrue((result["Pore_Pressure_Estimate_Valid"] == 1).all())
        self.assertTrue((result["Pore_Pressure_Estimate_Usable"] == 0).all())
        self.assertTrue((result["Pore_Pressure_QC_Flag"] == 1).all())
        self.assertTrue((result["Pore_Pressure_Hydrostatic_Reference_MPa"] == hydrostatic).all())
        dependent = [
            "Sig_V_Eff_MPa", "Shmin_MPa", "SHmax_MPa", "Pore_Pressure_EMW_SG",
            "Collapse_EMW_SG", "Shmin_EMW_SG", "Fracture_EMW_SG",
        ]
        for column in dependent:
            with self.subTest(column=column):
                self.assertTrue(result[column].isna().all())

    def test_in_range_estimate_remains_available_for_downstream_calculations(self):
        self.logs["DT"] = [160.0, 160.0, 160.0]
        result = self.compute()
        self.assertTrue((result["Pore_Pressure_Estimate_Usable"] == 1).all())
        self.assertTrue(result["Pore_Pressure_Pp_MPa"].notna().all())
        self.assertTrue(result["Shmin_MPa"].notna().all())
        self.assertTrue(result["Collapse_EMW_SG"].notna().all())

    def test_deviated_analysis_rejects_flagged_pore_pressure(self):
        result = self.compute()
        with self.assertRaisesRegex(ValueError, "not usable"):
            self.core.compute_deviated_wellbore_stability(
                mem_df=result,
                depth_m=float(result["Depth"].iloc[0]),
                well_inclination_deg=60.0,
                well_azimuth_deg=90.0,
                shmax_azimuth_deg=0.0,
            )

    def test_non_monotonic_tvd_is_rejected(self):
        self.logs.loc[1, "TVD"] = 990.0
        with self.assertRaisesRegex(ValueError, "ordered from shallow to deep"):
            self.compute()


if __name__ == "__main__":
    unittest.main()
