import unittest

import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class DeviatedMudWindowTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()

    def compute(self, row, minimum=0.9, maximum=1.0, step=0.05):
        return self.core.compute_deviated_wellbore_stability(
            mem_df=pd.DataFrame([row]),
            depth_m=2000.0,
            well_inclination_deg=0.0,
            well_azimuth_deg=0.0,
            shmax_azimuth_deg=0.0,
            mud_weight_sg_min=minimum,
            mud_weight_sg_max=maximum,
            mud_weight_sg_step=step,
        )

    def test_no_safe_value_is_not_reported_as_maximum(self):
        row = {
            "Depth": 2000.0,
            "Biot_Coefficient": 1.0,
            "Overburden_Stress_Sv_MPa": 100.0,
            "Shmin_MPa": 80.0,
            "SHmax_MPa": 90.0,
            "Pore_Pressure_Pp_MPa": 40.0,
            "Pore_Pressure_Estimate_Usable": 1,
            "Poisson_Ratio": 0.25,
            "UCS_MPa": 1.0,
        }
        result = self.compute(row)
        self.assertFalse(result["solution_found"])
        self.assertIsNone(result["collapse_emw_sg"])
        self.assertEqual(result["status"], "no_safe_weight_in_tested_range")
        self.assertAlmostEqual(result["max_tested_mud_weight_sg"], 1.0)

    def test_safe_solution_is_reported(self):
        row = {
            "Depth": 2000.0,
            "Biot_Coefficient": 1.0,
            "Overburden_Stress_Sv_MPa": 50.0,
            "Shmin_MPa": 30.0,
            "SHmax_MPa": 35.0,
            "Pore_Pressure_Pp_MPa": 15.0,
            "Pore_Pressure_Estimate_Usable": 1,
            "Poisson_Ratio": 0.25,
            "UCS_MPa": 350.0,
        }
        result = self.compute(row)
        self.assertTrue(result["solution_found"])
        self.assertIsNotNone(result["collapse_emw_sg"])
        self.assertEqual(result["status"], "safe_weight_found")

    def test_invalid_search_range_is_rejected(self):
        row = {
            "Depth": 2000.0,
            "Biot_Coefficient": 1.0,
            "Overburden_Stress_Sv_MPa": 50.0,
            "Shmin_MPa": 30.0,
            "SHmax_MPa": 35.0,
            "Pore_Pressure_Pp_MPa": 15.0,
            "Pore_Pressure_Estimate_Usable": 1,
            "Poisson_Ratio": 0.25,
            "UCS_MPa": 350.0,
        }
        with self.assertRaisesRegex(ValueError, "Mud-weight range must be finite"):
            self.compute(row, minimum=1.0, maximum=0.9, step=0.05)


if __name__ == "__main__":
    unittest.main()
