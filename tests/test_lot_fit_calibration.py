import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class LotFitCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.results = pd.DataFrame({
            "Depth": [1000.0, 1010.0, 1020.0],
            "Nu_Eff": [0.25, 0.25, 0.25],
            "Youngs_Modulus_GPa": [20.0, 20.0, 20.0],
            "Sig_V_Eff_MPa": [30.0, 30.0, 30.0],
            "Pore_Pressure_Pp_MPa": [10.0, 10.0, 10.0],
            "Overburden_Stress_Sv_MPa": [50.0, 50.0, 50.0],
        })

    def test_solution_includes_existing_ex_and_initial_ey_strains(self):
        alpha, ex, initial_ey = 0.8, 0.001, 0.0005
        target_shmin_mpa = 25.0
        result = self.core.solve_tectonic_ey_for_lot(
            self.results, 1010.0, target_shmin_mpa, alpha,
            tectonic_ex=ex, initial_tectonic_ey=initial_ey,
        )
        nu, e_pa, sv_eff_pa, pp_pa = 0.25, 20e9, 30e6, 10e6
        base_without_ey = nu / (1.0 - nu) * sv_eff_pa + e_pa / (1.0 - nu**2) * nu * ex
        expected_ey = ((target_shmin_mpa * 1e6 - alpha * pp_pa - base_without_ey)
                       * (1.0 - nu**2) / e_pa)
        self.assertAlmostEqual(result["tectonic_ey"], expected_ey, places=12)
        self.assertAlmostEqual(result["uncalibrated_shmin_mpa"], 34.0, places=6)
        self.assertTrue(result["calibration_applied"])

    def test_normal_faulting_maximum_follows_regime_order_not_fraction_of_sv(self):
        result = self.core.solve_tectonic_ey_for_lot(self.results, 1010.0, 49.5)
        self.assertAlmostEqual(result["achievable_max_mpa"], 20.0)
        self.assertTrue(result["attainable_range_exists"])
        self.assertFalse(result["within_range"])
        self.assertFalse(result["calibration_applied"])

    def test_lot_depth_far_from_sample_is_rejected(self):
        irregular = self.results.copy()
        irregular["Depth"] = [1000.0, 1010.0, 1050.0]
        with self.assertRaisesRegex(ValueError, "beyond half the median sampling interval"):
            self.core.solve_tectonic_ey_for_lot(irregular, 1030.0, 25.0)

    def test_depth_outside_log_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the log interval"):
            self.core.solve_tectonic_ey_for_lot(self.results, 1021.0, 25.0)

    def test_nearest_sample_with_missing_properties_is_rejected(self):
        invalid = self.results.copy()
        invalid.loc[1, "Youngs_Modulus_GPa"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            self.core.solve_tectonic_ey_for_lot(invalid, 1010.0, 25.0)

    def test_lot_attainability_uses_selected_stress_regime(self):
        strike = self.core.solve_tectonic_ey_for_lot(
            self.results, 1010.0, 40.0, tectonic_ex=0.003,
            stress_regime="strike_slip",
        )
        self.assertTrue(strike["calibration_applied"])
        self.assertEqual(strike["stress_regime"], "strike_slip")

        reverse = self.core.solve_tectonic_ey_for_lot(
            self.results, 1010.0, 60.0, tectonic_ex=0.003,
            stress_regime="reverse_faulting",
        )
        self.assertTrue(reverse["calibration_applied"])
        self.assertEqual(reverse["stress_regime"], "reverse_faulting")

    def test_out_of_range_target_is_not_reported_as_applied(self):
        result = self.core.solve_tectonic_ey_for_lot(self.results, 1010.0, 10.0)
        self.assertFalse(result["within_range"])
        self.assertFalse(result["calibration_applied"])


if __name__ == "__main__":
    unittest.main()
