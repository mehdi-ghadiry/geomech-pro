"""Synthetic data test usability and provenance, NOT engineering accuracy."""
import unittest
import numpy as np
import pandas as pd
from backend.geomechanics_core import GeomechanicsCore


class ExperimentalOptionsTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({"MD": [1000., 1010., 1020.], "DT": [90., 90., 90.],
                                  "RHOB": [2.2, 2.21, 2.22]})
        self.params = dict(df=self.logs, depth_col="MD", dt_col="DT", rhob_col="RHOB",
                           depth_reference="MD", calculation_mode="educational",
                           dt_surface=180., dt_matrix=100.,
                           apply_log_range_filter=False, apply_pressure_screen=False,
                           apply_stress_screen=False, apply_property_bounds=False,
                           experimental_vertical_depth=True, experimental_vs_ratio=0.5,
                           experimental_hydrostatic_fallback=True)

    def compute(self, **extra):
        return self.core.compute_1d_mem(**{**self.params, **extra})

    def test_quick_test_needs_no_calibration_dts_or_tvd_declaration(self):
        result = self.compute()
        self.assertTrue(np.isfinite(result["Shmin_MPa"]).all())
        self.assertTrue(result["Experimental_Hydrostatic_Fallback_Used"].eq(1).all())
        self.assertTrue(result["Pore_Pressure_Estimate_Valid"].eq(0).all())
        np.testing.assert_allclose(result["Pore_Pressure_Pp_MPa"], result["Pore_Pressure_Hydrostatic_Reference_MPa"])
        self.assertTrue(result["Vs_Estimation_Method"].str.contains("assumed").all())
        self.assertTrue(result["Depth_Reference_Used"].str.contains("NOT TRAJECTORY CONVERTED").all())
        self.assertFalse(result["Normal_Trend_Calibrated"].any())
        self.assertTrue(result["Result_Use_Warning"].str.contains("NOT FOR ENGINEERING").all())

    def test_engineering_cannot_use_relaxed_options_even_if_calibration_declared(self):
        with self.assertRaisesRegex(ValueError, "require educational"):
            self.compute(calculation_mode="engineering", normal_trend_calibrated=True)

    def test_pressure_fallback_is_explicit_and_can_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "No physically admissible"):
            self.compute(experimental_hydrostatic_fallback=False)

    def test_unknown_lithology_is_not_silently_assigned_an_empirical_fit(self):
        with self.assertRaisesRegex(ValueError, "lithology"):
            self.compute(experimental_vs_ratio=None)

    def test_range_filter_is_optional_not_positive_value_validation(self):
        self.logs["RHOB"] = [0.9, 0.9, 0.9]
        self.assertTrue(np.isfinite(self.compute()["Overburden_Stress_Sv_MPa"]).all())
        with self.assertRaisesRegex(ValueError, "No valid RHOB"):
            self.compute(apply_log_range_filter=True)
        self.logs["RHOB"] = [-1., -1., -1.]
        with self.assertRaisesRegex(ValueError, "No valid RHOB"):
            self.compute()

    def test_stress_screen_can_be_reenabled_without_changing_raw_stresses(self):
        relaxed = self.compute(tectonic_ex=0.002, stress_regime="normal_faulting")
        strict = self.compute(tectonic_ex=0.002, stress_regime="normal_faulting", apply_stress_screen=True)
        self.assertTrue(strict["Shmin_MPa"].isna().all())
        self.assertTrue(relaxed["Shmin_MPa"].notna().all())
        np.testing.assert_allclose(strict["Shmin_Raw_MPa"], relaxed["Shmin_Raw_MPa"])
        self.assertTrue(relaxed["Stress_Regime_QC_Flag"].eq(1).all())

    def test_physical_elastic_validation_cannot_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "ratio"):
            self.compute(experimental_vs_ratio=0.95)

    def test_gap_filling_is_optional(self):
        self.logs["DT"] = [90., np.nan, 90.]
        self.assertTrue(self.compute(interpolation_max_gap=5)["Youngs_Modulus_GPa"].notna().all())
        self.assertTrue(pd.isna(self.compute(interpolation_max_gap=0)["Youngs_Modulus_GPa"].iloc[1]))

    def test_deviated_view_can_use_flagged_experimental_fallback(self):
        result = self.compute()
        derived = self.core.compute_deviated_wellbore_stability(result, depth_m=1010., well_inclination_deg=0.,
                                                               well_azimuth_deg=0., shmax_azimuth_deg=0.)
        self.assertIsInstance(derived, dict)


if __name__ == "__main__":
    unittest.main()
