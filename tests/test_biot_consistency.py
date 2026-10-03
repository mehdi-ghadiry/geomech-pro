import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class BiotConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()

    def test_kirsch_effective_stresses_use_biot_coefficient(self):
        sigma_well = np.array([[40.0, 10.0, 0.0], [10.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
        theta = np.array([0.0])

        result = self.core._kirsch_wall_stresses(
            sigma_well, pp_mpa=10.0, pw_mpa=5.0, nu=0.25, biot_alpha=0.6, theta_rad=theta
        )

        self.assertAlmostEqual(result["sigma_rr_eff"][0], -1.0)
        self.assertAlmostEqual(result["sigma_tt_eff"][0], 9.0)
        self.assertAlmostEqual(result["sigma_zz_eff"][0], 14.0)

    def test_deviated_stability_uses_alpha_saved_with_mem_results(self):
        row = {
            "Depth": 2000.0,
            "Biot_Coefficient": 0.65,
            "Overburden_Stress_Sv_MPa": 50.0,
            "Shmin_MPa": 30.0,
            "SHmax_MPa": 40.0,
            "Pore_Pressure_Pp_MPa": 20.0,
            "Pore_Pressure_Estimate_Usable": 1,
            "Poisson_Ratio": 0.25,
            "UCS_MPa": 25.0,
        }
        result = self.core.compute_deviated_wellbore_stability(
            pd.DataFrame([row]), depth_m=2000.0, well_inclination_deg=45.0,
            well_azimuth_deg=90.0, shmax_azimuth_deg=45.0,
            mud_weight_sg_min=1.2, mud_weight_sg_max=1.2, mud_weight_sg_step=0.01,
        )
        self.assertEqual(result["biot_alpha_used"], 0.65)

    def test_legacy_results_require_explicit_biot_coefficient(self):
        legacy = pd.DataFrame([{ "Depth": 2000.0 }])
        with self.assertRaisesRegex(ValueError, "Biot coefficient is missing"):
            self.core.compute_deviated_wellbore_stability(
                legacy, depth_m=2000.0, well_inclination_deg=0.0,
                well_azimuth_deg=0.0, shmax_azimuth_deg=0.0,
            )

    def test_invalid_biot_coefficient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.core.compute_1d_mem(
                df=pd.DataFrame({"TVD": [1000.0], "DT": [90.0], "RHOB": [2.2]}),
                depth_col="TVD", dt_col="DT", rhob_col="RHOB",
                depth_reference="TVD", biot_alpha=1.1,
            )


if __name__ == "__main__":
    unittest.main()
