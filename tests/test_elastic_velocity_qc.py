import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class ElasticVelocityQCTests(unittest.TestCase):
    def compute(self, dts):
        logs = pd.DataFrame({
            "TVD": [1000.0],
            "DT": [90.0],
            "RHOB": [2.20],
            "DTS": [dts],
        })
        return GeomechanicsCore().compute_1d_mem(
            df=logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
            depth_reference="TVD", dt_surface=80.0, dt_matrix=55.5,
            compaction_coefficient=0.0003, normal_trend_calibrated=True,
        ).iloc[0]

    def test_vp_exceeds_vs_but_negative_bulk_modulus_is_masked(self):
        # DTS > DT means Vp > Vs, but this pair still violates K > 0 because
        # Vp/Vs = 100/90 < sqrt(4/3).
        row = self.compute(100.0)
        self.assertEqual(row["Elastic_Properties_Valid"], 0)
        self.assertEqual(row["Elastic_Properties_QC_Flag"], 1)
        for column in (
            "Youngs_Modulus_GPa", "Shear_Modulus_GPa", "Bulk_Modulus_GPa",
            "Poisson_Ratio", "UCS_MPa", "Tensile_Strength_MPa",
            "Shmin_MPa", "SHmax_MPa", "Collapse_EMW_SG",
        ):
            self.assertTrue(np.isnan(row[column]), column)

    def test_vs_faster_than_vp_is_rejected(self):
        row = self.compute(80.0)
        self.assertEqual(row["Elastic_Properties_Valid"], 0)
        self.assertTrue(np.isnan(row["Bulk_Modulus_GPa"]))

    def test_stable_velocity_pair_keeps_finite_moduli_and_strength(self):
        row = self.compute(180.0)
        self.assertEqual(row["Elastic_Properties_Valid"], 1)
        self.assertEqual(row["Elastic_Properties_QC_Flag"], 0)
        for column in (
            "Youngs_Modulus_GPa", "Shear_Modulus_GPa", "Bulk_Modulus_GPa",
            "Poisson_Ratio", "UCS_MPa", "Tensile_Strength_MPa",
        ):
            self.assertTrue(np.isfinite(row[column]), column)


if __name__ == "__main__":
    unittest.main()
