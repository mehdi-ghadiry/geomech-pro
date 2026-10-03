import unittest

import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class LogUnitValidationTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({
            "TVD": [1000.0, 1010.0, 1020.0],
            "DT": [90.0, 91.0, 92.0],
            "RHOB": [2.20, 2.21, 2.22],
            "DTS": [180.0, 181.0, 182.0],
        })

    def compute(self, logs=None, sonic_unit="us/ft", density_unit="g/cm3"):
        return self.core.compute_1d_mem(
            df=self.logs if logs is None else logs,
            depth_col="TVD",
            dt_col="DT",
            rhob_col="RHOB",
            dts_col="DTS",
            depth_reference="TVD",
            normal_trend_calibrated=True,
            dt_surface=80.0,
            dt_matrix=55.5,
            compaction_coefficient=0.0003,
            sonic_unit=sonic_unit,
            density_unit=density_unit,
        )

    def test_supported_us_ft_and_g_cm3_units_are_accepted(self):
        result = self.compute(sonic_unit="us/ft", density_unit="g/cm3")
        self.assertEqual(len(result), len(self.logs))

    def test_supported_us_m_and_kg_m3_units_convert_equivalently(self):
        metric_logs = self.logs.copy()
        metric_logs["DT"] *= 3.28084
        metric_logs["DTS"] *= 3.28084
        metric_logs["RHOB"] *= 1000.0

        baseline = self.compute()
        converted = self.compute(
            logs=metric_logs,
            sonic_unit=" US/M ",
            density_unit=" KG/M3 ",
        )
        pd.testing.assert_frame_equal(
            converted,
            baseline,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_unknown_sonic_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sonic unit must be explicitly selected"):
            self.compute(sonic_unit="us/feet")

    def test_unknown_density_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Density unit must be explicitly selected"):
            self.compute(density_unit="kg/m^3")

    def test_missing_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sonic unit must be explicitly selected"):
            self.compute(sonic_unit=None)
        with self.assertRaisesRegex(ValueError, "Density unit must be explicitly selected"):
            self.compute(density_unit=" ")


if __name__ == "__main__":
    unittest.main()
