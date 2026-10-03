import unittest

import numpy as np
import pandas as pd

from backend.geomechanics_core import GeomechanicsCore


class StressRegimeQCTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({
            "DEPT": [1000.0, 1010.0, 1020.0],
            "DT": [80.0, 80.0, 80.0],
            "DTS": [120.0, 120.0, 120.0],
            "RHOB": [2.3, 2.3, 2.3],
        })

    def compute(self, *, ex=0.0, ey=0.0, regime="normal_faulting"):
        return self.core.compute_1d_mem(
            self.logs,
            "DEPT",
            "DT",
            "RHOB",
            dts_col="DTS",
            depth_reference="TVD",
            dt_matrix=80.0,
            dt_surface=80.0,
            compaction_coefficient=0.0,
            normal_trend_calibrated=True,
            tectonic_ex=ex,
            tectonic_ey=ey,
            stress_regime=regime,
        )

    def test_normal_faulting_does_not_clip_shmin_to_098_sv(self):
        # Equal horizontal strains give equal horizontal stresses. This case is
        # inside the normal-faulting ordering while just above the former cap.
        result = self.compute(ex=0.000245, ey=0.000245)
        row = result.iloc[0]
        self.assertGreater(row["Shmin_Raw_MPa"], 0.98 * row["Overburden_Stress_Sv_MPa"])
        self.assertLessEqual(row["SHmax_Raw_MPa"], row["Overburden_Stress_Sv_MPa"])
        self.assertEqual(row["Stress_Regime_Valid"], 1)
        self.assertAlmostEqual(row["Shmin_MPa"], row["Shmin_Raw_MPa"])
        self.assertAlmostEqual(row["SHmax_MPa"], row["SHmax_Raw_MPa"])

    def test_strike_slip_and_reverse_regimes_are_supported_without_caps(self):
        strike = self.compute(ex=0.0005, ey=0.0002, regime="strike_slip").iloc[0]
        self.assertEqual(strike["Stress_Regime_Valid"], 1)
        self.assertLessEqual(strike["Shmin_MPa"], strike["Overburden_Stress_Sv_MPa"])
        self.assertGreaterEqual(strike["SHmax_MPa"], strike["Overburden_Stress_Sv_MPa"])

        reverse = self.compute(ex=0.001, ey=0.001, regime="reverse_faulting").iloc[0]
        self.assertEqual(reverse["Stress_Regime_Valid"], 1)
        self.assertGreaterEqual(reverse["Shmin_MPa"], reverse["Overburden_Stress_Sv_MPa"])
        self.assertGreaterEqual(reverse["SHmax_MPa"], reverse["Shmin_MPa"])

    def test_regime_inconsistent_stresses_are_preserved_raw_and_withheld(self):
        result = self.compute(ex=0.0004, ey=0.0002, regime="normal_faulting")
        row = result.iloc[0]
        self.assertEqual(row["Stress_Regime_QC_Flag"], 1)
        self.assertTrue(np.isfinite(row["Shmin_Raw_MPa"]))
        self.assertTrue(np.isfinite(row["SHmax_Raw_MPa"]))
        self.assertTrue(pd.isna(row["Shmin_MPa"]))
        self.assertTrue(pd.isna(row["SHmax_MPa"]))
        self.assertIn("do not satisfy", row["Stress_Regime_QC_Reason"])

    def test_lot_solver_recomputes_to_target_inside_strike_slip_regime(self):
        initial = self.compute(ex=0.0005, ey=0.0, regime="strike_slip")
        calibration = self.core.solve_tectonic_ey_for_lot(
            initial,
            lot_depth=1010.0,
            lot_pressure_mpa=18.0,
            tectonic_ex=0.0005,
            initial_tectonic_ey=0.0,
            stress_regime="strike_slip",
        )
        self.assertTrue(calibration["calibration_applied"])
        recomputed = self.core.compute_1d_mem(
            self.logs,
            "DEPT",
            "DT",
            "RHOB",
            dts_col="DTS",
            depth_reference="TVD",
            dt_matrix=80.0,
            dt_surface=80.0,
            compaction_coefficient=0.0,
            normal_trend_calibrated=True,
            tectonic_ex=0.0005,
            tectonic_ey=calibration["tectonic_ey"],
            stress_regime="strike_slip",
        )
        matched = recomputed.loc[recomputed["Depth"] == calibration["matched_depth"]].iloc[0]
        self.assertEqual(matched["Stress_Regime_Valid"], 1)
        self.assertAlmostEqual(matched["Shmin_MPa"], 18.0, places=6)

    def test_unknown_stress_regime_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Stress regime must be"):
            self.compute(regime="unknown")


if __name__ == "__main__":
    unittest.main()
