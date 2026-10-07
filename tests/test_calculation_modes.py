"""Synthetic fixtures test software behavior, not engineering validity."""
import io
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from backend.geomechanics_core import GeomechanicsCore
from frontend.log_preview import (find_shear_curve, raw_log_figure, is_educational,
                                  label_educational_chart)


class CalculationModeTests(unittest.TestCase):
    def setUp(self):
        self.core = GeomechanicsCore()
        self.logs = pd.DataFrame({"TVD": [1000., 1010., 1020.],
                                  "DT": [90., 90., 90.], "RHOB": [2.2, 2.21, 2.22],
                                  "DTS": [180., 181., 182.]})

    def compute(self, **overrides):
        params = dict(df=self.logs, depth_col="TVD", dt_col="DT", rhob_col="RHOB",
                      dts_col="DTS", depth_reference="TVD", dt_surface=80.,
                      dt_matrix=55.5, calculation_mode="educational")
        params.update(overrides)
        return self.core.compute_1d_mem(**params)

    def test_engineering_still_blocks_uncalibrated_trend(self):
        with self.assertRaisesRegex(ValueError, "disabled until"):
            self.compute(calculation_mode="engineering")

    def test_unknown_mode_cannot_bypass_gate(self):
        with self.assertRaisesRegex(ValueError, "calculation_mode"):
            self.compute(calculation_mode="demo")

    def test_educational_metadata_survives_json_and_csv(self):
        result = self.compute()
        for restored in (pd.DataFrame(result.to_dict(orient="records")),
                         pd.read_csv(io.StringIO(result.to_csv(index=False)))):
            self.assertTrue(is_educational(restored))
            self.assertFalse(restored["Normal_Trend_Calibrated"].any())
            self.assertTrue(restored["Result_Use_Warning"].str.contains("NOT FOR ENGINEERING").all())

    def test_modes_have_identical_numerical_results(self):
        educational = self.compute()
        engineering = self.compute(calculation_mode="engineering", normal_trend_calibrated=True)
        columns = educational.select_dtypes(include=[np.number]).columns
        pd.testing.assert_frame_equal(educational[columns], engineering[columns])

    def test_educational_preserves_depth_units_and_lithology_gates(self):
        for overrides, message in (({"depth_reference": "MD"}, "TVD"),
                                   ({"sonic_unit": "unknown"}, "sonic unit"),
                                   ({"dts_col": None}, "lithology")):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.compute(**overrides)

    def test_all_invalid_pressure_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "No physically admissible"):
            self.compute(dt_surface=180., dt_matrix=100.)

    def test_invalid_sample_remains_masked(self):
        self.logs["DT"] = [90., 165., 165.]
        result = self.compute(dt_surface=180., dt_matrix=100.)
        self.assertTrue(pd.isna(result["Pore_Pressure_Pp_MPa"].iloc[0]))
        self.assertTrue(pd.isna(result["Shmin_MPa"].iloc[0]))

    def test_shear_mapping_never_falls_back_to_other_curves(self):
        self.assertIsNone(find_shear_curve(["DEPTH", "DT", "RHOB"]))
        self.assertEqual(find_shear_curve(["DEPTH", "DT", "DTSM"]), "DTSM")

    def test_input_preview_has_no_calibration_or_depth_reference_requirement(self):
        records = [{"MD": 10, "GR": 15}, {"MD": 11, "GR": None}, {"MD": 12, "GR": 20}]
        fig = raw_log_figure(records, "MD", ["GR"])
        self.assertTrue(pd.isna(fig.data[0].x[1]))
        self.assertFalse(fig.data[0].connectgaps)
        self.assertIn("unverified", fig.layout.title.text)

    def test_chart_warning_embedded_in_html_export(self):
        fig = label_educational_chart(go.Figure())
        self.assertIn("NOT FOR ENGINEERING DECISIONS", fig.to_html())


if __name__ == "__main__":
    unittest.main()
