"""API and PDF boundary regression tests using synthetic data only."""
import os
from pathlib import Path
import sys
import unittest
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("GEOMECH_SECRET_KEY", "test-only-not-a-deployment-secret")
from fastapi.testclient import TestClient
from main import app


class CalculationModeAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.csv = b"TVD,DT,RHOB,DTS\n1000,90,2.2,180\n1010,90,2.21,181\n1020,90,2.22,182\n"
        self.params = dict(depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
                           depth_reference="TVD", depth_unit="m", dt_surface=80,
                           dt_matrix=55.5)

    def request(self, **extra):
        return self.client.post("/api/v1/mem/compute",
                                files={"las_file": ("synthetic.csv", self.csv)},
                                data={**self.params, **extra})

    def test_raw_endpoint_works_without_calibration_and_serializes_nulls(self):
        response = self.client.post("/api/v1/las/columns",
                                    files={"las_file": ("raw.csv", b"MD,GR\n10,15\n11,\n12,inf\n")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["records"][1]["GR"], None)
        self.assertEqual(response.json()["records"][2]["GR"], None)

    def test_api_engineering_default_remains_protected(self):
        response = self.request()
        self.assertEqual(response.status_code, 422)
        self.assertIn("disabled until", response.text)

    def test_api_educational_preserves_status_and_pdf_warning(self):
        response = self.request(calculation_mode="educational")
        self.assertEqual(response.status_code, 200, response.text)
        records = response.json()["results"]
        self.assertEqual(records[0]["Calculation_Mode"], "educational")
        self.assertFalse(records[0]["Normal_Trend_Calibrated"])
        report = self.client.post("/api/v1/report/pdf", json={"results": records, "well_name": "SYNTHETIC SOFTWARE TEST"})
        self.assertEqual(report.status_code, 200, report.text[:200] if report.status_code != 200 else "")
        self.assertTrue(report.content.startswith(b"%PDF"))
        # Check the actual PDF content stream without an extra PDF-reader dependency.
        import re
        import zlib
        streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", report.content, re.S)
        content = b""
        for stream in streams:
            try:
                content += zlib.decompress(stream)
            except zlib.error:
                content += stream
        self.assertIn(b"EDUCATIONAL ONLY - NOT FOR ENGINEERING DECISIONS", content)
        self.assertIn(b"UNCALIBRATED", content)

    def test_derived_stability_retains_educational_provenance(self):
        response = self.request(calculation_mode="educational", tectonic_ex=0, tectonic_ey=0)
        self.assertEqual(response.status_code, 200, response.text)
        derived = self.client.post("/api/v1/mem/deviated_stability", json={
            "results": response.json()["results"], "depth_m": 1010,
            "well_inclination_deg": 0, "well_azimuth_deg": 0, "shmax_azimuth_deg": 0,
        })
        self.assertEqual(derived.status_code, 200, derived.text)
        self.assertEqual(derived.json()["Calculation_Mode"], "educational")
        self.assertFalse(derived.json()["Normal_Trend_Calibrated"])
        self.assertIn("NOT FOR ENGINEERING", derived.json()["Result_Use_Warning"])

    def test_api_educational_still_rejects_md_and_bad_units(self):
        for extra in ({"depth_reference": "MD"}, {"sonic_unit": "unknown"}):
            response = self.request(calculation_mode="educational", **extra)
            self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
