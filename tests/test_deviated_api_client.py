"""Tab-3 transport regression tests; synthetic data, not engineering validation."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("GEOMECH_SECRET_KEY", "test-only-not-a-deployment-secret")
import api_client
from fastapi.testclient import TestClient
from main import app


class DeviatedAPIClientTests(unittest.TestCase):
    def payload(self):
        return dict(results=[dict(Depth=1000, Missing=float("nan"),
                                  PositiveInfinity=float("inf"), NegativeInfinity=-float("inf"))],
                    depth_m=1000.0, well_inclination_deg=30.0,
                    well_azimuth_deg=0.0, shmax_azimuth_deg=0.0,
                    friction_angle_deg=30.0, biot_alpha=1.0)

    def test_strict_json_preserves_rows_and_missing_values(self):
        payload = self.payload()
        with self.assertRaises(ValueError):
            json.dumps(payload, allow_nan=False)  # Reproduce original failure.
        response = Mock(ok=True)
        response.json.return_value = {"solution_found": True}
        with patch.object(api_client.requests, "post", return_value=response) as post:
            self.assertTrue(api_client.compute_deviated_stability(payload)["solution_found"])
        sent = post.call_args.kwargs["json"]
        json.dumps(sent, allow_nan=False)
        self.assertEqual(len(sent["results"]), 1)
        self.assertEqual(sent["results"][0]["Depth"], 1000)
        for key in ("Missing", "PositiveInfinity", "NegativeInfinity"):
            self.assertIsNone(sent["results"][0][key])
        self.assertNotEqual(payload["results"][0]["Missing"], payload["results"][0]["Missing"])

    def test_invalid_scalar_fails_before_network_call(self):
        payload = self.payload()
        payload["depth_m"] = float("nan")
        with patch.object(api_client.requests, "post") as post:
            with self.assertRaisesRegex(api_client.BackendError, "depth_m.*finite"):
                api_client.compute_deviated_stability(payload)
            post.assert_not_called()

    def test_experimental_tab3_works_with_unrelated_missing_cells(self):
        client = TestClient(app)
        response = client.post("/api/v1/mem/compute",
                               files={"las_file": ("synthetic.csv", b"TVD,DT,RHOB,DTS\n1000,90,2.2,180\n1010,90,2.21,181\n1020,90,2.22,182\n")},
                               data=dict(depth_col="TVD", dt_col="DT", rhob_col="RHOB", dts_col="DTS",
                                         depth_reference="TVD", depth_unit="m", dt_surface=80,
                                         dt_matrix=55.5, calculation_mode="educational", tectonic_ex=0, tectonic_ey=0))
        self.assertEqual(response.status_code, 200, response.text)
        payload = self.payload()
        payload["results"] = response.json()["results"]
        payload["depth_m"] = 1010
        for row in payload["results"]:
            row["Unused_Test_Column"] = float("nan")

        def send(url, json, timeout):
            response = client.post("/api/v1/mem/deviated_stability", json=json)
            # TestClient uses httpx; production uses requests.Response.
            response.ok = response.is_success
            return response

        with patch.object(api_client.requests, "post", side_effect=send):
            result = api_client.compute_deviated_stability(payload)
            self.assertEqual(result["Calculation_Mode"], "educational")
            self.assertEqual(len(result["sigma_tt_eff"]), 181)
            invalid = copy.deepcopy(payload)
            invalid["results"][1]["Pore_Pressure_Pp_MPa"] = float("nan")
            with self.assertRaisesRegex(api_client.BackendError, "missing or non-finite"):
                api_client.compute_deviated_stability(invalid)


if __name__ == "__main__":
    unittest.main()
