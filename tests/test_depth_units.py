import unittest

import pandas as pd

from frontend.depth_units import depth_from_meters, depth_to_meters


class DepthUnitDisplayTests(unittest.TestCase):
    def test_metre_display_is_unchanged(self):
        values = pd.Series([0.0, 1000.0, 2500.0])
        pd.testing.assert_series_equal(depth_from_meters(values, "m"), values)

    def test_metre_values_display_in_feet(self):
        values = pd.Series([0.0, 304.8, 1000.0])
        displayed = depth_from_meters(values, "ft")
        self.assertAlmostEqual(displayed.iloc[1], 1000.0, places=8)
        self.assertAlmostEqual(displayed.iloc[2], 3280.839895, places=5)

    def test_displayed_feet_round_trip_to_metres(self):
        displayed = pd.Series([0.0, 1000.0, 3280.839895])
        converted = depth_to_meters(displayed, "ft")
        pd.testing.assert_series_equal(
            converted,
            pd.Series([0.0, 304.8, 999.999999968], dtype="float64"),
            check_exact=False,
            atol=1e-6,
        )

    def test_scalar_depth_conversion(self):
        self.assertAlmostEqual(depth_to_meters(1000.0, "ft"), 304.8, places=8)
        self.assertAlmostEqual(depth_from_meters(304.8, "ft"), 1000.0, places=8)

    def test_unsupported_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported depth unit"):
            depth_from_meters(100.0, "yards")


if __name__ == "__main__":
    unittest.main()
