"""Helpers for displaying depths while keeping calculations in metres."""

FEET_PER_METRE = 3.280839895013123
SUPPORTED_DEPTH_UNITS = {"m", "ft"}


def _factor(unit: str) -> float:
    if unit not in SUPPORTED_DEPTH_UNITS:
        raise ValueError(f"Unsupported depth unit: {unit!r}. Choose 'm' or 'ft'.")
    return FEET_PER_METRE if unit == "ft" else 1.0


def depth_from_meters(depth_m, unit: str):
    """Convert metre values to the requested display unit; supports scalars and pandas objects."""
    return depth_m * _factor(unit)


def depth_to_meters(depth_value, unit: str):
    """Convert a displayed depth value back to metres for calculations and API calls."""
    return depth_value / _factor(unit)
