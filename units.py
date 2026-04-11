from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LENGTH_FACTORS_TO_METERS = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
}

FORCE_FACTORS_TO_NEWTONS = {
    "N": 1.0,
    "kN": 1e3,
}

MOMENT_FACTORS_TO_NM = {
    "Nmm": 1e-3,
    "Nm": 1.0,
    "kNm": 1e3,
}


@dataclass(frozen=True)
class UnitPolicy:
    user_length_unit: str = "m"
    user_force_unit: str = "N"
    solver_geometry_length_unit: str = "mm"
    solver_force_unit: str = "N"
    solver_coordinate_unit: str = "mm"
    solver_moment_unit: str = "Nmm"
    solver_displacement_unit: str = "m"
    visualization_length_unit: str = "m"
    visualization_displacement_unit: str = "mm"
    visualization_force_unit: str = "kN"
    visualization_moment_unit: str = "kNm"
    visualization_rotation_unit: str = "rad"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_UNIT_POLICY = UnitPolicy()


def _require_supported(unit: str, table: dict[str, float], kind: str) -> None:
    if unit not in table:
        raise ValueError(f"Unsupported {kind} unit: {unit}")


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    _require_supported(from_unit, LENGTH_FACTORS_TO_METERS, "length")
    _require_supported(to_unit, LENGTH_FACTORS_TO_METERS, "length")
    meters = value * LENGTH_FACTORS_TO_METERS[from_unit]
    return meters / LENGTH_FACTORS_TO_METERS[to_unit]


def convert_force(value: float, from_unit: str, to_unit: str) -> float:
    _require_supported(from_unit, FORCE_FACTORS_TO_NEWTONS, "force")
    _require_supported(to_unit, FORCE_FACTORS_TO_NEWTONS, "force")
    newtons = value * FORCE_FACTORS_TO_NEWTONS[from_unit]
    return newtons / FORCE_FACTORS_TO_NEWTONS[to_unit]


def convert_moment(value: float, from_unit: str, to_unit: str) -> float:
    _require_supported(from_unit, MOMENT_FACTORS_TO_NM, "moment")
    _require_supported(to_unit, MOMENT_FACTORS_TO_NM, "moment")
    nm = value * MOMENT_FACTORS_TO_NM[from_unit]
    return nm / MOMENT_FACTORS_TO_NM[to_unit]


def normalize_user_params(
    params: dict[str, Any],
    policy: UnitPolicy = DEFAULT_UNIT_POLICY,
    input_length_unit: str | None = None,
    input_force_unit: str | None = None,
) -> dict[str, Any]:
    """
    Convert user-provided geometry/load values into solver-facing units.
    """
    user_length_unit = input_length_unit or policy.user_length_unit
    user_force_unit = input_force_unit or policy.user_force_unit
    out = dict(params)
    out["length_m"] = convert_length(
        float(out["length_m"]),
        user_length_unit,
        policy.solver_geometry_length_unit,
    )
    if out.get("HY") is not None:
        out["HY"] = convert_length(
            float(out["HY"]),
            user_length_unit,
            policy.solver_geometry_length_unit,
        )
    if out.get("HZ") is not None:
        out["HZ"] = convert_length(
            float(out["HZ"]),
            user_length_unit,
            policy.solver_geometry_length_unit,
        )
    out["FY"] = convert_force(
        float(out["FY"]),
        user_force_unit,
        policy.solver_force_unit,
    )
    return out


def normalize_user_params_mixed_units(
    params: dict[str, Any],
    policy: UnitPolicy = DEFAULT_UNIT_POLICY,
    length_unit: str | None = None,
    hy_unit: str | None = None,
    hz_unit: str | None = None,
    force_unit: str | None = None,
) -> dict[str, Any]:
    """
    Convert user-provided geometry/load values into solver-facing units
    when each value may use a different unit.
    """
    out = dict(params)
    out["length_m"] = convert_length(
        float(out["length_m"]),
        length_unit or policy.user_length_unit,
        policy.solver_geometry_length_unit,
    )
    if out.get("HY") is not None:
        out["HY"] = convert_length(
            float(out["HY"]),
            hy_unit or policy.user_length_unit,
            policy.solver_geometry_length_unit,
        )
    if out.get("HZ") is not None:
        out["HZ"] = convert_length(
            float(out["HZ"]),
            hz_unit or policy.user_length_unit,
            policy.solver_geometry_length_unit,
        )
    out["FY"] = convert_force(
        float(out["FY"]),
        force_unit or policy.user_force_unit,
        policy.solver_force_unit,
    )
    return out
