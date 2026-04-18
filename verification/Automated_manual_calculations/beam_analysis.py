"""Beam solver core logic.

This module loads either JSON inputs or Code_Aster input files, solves the
statically determinate beam, and prepares the final result payload.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EPSILON = 1e-9
BACKEND_NAME = "verification-codeasterprj"
BACKEND_VERSION = "0.1.0"
N_PER_MM2_TO_PA = 1_000_000.0


# Basic data containers used across the solver.
@dataclass(frozen=True)
class Support:
    name: str
    type: str
    x: float


@dataclass(frozen=True)
class PointForce:
    x: float
    fx: float
    fy: float
    label: str | None = None


@dataclass(frozen=True)
class PointMoment:
    x: float
    mz: float
    label: str | None = None


@dataclass(frozen=True)
class DistributedForce:
    x_start: float
    x_end: float
    qx_start: float
    qx_end: float
    qy_start: float
    qy_end: float
    label: str | None = None


@dataclass(frozen=True)
class DistributedMoment:
    x_start: float
    x_end: float
    mz_start: float
    mz_end: float
    label: str | None = None


@dataclass(frozen=True)
class BeamModel:
    length: float
    section: dict[str, Any]
    material: dict[str, Any]
    supports: list[Support]
    point_forces: list[PointForce]
    point_moments: list[PointMoment]
    distributed_forces: list[DistributedForce]
    distributed_moments: list[DistributedMoment]
    stations: int
    model_id: str | None = None
    case_type: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    response_positions: list[float] = field(default_factory=list)
    num_elements: int | None = None


@dataclass(frozen=True)
class DeflectionProfile:
    x_values: list[float]
    slope_values: list[float]
    displacement_values: list[float]
    flexural_rigidity: float
    inertia_field: str


# Generic parsing helpers.
def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be numeric.") from exc


def _parse_linear_value(raw_value: Any, field_name: str) -> tuple[float, float]:
    if raw_value is None:
        return 0.0, 0.0
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return value, value
    if isinstance(raw_value, dict):
        start = _as_float(raw_value.get("start"), f"{field_name}.start")
        end = _as_float(raw_value.get("end"), f"{field_name}.end")
        return start, end
    raise ValueError(
        f"Field '{field_name}' must be a number or an object with 'start' and 'end'."
    )


def _normalize_support_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    aliases = {
        "roller": "roller_y",
        "fixed_support": "fixed",
        "hinge": "pin",
        "pinned": "pin",
    }
    return aliases.get(normalized, normalized)


def _support_components(support_type: str) -> list[str]:
    components = {
        "fixed": ["Rx", "Ry", "Mz"],
        "pin": ["Rx", "Ry"],
        "roller_y": ["Ry"],
        "roller_x": ["Rx"],
        "free": [],
    }.get(support_type)
    if components is None:
        raise ValueError(
            "Unsupported support type. Use one of: fixed, pin, roller_y, roller_x, free."
        )
    return components


def _direction_to_components(direction: str, magnitude: float) -> tuple[float, float]:
    signed_magnitude = abs(magnitude)
    components = {
        "x_positive": (signed_magnitude, 0.0),
        "x_negative": (-signed_magnitude, 0.0),
        "y_positive": (0.0, signed_magnitude),
        "y_negative": (0.0, -signed_magnitude),
    }.get(direction)
    if components is None:
        raise ValueError(
            "load.direction must be one of: x_positive, x_negative, y_positive, y_negative."
        )
    return components


def _load_canonical_beam_model(data: dict[str, Any]) -> BeamModel:
    """Load the canonical case-driven JSON input format."""
    case_type = str(data.get("case_type", "")).strip()
    beam_data = data.get("beam")
    material = data.get("material")
    section = data.get("section")
    load_data = data.get("load")

    if not isinstance(beam_data, dict):
        raise ValueError("Input JSON must contain a 'beam' object.")
    if not isinstance(material, dict):
        raise ValueError("Input JSON must contain a 'material' object.")
    if not isinstance(section, dict):
        raise ValueError("Input JSON must contain a 'section' object.")
    if not isinstance(load_data, dict):
        raise ValueError("Input JSON must contain a 'load' object.")

    length = _as_float(beam_data.get("length"), "beam.length")
    if length <= 0.0:
        raise ValueError("Beam length must be greater than zero.")

    num_elements = int(beam_data.get("num_elements", 20))
    if num_elements < 1:
        raise ValueError("beam.num_elements must be at least 1.")

    analysis_data = data.get("analysis", {})
    if analysis_data is None:
        analysis_data = {}
    if not isinstance(analysis_data, dict):
        raise ValueError("analysis must be an object when provided.")

    stations = int(analysis_data.get("stations", 201))
    if stations < 2:
        raise ValueError("analysis.stations must be at least 2.")

    response_positions = [
        _as_float(value, f"analysis.response_positions[{index}]")
        for index, value in enumerate(analysis_data.get("response_positions", []))
    ]
    for position in response_positions:
        if not 0.0 <= position <= length:
            raise ValueError("analysis.response_positions values must lie on the beam.")

    point_load_type = str(load_data.get("type", "")).strip().lower()
    if point_load_type != "point_force":
        raise ValueError("Canonical input currently supports only load.type='point_force'.")

    magnitude = _as_float(load_data.get("magnitude"), "load.magnitude")
    direction = str(load_data.get("direction", "")).strip()
    fx, fy = _direction_to_components(direction, magnitude)

    if case_type == "cantilever_tip_load":
        supports = [Support(name="A", type="fixed", x=0.0)]
        point_forces = [PointForce(x=length, fx=fx, fy=fy, label="TIP")]
    elif case_type == "simply_supported_midspan_point_load":
        supports = [
            Support(name="A", type="pin", x=0.0),
            Support(name="B", type="roller_y", x=length),
        ]
        point_forces = [PointForce(x=0.5 * length, fx=fx, fy=fy, label="MIDSPAN")]
    else:
        raise ValueError(
            "case_type must be one of: cantilever_tip_load, simply_supported_midspan_point_load."
        )

    return BeamModel(
        length=length,
        section=section,
        material=material,
        supports=supports,
        point_forces=point_forces,
        point_moments=[],
        distributed_forces=[],
        distributed_moments=[],
        stations=stations,
        model_id=str(data["model_id"]) if "model_id" in data else None,
        case_type=case_type,
        units={
            str(key): str(value)
            for key, value in data.get("units", {}).items()
            if value is not None
        },
        response_positions=response_positions,
        num_elements=num_elements,
    )


def _load_legacy_beam_model(data: dict[str, Any]) -> BeamModel:
    """Load the previous explicit beam JSON format."""
    beam_data = data.get("beam")
    if not isinstance(beam_data, dict):
        raise ValueError("Input JSON must contain a 'beam' object.")

    length = _as_float(beam_data.get("length"), "beam.length")
    if length <= 0.0:
        raise ValueError("Beam length must be greater than zero.")

    section = beam_data.get("section", {})
    material = beam_data.get("material", {})
    stations = int(data.get("analysis", {}).get("stations", 201))
    if stations < 2:
        raise ValueError("analysis.stations must be at least 2.")

    supports: list[Support] = []
    for index, raw_support in enumerate(data.get("supports", [])):
        support = Support(
            name=str(raw_support.get("name", f"S{index + 1}")),
            type=_normalize_support_type(str(raw_support.get("type", ""))),
            x=_as_float(raw_support.get("x"), f"supports[{index}].x"),
        )
        if not 0.0 <= support.x <= length:
            raise ValueError(f"Support '{support.name}' lies outside the beam length.")
        _support_components(support.type)
        supports.append(support)

    if not supports:
        raise ValueError("At least one support must be defined.")

    point_forces: list[PointForce] = []
    point_moments: list[PointMoment] = []
    distributed_forces: list[DistributedForce] = []
    distributed_moments: list[DistributedMoment] = []

    for index, raw_load in enumerate(data.get("loads", [])):
        load_type = str(raw_load.get("type", "")).strip().lower()
        label = raw_load.get("label")

        if load_type == "point_force":
            point_force = PointForce(
                x=_as_float(raw_load.get("x"), f"loads[{index}].x"),
                fx=_as_float(raw_load.get("fx", 0.0), f"loads[{index}].fx"),
                fy=_as_float(raw_load.get("fy", 0.0), f"loads[{index}].fy"),
                label=str(label) if label is not None else None,
            )
            point_forces.append(point_force)
            continue

        if load_type == "point_moment":
            point_moments.append(
                PointMoment(
                    x=_as_float(raw_load.get("x"), f"loads[{index}].x"),
                    mz=_as_float(raw_load.get("mz"), f"loads[{index}].mz"),
                    label=str(label) if label is not None else None,
                )
            )
            continue

        if load_type == "distributed_force":
            x_start = _as_float(raw_load.get("x_start"), f"loads[{index}].x_start")
            x_end = _as_float(raw_load.get("x_end"), f"loads[{index}].x_end")
            qx_start, qx_end = _parse_linear_value(
                raw_load.get("qx"), f"loads[{index}].qx"
            )
            qy_start, qy_end = _parse_linear_value(
                raw_load.get("qy"), f"loads[{index}].qy"
            )
            distributed_forces.append(
                DistributedForce(
                    x_start=x_start,
                    x_end=x_end,
                    qx_start=qx_start,
                    qx_end=qx_end,
                    qy_start=qy_start,
                    qy_end=qy_end,
                    label=str(label) if label is not None else None,
                )
            )
            continue

        if load_type == "distributed_moment":
            x_start = _as_float(raw_load.get("x_start"), f"loads[{index}].x_start")
            x_end = _as_float(raw_load.get("x_end"), f"loads[{index}].x_end")
            mz_start, mz_end = _parse_linear_value(
                raw_load.get("mz"), f"loads[{index}].mz"
            )
            distributed_moments.append(
                DistributedMoment(
                    x_start=x_start,
                    x_end=x_end,
                    mz_start=mz_start,
                    mz_end=mz_end,
                    label=str(label) if label is not None else None,
                )
            )
            continue

        raise ValueError(
            "Unsupported load type. Use point_force, point_moment, distributed_force, "
            "or distributed_moment."
        )

    _validate_load_locations(
        length, point_forces, point_moments, distributed_forces, distributed_moments
    )

    return BeamModel(
        length=length,
        section=section,
        material=material,
        supports=supports,
        point_forces=point_forces,
        point_moments=point_moments,
        distributed_forces=distributed_forces,
        distributed_moments=distributed_moments,
        stations=stations,
    )


def load_beam_model(path: str | Path) -> BeamModel:
    """Load the solver input from either the canonical or legacy JSON format."""
    data = json.loads(Path(path).read_text())
    if "case_type" in data:
        return _load_canonical_beam_model(data)
    return _load_legacy_beam_model(data)


# Code_Aster text parsing helpers.
def _extract_parenthesized(text: str, open_paren_index: int) -> tuple[str, int]:
    depth = 0
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index + 1 : index], index
    raise ValueError("Unbalanced parentheses while parsing Code_Aster input.")


def _find_call_blocks(text: str, call_name: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(rf"\b{re.escape(call_name)}\s*\(")
    for match in pattern.finditer(text):
        open_paren_index = match.end() - 1
        _, close_paren_index = _extract_parenthesized(text, open_paren_index)
        blocks.append(text[match.start() : close_paren_index + 1])
    return blocks


def _extract_f_entries(block: str, keyword: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(keyword)}\s*=")
    match = pattern.search(block)
    if match is None:
        return []

    index = match.end()
    while index < len(block) and block[index].isspace():
        index += 1

    if block.startswith("_F(", index):
        entry, _ = _extract_parenthesized(block, index + 2)
        return [entry]

    if index < len(block) and block[index] == "(":
        content, _ = _extract_parenthesized(block, index)
        entries: list[str] = []
        sub_index = 0
        while sub_index < len(content):
            f_index = content.find("_F(", sub_index)
            if f_index == -1:
                break
            entry, end_index = _extract_parenthesized(content, f_index + 2)
            entries.append(entry)
            sub_index = end_index + 1
        return entries

    return []


def _extract_numeric_assignments(text: str, names: tuple[str, ...]) -> dict[str, float]:
    assignments: dict[str, float] = {}
    for name in names:
        match = re.search(
            rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text
        )
        if match is not None:
            assignments[name] = float(match.group(1))
    return assignments


def _extract_string_assignment(text: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*'([^']+)'", text)
    if match is None:
        return None
    return match.group(1)


def _extract_tuple_strings(text: str, name: str) -> list[str]:
    match = re.search(rf"\b{name}\s*=\s*\((.*?)\)", text, flags=re.DOTALL)
    if match is None:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def _extract_tuple_numbers(text: str, name: str) -> list[float]:
    match = re.search(rf"\b{name}\s*=\s*\((.*?)\)", text, flags=re.DOTALL)
    if match is None:
        return []
    return [
        float(value)
        for value in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(1))
    ]


def _parse_code_aster_export(path: str | Path) -> dict[str, str]:
    """Read small metadata items from the Code_Aster export file."""
    metadata: dict[str, str] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("P nomjob "):
            metadata["job_name"] = line.split(maxsplit=2)[2]
    return metadata


def _parse_code_aster_geo(path: str | Path) -> dict[str, Any]:
    """Read beam geometry, group positions, and station count from Gmsh geometry."""
    text = Path(path).read_text()
    geometry_in_mm = "(mm)" in text

    points = {
        int(point_id): (float(x), float(y), float(z))
        for point_id, x, y, z in re.findall(
            r"Point\((\d+)\)\s*=\s*\{\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\};",
            text,
        )
    }
    if not points:
        raise ValueError("No Point definitions found in beam.geo.")

    lines = {
        int(line_id): (int(start_point), int(end_point))
        for line_id, start_point, end_point in re.findall(
            r"Line\((\d+)\)\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*\};",
            text,
        )
    }
    if not lines:
        raise ValueError("No Line definitions found in beam.geo.")

    physical_points = {
        name: [int(point_id) for point_id in re.findall(r"\d+", point_ids)]
        for name, point_ids in re.findall(
            r'Physical Point\("([^"]+)"\)\s*=\s*\{([^}]*)\};',
            text,
        )
    }
    physical_curves = {
        name: [int(line_id) for line_id in re.findall(r"\d+", line_ids)]
        for name, line_ids in re.findall(
            r'Physical Curve\("([^"]+)"\)\s*=\s*\{([^}]*)\};',
            text,
        )
    }

    beam_line_ids = physical_curves.get("BEAM")
    if not beam_line_ids:
        raise ValueError('beam.geo must define Physical Curve("BEAM").')

    beam_length = 0.0
    for line_id in beam_line_ids:
        if line_id not in lines:
            raise ValueError(
                f"beam.geo references undefined line {line_id} in BEAM group."
            )
        start_point, end_point = lines[line_id]
        x0, y0, z0 = points[start_point]
        x1, y1, z1 = points[end_point]
        beam_length += math.dist((x0, y0, z0), (x1, y1, z1))

    transfinite_match = re.search(
        r"Transfinite Line\s*\{[^}]*\}\s*=\s*(\d+)\s*;",
        text,
    )
    stations = int(transfinite_match.group(1)) if transfinite_match is not None else 201

    group_positions: dict[str, float] = {}
    for group_name, point_ids in physical_points.items():
        x_positions = [points[point_id][0] for point_id in point_ids]
        if not x_positions:
            continue
        if max(x_positions) - min(x_positions) > EPSILON:
            raise ValueError(
                f"Physical Point group '{group_name}' spans multiple x locations, which is unsupported."
            )
        group_positions[group_name] = x_positions[0]

    if geometry_in_mm:
        beam_length /= 1000.0
        group_positions = {
            group_name: position / 1000.0
            for group_name, position in group_positions.items()
        }

    return {
        "length": beam_length,
        "stations": max(stations, 2),
        "group_positions": group_positions,
        "physical_curves": physical_curves,
    }


def _parse_code_aster_section(comm_text: str) -> dict[str, Any]:
    """Read section data from the beam property definition."""
    blocks = _find_call_blocks(comm_text, "AFFE_CARA_ELEM")
    if not blocks:
        return {}

    beam_definition_entries = _extract_f_entries(blocks[0], "POUTRE")
    if not beam_definition_entries:
        return {}

    beam_definition = beam_definition_entries[0]
    section_type = _extract_string_assignment(beam_definition, "SECTION")
    cara_names = _extract_tuple_strings(beam_definition, "CARA")
    cara_values = _extract_tuple_numbers(beam_definition, "VALE")
    section: dict[str, Any] = {
        "section_type": section_type,
    }

    for name, value in zip(cara_names, cara_values, strict=False):
        section[name] = value

    if section_type == "GENERALE":
        for area_name in ("A", "AY", "AZ"):
            area_value = section.get(area_name)
            if isinstance(area_value, (int, float)):
                section[area_name] = area_value / 1_000_000.0
        for inertia_name in ("IY", "IZ", "JX"):
            inertia_value = section.get(inertia_name)
            if isinstance(inertia_value, (int, float)):
                section[inertia_name] = inertia_value / 1_000_000_000_000.0

    # Code_Aster sample beam sections are defined in millimetres while the
    # beam geometry is expressed in metres. Normalize section dimensions to SI
    # before deriving area and inertia so displacement uses a consistent EI.
    if section_type == "RECTANGLE":
        for dimension_name in ("HY", "HZ"):
            dimension_value = section.get(dimension_name)
            if isinstance(dimension_value, (int, float)):
                section[dimension_name] = dimension_value / 1000.0

    if section_type == "RECTANGLE":
        hy = section.get("HY")
        hz = section.get("HZ")
        if isinstance(hy, (int, float)) and isinstance(hz, (int, float)):
            section["A"] = hy * hz
            section["Iy"] = hy * hz**3 / 12.0
            section["Iz"] = hz * hy**3 / 12.0

    if "IY" in section and "Iy" not in section:
        section["Iy"] = section["IY"]
    if "IZ" in section and "Iz" not in section:
        section["Iz"] = section["IZ"]

    return section


def _parse_code_aster_material(comm_text: str) -> dict[str, Any]:
    """Read elastic material values from the material definition."""
    blocks = _find_call_blocks(comm_text, "DEFI_MATERIAU")
    if not blocks:
        return {}

    elas_entries = _extract_f_entries(blocks[0], "ELAS")
    if not elas_entries:
        return {}

    properties = _extract_numeric_assignments(
        elas_entries[0],
        ("E", "NU", "RHO"),
    )
    material: dict[str, Any] = {}
    if "E" in properties:
        elastic_modulus = properties["E"]
        if elastic_modulus < 1e7:
            elastic_modulus *= N_PER_MM2_TO_PA
        material["E"] = elastic_modulus
    if "NU" in properties:
        material["nu"] = properties["NU"]
    if "RHO" in properties:
        material["rho"] = properties["RHO"]
    return material


def _support_type_from_ddl_impo(entry: str) -> str:
    """Map Code_Aster fixed degrees of freedom to the solver support types."""
    constrained = {
        name
        for name, value in _extract_numeric_assignments(
            entry,
            ("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        ).items()
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=EPSILON)
    }

    has_dx = "DX" in constrained
    has_dy = "DY" in constrained
    has_drz = "DRZ" in constrained
    other_constraints = constrained - {"DX", "DY", "DRZ", "DZ", "DRX"}

    if has_dx and has_dy and has_drz:
        return "fixed"
    if has_dx and has_dy:
        return "pin"
    if has_dy and not has_dx and not has_drz and not other_constraints:
        return "roller_y"
    if has_dx and not has_dy and not has_drz and not other_constraints:
        return "roller_x"
    if not constrained:
        return "free"
    raise ValueError(
        "Unsupported DDL_IMPO support condition for the 2D solver. "
        "Use a pattern equivalent to fixed, pin, roller_y, or roller_x."
    )


def _parse_code_aster_supports(
    comm_text: str, group_positions: dict[str, float]
) -> list[Support]:
    """Read support positions and support types from DDL_IMPO blocks."""
    supports: list[Support] = []
    for block in _find_call_blocks(comm_text, "AFFE_CHAR_MECA"):
        for entry in _extract_f_entries(block, "DDL_IMPO"):
            group_names = _extract_tuple_strings(entry, "GROUP_NO")
            for group_name in group_names:
                if group_name not in group_positions:
                    raise ValueError(
                        f"Support group '{group_name}' is not defined as a Physical Point in beam.geo."
                    )
                supports.append(
                    Support(
                        name=group_name,
                        type=_support_type_from_ddl_impo(entry),
                        x=group_positions[group_name],
                    )
                )
    return supports


def _parse_code_aster_force_nodale(
    entry: str, group_positions: dict[str, float]
) -> tuple[list[PointForce], list[PointMoment]]:
    """Convert one FORCE_NODALE entry into solver point loads."""
    point_forces: list[PointForce] = []
    point_moments: list[PointMoment] = []
    group_names = _extract_tuple_strings(entry, "GROUP_NO")
    components = _extract_numeric_assignments(entry, ("FX", "FY", "MZ"))

    for group_name in group_names:
        if group_name not in group_positions:
            raise ValueError(
                f"Load group '{group_name}' is not defined as a Physical Point in beam.geo."
            )
        x_position = group_positions[group_name]
        if "FX" in components or "FY" in components:
            point_forces.append(
                PointForce(
                    x=x_position,
                    fx=components.get("FX", 0.0),
                    fy=components.get("FY", 0.0),
                    label=group_name,
                )
            )
        if "MZ" in components:
            point_moments.append(
                PointMoment(
                    x=x_position,
                    mz=components["MZ"],
                    label=group_name,
                )
            )

    return point_forces, point_moments


def _parse_code_aster_force_poutre(
    entry: str, physical_curves: dict[str, list[int]], beam_length: float
) -> tuple[list[DistributedForce], list[DistributedMoment]]:
    """Convert one FORCE_POUTRE entry into solver distributed loads."""
    distributed_forces: list[DistributedForce] = []
    distributed_moments: list[DistributedMoment] = []
    group_names = _extract_tuple_strings(entry, "GROUP_MA")
    if not group_names:
        group_names = ["BEAM"]

    components = _extract_numeric_assignments(entry, ("FX", "FY", "MZ"))
    for group_name in group_names:
        if group_name not in physical_curves:
            raise ValueError(
                f"Load group '{group_name}' is not defined as a Physical Curve in beam.geo."
            )
        if "FX" in components or "FY" in components:
            distributed_forces.append(
                DistributedForce(
                    x_start=0.0,
                    x_end=beam_length,
                    qx_start=components.get("FX", 0.0),
                    qx_end=components.get("FX", 0.0),
                    qy_start=components.get("FY", 0.0),
                    qy_end=components.get("FY", 0.0),
                    label=group_name,
                )
            )
        if "MZ" in components:
            distributed_moments.append(
                DistributedMoment(
                    x_start=0.0,
                    x_end=beam_length,
                    mz_start=components["MZ"],
                    mz_end=components["MZ"],
                    label=group_name,
                )
            )
    return distributed_forces, distributed_moments


def _parse_code_aster_loads(
    comm_text: str,
    group_positions: dict[str, float],
    physical_curves: dict[str, list[int]],
    beam_length: float,
) -> tuple[
    list[PointForce], list[PointMoment], list[DistributedForce], list[DistributedMoment]
]:
    """Read all supported load blocks from the Code_Aster command file."""
    point_forces: list[PointForce] = []
    point_moments: list[PointMoment] = []
    distributed_forces: list[DistributedForce] = []
    distributed_moments: list[DistributedMoment] = []

    for block in _find_call_blocks(comm_text, "AFFE_CHAR_MECA"):
        if "DDL_IMPO" in block:
            continue

        for entry in _extract_f_entries(block, "FORCE_NODALE"):
            parsed_point_forces, parsed_point_moments = _parse_code_aster_force_nodale(
                entry,
                group_positions,
            )
            point_forces.extend(parsed_point_forces)
            point_moments.extend(parsed_point_moments)

        for entry in _extract_f_entries(block, "FORCE_POUTRE"):
            parsed_distributed_forces, parsed_distributed_moments = (
                _parse_code_aster_force_poutre(
                    entry,
                    physical_curves,
                    beam_length,
                )
            )
            distributed_forces.extend(parsed_distributed_forces)
            distributed_moments.extend(parsed_distributed_moments)

    return point_forces, point_moments, distributed_forces, distributed_moments


def load_code_aster_model(path: str | Path) -> BeamModel:
    """Build one beam model from a folder with model.comm, model.export, and beam/model.geo."""
    folder = Path(path)
    comm_path = folder / "model.comm"
    export_path = folder / "model.export"
    geo_path = folder / "beam.geo"
    if not geo_path.exists():
        geo_path = folder / "model.geo"

    missing_paths = [
        file_path.name
        for file_path in (comm_path, export_path, geo_path)
        if not file_path.exists()
    ]
    if missing_paths:
        raise ValueError(
            "Code_Aster input folder is missing required files: "
            + ", ".join(sorted(missing_paths))
        )

    comm_text = comm_path.read_text()
    geo_data = _parse_code_aster_geo(geo_path)
    _parse_code_aster_export(export_path)

    supports = _parse_code_aster_supports(comm_text, geo_data["group_positions"])
    if not supports:
        raise ValueError("No supported beam supports were found in model.comm.")

    point_forces, point_moments, distributed_forces, distributed_moments = (
        _parse_code_aster_loads(
            comm_text,
            geo_data["group_positions"],
            geo_data["physical_curves"],
            geo_data["length"],
        )
    )

    _validate_load_locations(
        geo_data["length"],
        point_forces,
        point_moments,
        distributed_forces,
        distributed_moments,
    )

    return BeamModel(
        length=geo_data["length"],
        section=_parse_code_aster_section(comm_text),
        material=_parse_code_aster_material(comm_text),
        supports=supports,
        point_forces=point_forces,
        point_moments=point_moments,
        distributed_forces=distributed_forces,
        distributed_moments=distributed_moments,
        stations=geo_data["stations"],
        num_elements=max(geo_data["stations"] - 1, 1),
    )


# Validation and statics solution helpers.
def _validate_load_locations(
    length: float,
    point_forces: list[PointForce],
    point_moments: list[PointMoment],
    distributed_forces: list[DistributedForce],
    distributed_moments: list[DistributedMoment],
) -> None:
    for point_force in point_forces:
        if not 0.0 <= point_force.x <= length:
            raise ValueError("A point force lies outside the beam length.")
    for point_moment in point_moments:
        if not 0.0 <= point_moment.x <= length:
            raise ValueError("A point moment lies outside the beam length.")
    for distributed_force in distributed_forces:
        if not (0.0 <= distributed_force.x_start < distributed_force.x_end <= length):
            raise ValueError("A distributed force has invalid start/end coordinates.")
    for distributed_moment in distributed_moments:
        if not (0.0 <= distributed_moment.x_start < distributed_moment.x_end <= length):
            raise ValueError("A distributed moment has invalid start/end coordinates.")


def _solve_linear_system_3x3(
    matrix: list[list[float]], rhs: list[float]
) -> list[float]:
    """Solve the 3x3 equilibrium system for the reaction unknowns."""
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs, strict=True)]

    for pivot_index in range(3):
        pivot_row = max(
            range(pivot_index, 3),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < EPSILON:
            raise ValueError(
                "Support configuration is not solvable by 2D statics. "
                "Use a statically determinate set of reactions."
            )
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, 4):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(3):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, 4):
                augmented[row_index][column_index] -= (
                    factor * augmented[pivot_index][column_index]
                )

    return [augmented[row_index][3] for row_index in range(3)]


def _distributed_resultant(
    start_value: float, end_value: float, length: float
) -> float:
    return 0.5 * (start_value + end_value) * length


def _distributed_first_moment_about_start(
    start_value: float, end_value: float, length: float
) -> float:
    slope = (end_value - start_value) / length
    return 0.5 * start_value * length * length + (slope * length**3) / 3.0


def solve_reactions(model: BeamModel) -> dict[str, dict[str, float]]:
    """Solve the support reactions from global force and moment equilibrium."""
    unknowns: list[tuple[str, str, float]] = []
    for support in model.supports:
        for component in _support_components(support.type):
            unknowns.append((support.name, component, support.x))

    if len(unknowns) != 3:
        raise ValueError(
            f"This solver handles statically determinate 2D beams with 3 reaction unknowns. "
            f"The current support set creates {len(unknowns)} unknowns."
        )

    force_x_total = sum(load.fx for load in model.point_forces)
    force_y_total = sum(load.fy for load in model.point_forces)
    moment_total = sum(load.mz for load in model.point_moments)
    moment_total += sum(load.x * load.fy for load in model.point_forces)

    for load in model.distributed_forces:
        span = load.x_end - load.x_start
        resultant_x = _distributed_resultant(load.qx_start, load.qx_end, span)
        resultant_y = _distributed_resultant(load.qy_start, load.qy_end, span)
        force_x_total += resultant_x
        force_y_total += resultant_y
        moment_total += (
            load.x_start * resultant_y
            + _distributed_first_moment_about_start(load.qy_start, load.qy_end, span)
        )

    for load in model.distributed_moments:
        span = load.x_end - load.x_start
        moment_total += _distributed_resultant(load.mz_start, load.mz_end, span)

    matrix = [
        [1.0 if component == "Rx" else 0.0 for _, component, _ in unknowns],
        [1.0 if component == "Ry" else 0.0 for _, component, _ in unknowns],
        [
            x if component == "Ry" else 1.0 if component == "Mz" else 0.0
            for _, component, x in unknowns
        ],
    ]
    rhs = [-force_x_total, -force_y_total, -moment_total]
    values = _solve_linear_system_3x3(matrix, rhs)

    reactions: dict[str, dict[str, float]] = {
        support.name: {} for support in model.supports
    }
    for (support_name, component, _), value in zip(unknowns, values, strict=True):
        reactions[support_name][component] = value

    return reactions


def _reaction_point_actions(
    model: BeamModel, reactions: dict[str, dict[str, float]]
) -> tuple[list[PointForce], list[PointMoment]]:
    """Turn support reactions into equivalent point actions on the beam."""
    point_forces: list[PointForce] = []
    point_moments: list[PointMoment] = []

    for support in model.supports:
        support_reactions = reactions.get(support.name, {})
        point_forces.append(
            PointForce(
                x=support.x,
                fx=support_reactions.get("Rx", 0.0),
                fy=support_reactions.get("Ry", 0.0),
                label=f"Reaction {support.name}",
            )
        )
        if "Mz" in support_reactions:
            point_moments.append(
                PointMoment(
                    x=support.x,
                    mz=support_reactions["Mz"],
                    label=f"Reaction moment {support.name}",
                )
            )

    return point_forces, point_moments


def _partial_linear_resultant(
    start_value: float, end_value: float, span_length: float, partial_length: float
) -> float:
    slope = (end_value - start_value) / span_length
    return start_value * partial_length + 0.5 * slope * partial_length**2


def _partial_linear_first_moment_about_start(
    start_value: float, end_value: float, span_length: float, partial_length: float
) -> float:
    slope = (end_value - start_value) / span_length
    return 0.5 * start_value * partial_length**2 + (slope * partial_length**3) / 3.0


def evaluate_internal_actions(
    model: BeamModel, reactions: dict[str, dict[str, float]], x: float
) -> dict[str, float]:
    """Evaluate axial force, shear force, and bending moment at one position."""
    if not 0.0 <= x <= model.length:
        raise ValueError("Evaluation coordinate lies outside the beam length.")

    # Add the support reactions to the user loads before cutting the beam at x.
    reaction_forces, reaction_moments = _reaction_point_actions(model, reactions)
    point_forces = [*model.point_forces, *reaction_forces]
    point_moments = [*model.point_moments, *reaction_moments]

    force_x_left = 0.0
    force_y_left = 0.0
    moment_left = 0.0

    for load in point_forces:
        if load.x <= x + EPSILON:
            force_x_left += load.fx
            force_y_left += load.fy
            moment_left += (load.x - x) * load.fy

    for load in point_moments:
        if load.x <= x + EPSILON:
            moment_left += load.mz

    for load in model.distributed_forces:
        if x <= load.x_start + EPSILON:
            continue
        partial_end = min(x, load.x_end)
        partial_length = partial_end - load.x_start
        if partial_length <= 0.0:
            continue
        span_length = load.x_end - load.x_start
        resultant_x = _partial_linear_resultant(
            load.qx_start, load.qx_end, span_length, partial_length
        )
        resultant_y = _partial_linear_resultant(
            load.qy_start, load.qy_end, span_length, partial_length
        )
        first_moment_y = _partial_linear_first_moment_about_start(
            load.qy_start, load.qy_end, span_length, partial_length
        )
        force_x_left += resultant_x
        force_y_left += resultant_y
        moment_left += load.x_start * resultant_y + first_moment_y - x * resultant_y

    for load in model.distributed_moments:
        if x <= load.x_start + EPSILON:
            continue
        partial_end = min(x, load.x_end)
        partial_length = partial_end - load.x_start
        if partial_length <= 0.0:
            continue
        span_length = load.x_end - load.x_start
        moment_left += _partial_linear_resultant(
            load.mz_start, load.mz_end, span_length, partial_length
        )

    return {
        "N": -force_x_left,
        "V": -force_y_left,
        "M": -moment_left,
    }


# Extrema search helpers.
def _global_linear_coefficients(
    start_x: float, end_x: float, start_value: float, end_value: float
) -> tuple[float, float]:
    slope = (end_value - start_value) / (end_x - start_x)
    intercept = start_value - slope * start_x
    return intercept, slope


def _event_positions(model: BeamModel) -> list[float]:
    positions = {0.0, model.length}
    positions.update(support.x for support in model.supports)
    positions.update(load.x for load in model.point_forces)
    positions.update(load.x for load in model.point_moments)
    positions.update(load.x_start for load in model.distributed_forces)
    positions.update(load.x_end for load in model.distributed_forces)
    positions.update(load.x_start for load in model.distributed_moments)
    positions.update(load.x_end for load in model.distributed_moments)
    return sorted(positions)


def _build_event_actions(
    model: BeamModel, reactions: dict[str, dict[str, float]]
) -> dict[float, dict[str, float]]:
    events = {x: {"fx": 0.0, "fy": 0.0, "mz": 0.0} for x in _event_positions(model)}

    reaction_forces, reaction_moments = _reaction_point_actions(model, reactions)

    for load in [*model.point_forces, *reaction_forces]:
        events[load.x]["fx"] += load.fx
        events[load.x]["fy"] += load.fy

    for load in [*model.point_moments, *reaction_moments]:
        events[load.x]["mz"] += load.mz

    return events


def _interval_load_coefficients(
    model: BeamModel, x0: float, x1: float
) -> dict[str, float]:
    ax = bx = ay = by = am = bm = 0.0

    for load in model.distributed_forces:
        if load.x_start <= x0 + EPSILON and load.x_end >= x1 - EPSILON:
            intercept_x, slope_x = _global_linear_coefficients(
                load.x_start, load.x_end, load.qx_start, load.qx_end
            )
            intercept_y, slope_y = _global_linear_coefficients(
                load.x_start, load.x_end, load.qy_start, load.qy_end
            )
            ax += intercept_x
            bx += slope_x
            ay += intercept_y
            by += slope_y

    for load in model.distributed_moments:
        if load.x_start <= x0 + EPSILON and load.x_end >= x1 - EPSILON:
            intercept_m, slope_m = _global_linear_coefficients(
                load.x_start, load.x_end, load.mz_start, load.mz_end
            )
            am += intercept_m
            bm += slope_m

    return {"ax": ax, "bx": bx, "ay": ay, "by": by, "am": am, "bm": bm}


def _solve_linear_root(c0: float, c1: float, interval_length: float) -> list[float]:
    if abs(c1) < EPSILON:
        return []
    root = -c0 / c1
    if EPSILON < root < interval_length - EPSILON:
        return [root]
    return []


def _solve_quadratic_roots(
    a: float, b: float, c: float, interval_length: float
) -> list[float]:
    if abs(a) < EPSILON:
        return _solve_linear_root(c, b, interval_length)

    discriminant = b * b - 4.0 * a * c
    if discriminant < -EPSILON:
        return []
    discriminant = max(discriminant, 0.0)
    sqrt_discriminant = math.sqrt(discriminant)
    roots = [
        (-b - sqrt_discriminant) / (2.0 * a),
        (-b + sqrt_discriminant) / (2.0 * a),
    ]
    return sorted(
        {root for root in roots if EPSILON < root < interval_length - EPSILON}
    )


def _evaluate_interval_state(
    start_state: dict[str, float],
    coeffs: dict[str, float],
    x0: float,
    t: float,
) -> dict[str, float]:
    """Advance the internal-force state inside one load interval."""
    qx0 = coeffs["ax"] + coeffs["bx"] * x0
    qy0 = coeffs["ay"] + coeffs["by"] * x0
    mz0 = coeffs["am"] + coeffs["bm"] * x0

    n_value = start_state["N"] - qx0 * t - 0.5 * coeffs["bx"] * t**2
    v_value = start_state["V"] - qy0 * t - 0.5 * coeffs["by"] * t**2
    m_value = (
        start_state["M"]
        - start_state["V"] * t
        + 0.5 * qy0 * t**2
        + (coeffs["by"] * t**3) / 6.0
        - mz0 * t
        - 0.5 * coeffs["bm"] * t**2
    )

    return {"N": n_value, "V": v_value, "M": m_value}


def _build_extrema_entry(
    values: list[dict[str, float]], key: str
) -> dict[str, float | str]:
    # The solver keeps left/right/inside samples, so the reported location can
    # point to the exact cut position where the extreme occurs.
    maximum = max(values, key=lambda item: item[key])
    minimum = min(values, key=lambda item: item[key])
    absolute = max(values, key=lambda item: abs(item[key]))
    return {
        "max": maximum[key],
        "max_x": maximum["x"],
        "max_side": maximum["side"],
        "min": minimum[key],
        "min_x": minimum["x"],
        "min_side": minimum["side"],
        "max_abs": abs(absolute[key]),
        "max_abs_x": absolute["x"],
        "max_abs_side": absolute["side"],
        "max_abs_signed_value": absolute[key],
    }


def _build_profile_extrema(
    x_values: list[float], y_values: list[float]
) -> dict[str, float]:
    """Summarize the peak signed and absolute values from a sampled profile."""
    # This keeps the JSON output compact so reports do not need to scan all stations.
    samples = [
        {"x": x, "value": value} for x, value in zip(x_values, y_values, strict=True)
    ]
    maximum = max(samples, key=lambda item: item["value"])
    minimum = min(samples, key=lambda item: item["value"])
    absolute = max(samples, key=lambda item: abs(item["value"]))
    return {
        "max": maximum["value"],
        "max_x": maximum["x"],
        "min": minimum["value"],
        "min_x": minimum["x"],
        "max_abs": abs(absolute["value"]),
        "max_abs_x": absolute["x"],
        "max_abs_signed_value": absolute["value"],
    }


# Deflection helpers.
def _resolve_flexural_rigidity(model: BeamModel) -> tuple[float, str]:
    youngs_modulus = _as_float(model.material.get("E"), "beam.material.E")
    if youngs_modulus <= 0.0:
        raise ValueError(
            "beam.material.E must be greater than zero for displacement calculation."
        )

    for inertia_field in ("Iz", "Izz", "Iy", "Ix", "I"):
        if inertia_field in model.section:
            second_moment = _as_float(
                model.section.get(inertia_field), f"beam.section.{inertia_field}"
            )
            if second_moment <= 0.0:
                raise ValueError(
                    f"beam.section.{inertia_field} must be greater than zero for displacement calculation."
                )
            return youngs_modulus * second_moment, inertia_field

    raise ValueError(
        "Displacement calculation requires a positive section inertia. "
        "Add one of: Iz, Izz, Iy, Ix, or I."
    )


def _station_positions(model: BeamModel) -> list[float]:
    return [
        model.length * index / (model.stations - 1) for index in range(model.stations)
    ]


def _target_displacement_positions(model: BeamModel) -> list[float]:
    """Build a dense list of positions used for the deflection integration."""
    positions: set[float] = {0.0, model.length}
    positions.update(_event_positions(model))
    positions.update(_station_positions(model))
    # Use a denser sampling than the output stations so displacement extrema are stable.
    dense_station_count = max(model.stations, 2001)
    positions.update(
        model.length * index / (dense_station_count - 1)
        for index in range(dense_station_count)
    )

    supports_by_x = {support.x: support for support in model.supports}
    vertical_restraints = [
        support
        for support in model.supports
        if "Ry" in _support_components(support.type)
    ]

    if (
        len(model.supports) == 2
        and len(vertical_restraints) == 2
        and all(
            "Mz" not in _support_components(support.type) for support in model.supports
        )
    ):
        positions.add(0.5 * (min(supports_by_x) + max(supports_by_x)))

    positions.update(load.x for load in model.point_forces if abs(load.fy) > EPSILON)
    return sorted(positions)


def _lookup_profile_value(
    x_values: list[float], y_values: list[float], x: float
) -> float:
    index = bisect.bisect_left(x_values, x)
    if index < len(x_values) and math.isclose(
        x_values[index], x, rel_tol=0.0, abs_tol=EPSILON
    ):
        return y_values[index]
    if index > 0 and math.isclose(x_values[index - 1], x, rel_tol=0.0, abs_tol=EPSILON):
        return y_values[index - 1]
    if index == 0 or index == len(x_values):
        raise ValueError(
            "Requested displacement position lies outside the sampled profile."
        )

    x0 = x_values[index - 1]
    x1 = x_values[index]
    y0 = y_values[index - 1]
    y1 = y_values[index]
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def _displacement_constraints(model: BeamModel) -> list[tuple[str, float]]:
    constraints: list[tuple[str, float]] = []

    for support in model.supports:
        components = _support_components(support.type)
        if "Mz" in components:
            constraints.append(("theta", support.x))
            constraints.append(("uy", support.x))
            continue
        if "Ry" in components:
            constraints.append(("uy", support.x))

    unique_constraints: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for constraint in constraints:
        if constraint in seen:
            continue
        unique_constraints.append(constraint)
        seen.add(constraint)
    return unique_constraints


def _solve_deflection_integration_constants(
    constraints: list[tuple[str, float]],
    x_values: list[float],
    raw_slope_values: list[float],
    raw_displacement_values: list[float],
) -> tuple[float, float]:
    """Use support boundary conditions to anchor slope and displacement."""
    if len(constraints) < 2:
        raise ValueError(
            "The support layout does not provide enough vertical boundary conditions "
            "to determine beam displacement."
        )

    equations: list[tuple[float, float, float]] = []
    for kind, x in constraints:
        raw_slope = _lookup_profile_value(x_values, raw_slope_values, x)
        raw_displacement = _lookup_profile_value(x_values, raw_displacement_values, x)
        if kind == "theta":
            equations.append((1.0, 0.0, -raw_slope))
        else:
            equations.append((x, 1.0, -raw_displacement))

    determinant = None
    for index, (a1, b1, c1) in enumerate(equations):
        for a2, b2, c2 in equations[index + 1 :]:
            candidate_determinant = a1 * b2 - a2 * b1
            if abs(candidate_determinant) < EPSILON:
                continue
            determinant = candidate_determinant
            c1_value = (c1 * b2 - c2 * b1) / determinant
            c2_value = (a1 * c2 - a2 * c1) / determinant
            return c1_value, c2_value

    raise ValueError(
        "The support layout produced redundant displacement boundary conditions."
    )


def compute_deflection_profile(
    model: BeamModel, reactions: dict[str, dict[str, float]]
) -> DeflectionProfile:
    """Integrate M/EI to build slope and vertical displacement along the beam."""
    flexural_rigidity, inertia_field = _resolve_flexural_rigidity(model)
    x_values = _target_displacement_positions(model)
    curvature_values = [
        evaluate_internal_actions(model, reactions, x)["M"] / flexural_rigidity
        for x in x_values
    ]

    raw_slope_values = [0.0]
    raw_displacement_values = [0.0]
    # First integrate curvature to slope, then integrate slope to displacement.
    for index in range(1, len(x_values)):
        dx = x_values[index] - x_values[index - 1]
        raw_slope_values.append(
            raw_slope_values[-1]
            + 0.5 * (curvature_values[index - 1] + curvature_values[index]) * dx
        )
        raw_displacement_values.append(
            raw_displacement_values[-1]
            + 0.5 * (raw_slope_values[-2] + raw_slope_values[-1]) * dx
        )

    slope_offset, displacement_offset = _solve_deflection_integration_constants(
        _displacement_constraints(model),
        x_values,
        raw_slope_values,
        raw_displacement_values,
    )

    slope_values = [value + slope_offset for value in raw_slope_values]
    displacement_values = [
        value + slope_offset * x + displacement_offset
        for x, value in zip(x_values, raw_displacement_values, strict=True)
    ]

    return DeflectionProfile(
        x_values=x_values,
        slope_values=slope_values,
        displacement_values=displacement_values,
        flexural_rigidity=flexural_rigidity,
        inertia_field=inertia_field,
    )


def _build_displacement_summary(
    model: BeamModel, profile: DeflectionProfile
) -> dict[str, Any]:
    """Prepare the short displacement values shown in the final JSON output."""
    supports_with_vertical_restraint = [
        support
        for support in model.supports
        if "Ry" in _support_components(support.type)
    ]
    summary: dict[str, Any] = {
        "available": True,
        "axis": "vertical_y",
        "rotation_units": "radians",
        "displacement_units": "consistent_with_input_length_units",
        "flexural_rigidity": profile.flexural_rigidity,
        "inertia_field": profile.inertia_field,
        # Report-friendly extrema for the full sampled displacement profile.
        "extrema": {
            "uy": _build_profile_extrema(
                profile.x_values,
                profile.displacement_values,
            ),
            "theta": _build_profile_extrema(
                profile.x_values,
                profile.slope_values,
            ),
        },
        "under_point_loads": [
            {
                "label": load.label,
                "x": load.x,
                "uy": _lookup_profile_value(
                    profile.x_values, profile.displacement_values, load.x
                ),
            }
            for load in model.point_forces
            if abs(load.fy) > EPSILON
        ],
    }

    is_cantilever = len(model.supports) == 1 and model.supports[0].type == "fixed"
    if is_cantilever:
        summary["free_end"] = {
            "x": model.length,
            "uy": _lookup_profile_value(
                profile.x_values, profile.displacement_values, model.length
            ),
            "theta": _lookup_profile_value(
                profile.x_values, profile.slope_values, model.length
            ),
        }

    is_simple_beam = (
        len(model.supports) == 2
        and len(supports_with_vertical_restraint) == 2
        and all(
            "Mz" not in _support_components(support.type) for support in model.supports
        )
    )
    if is_simple_beam:
        span_midpoint = 0.5 * (
            min(support.x for support in supports_with_vertical_restraint)
            + max(support.x for support in supports_with_vertical_restraint)
        )
        summary["midspan"] = {
            "x": span_midpoint,
            "uy": _lookup_profile_value(
                profile.x_values, profile.displacement_values, span_midpoint
            ),
            "theta": _lookup_profile_value(
                profile.x_values, profile.slope_values, span_midpoint
            ),
        }

    return summary


def _classify_report_case(model: BeamModel) -> dict[str, Any]:
    """Describe which built-in report template matches the solved beam."""
    supports_with_vertical_restraint = [
        support
        for support in model.supports
        if "Ry" in _support_components(support.type)
    ]

    is_cantilever = len(model.supports) == 1 and model.supports[0].type == "fixed"
    if is_cantilever:
        return {
            "case": "cantilever",
            "default_template": "template_report/cantilever_beam_report.txt",
            "supported": True,
        }

    is_simple_beam = (
        len(model.supports) == 2
        and len(supports_with_vertical_restraint) == 2
        and all(
            "Mz" not in _support_components(support.type) for support in model.supports
        )
    )
    if is_simple_beam:
        return {
            "case": "simply_supported",
            "default_template": "template_report/simply_supported_beam_report.txt",
            "supported": True,
        }

    return {
        "case": "generic",
        "default_template": None,
        "supported": False,
    }


def _infer_canonical_case_type(model: BeamModel) -> str:
    """Infer the canonical case label from the solved beam model."""
    if model.case_type is not None:
        return model.case_type

    if (
        len(model.supports) == 1
        and model.supports[0].type == "fixed"
        and not model.point_moments
        and not model.distributed_forces
        and not model.distributed_moments
        and len(model.point_forces) == 1
        and math.isclose(model.point_forces[0].x, model.length, rel_tol=0.0, abs_tol=EPSILON)
    ):
        return "cantilever_tip_load"

    if (
        len(model.supports) == 2
        and {support.type for support in model.supports} == {"pin", "roller_y"}
        and not model.point_moments
        and not model.distributed_forces
        and not model.distributed_moments
        and len(model.point_forces) == 1
        and math.isclose(
            model.point_forces[0].x, 0.5 * model.length, rel_tol=0.0, abs_tol=EPSILON
        )
    ):
        return "simply_supported_midspan_point_load"

    raise ValueError(
        "The solved model does not match the canonical interface cases. "
        "Supported cases are: cantilever_tip_load and simply_supported_midspan_point_load."
    )


def _canonical_load_direction(load: PointForce) -> str:
    if abs(load.fx) > EPSILON and abs(load.fy) > EPSILON:
        raise ValueError("Canonical output supports point loads in one axis only.")
    if abs(load.fx) > EPSILON:
        return "x_positive" if load.fx >= 0.0 else "x_negative"
    return "y_positive" if load.fy >= 0.0 else "y_negative"


def _canonical_load_result(model: BeamModel) -> dict[str, Any]:
    if len(model.point_forces) != 1:
        raise ValueError("Canonical output requires exactly one point_force load.")
    if model.point_moments or model.distributed_forces or model.distributed_moments:
        raise ValueError("Canonical output supports only a single point_force load.")

    load = model.point_forces[0]
    magnitude = abs(load.fx) if abs(load.fx) > EPSILON else abs(load.fy)
    return {
        "type": "point_force",
        "magnitude": magnitude,
        "direction": _canonical_load_direction(load),
        "x": load.x,
    }


def _canonical_section(section: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {
        key: float(section[key])
        for key in ("A", "Iy", "Iz")
        if key in section
    }
    # Optionally include width and height if available
    for key in ("width", "height"):
        if key in section:
            try:
                filtered[key] = float(section[key])
            except (TypeError, ValueError):
                pass
    if "A" not in filtered or "Iz" not in filtered:
        raise ValueError("Canonical output requires section.A and section.Iz.")
    return filtered


def _canonical_material(material: dict[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {
        key: float(material[key])
        for key in ("E", "nu", "rho")
        if key in material
    }
    # Optionally include material name if available
    if "name" in material:
        filtered["name"] = str(material["name"])
    if "E" not in filtered:
        raise ValueError("Canonical output requires material.E.")
    return filtered


def _canonical_reactions(reactions: dict[str, dict[str, float]], support_name: str) -> dict[str, float]:
    support_reactions = reactions.get(support_name, {})
    return {
        "Rx": support_reactions.get("Rx", 0.0),
        "Ry": support_reactions.get("Ry", 0.0),
        "Mz": support_reactions.get("Mz", 0.0),
    }


def _canonical_force_extreme(entry: dict[str, Any]) -> dict[str, float]:
    return {
        "value": float(entry["max_abs_signed_value"]),
        "x": float(entry["max_abs_x"]),
    }


def _canonical_profile_extreme(entry: dict[str, Any]) -> dict[str, float]:
    return {
        "value": float(entry["max_abs_signed_value"]),
        "x": float(entry["max_abs_x"]),
    }


def _response_point_candidates(model: BeamModel, case_type: str) -> list[tuple[str, float]]:
    candidates = [(support.name, support.x) for support in model.supports]
    if case_type == "cantilever_tip_load":
        candidates.append(("TIP", model.length))
    elif case_type == "simply_supported_midspan_point_load":
        candidates.append(("MIDSPAN", 0.5 * model.length))
    for load in model.point_forces:
        if load.label:
            candidates.append((load.label, load.x))
    for index, position in enumerate(model.response_positions, start=1):
        candidates.append((f"R{index}", position))
    return candidates


def _build_canonical_response_points(
    model: BeamModel,
    reactions: dict[str, dict[str, float]],
    profile: DeflectionProfile,
    case_type: str,
) -> list[dict[str, float | str]]:
    response_points: list[dict[str, float | str]] = []
    seen_positions: set[float] = set()
    for label, x in _response_point_candidates(model, case_type):
        rounded_x = round(x, 12)
        if rounded_x in seen_positions:
            continue
        seen_positions.add(rounded_x)
        actions = evaluate_internal_actions(model, reactions, x)
        response_points.append(
            {
                "label": label,
                "x": x,
                "N": actions["N"],
                "V": actions["V"],
                "M": actions["M"],
                "uy": _lookup_profile_value(
                    profile.x_values, profile.displacement_values, x
                ),
                "rz": _lookup_profile_value(profile.x_values, profile.slope_values, x),
            }
        )
    return response_points


def _build_canonical_stations(
    model: BeamModel,
    reactions: dict[str, dict[str, float]],
    profile: DeflectionProfile,
) -> list[dict[str, float]]:
    return [
        {
            "x": station["x"],
            "N": station["N"],
            "V": station["V"],
            "M": station["M"],
            "uy": station["uy"],
            "rz": station["theta"],
        }
        for station in build_station_results(model, reactions, profile)
    ]


def _build_summary_text(case_type: str, model: BeamModel, reactions: dict[str, dict[str, float]], extrema: dict[str, Any]) -> str:
    if case_type == "cantilever_tip_load":
        support = model.supports[0]
        support_reactions = _canonical_reactions(reactions, support.name)
        return (
            f"Cantilever beam, L={model.length}, tip point load analyzed. "
            f"Ry={support_reactions['Ry']}, Mz={support_reactions['Mz']}, "
            f"|M|max={abs(extrema['bending_moment']['value'])}."
        )
    support_a = _canonical_reactions(reactions, model.supports[0].name)
    support_b = _canonical_reactions(reactions, model.supports[1].name)
    return (
        f"Simply supported beam, L={model.length}, midspan point load analyzed. "
        f"RA={support_a['Ry']}, RB={support_b['Ry']}, "
        f"|M|max={abs(extrema['bending_moment']['value'])}."
    )


def compute_extrema(
    model: BeamModel, reactions: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Find the extreme axial force, shear force, and moment values along the beam."""
    positions = _event_positions(model)
    events = _build_event_actions(model, reactions)
    current_state = {"N": 0.0, "V": 0.0, "M": 0.0}
    candidates: list[dict[str, float | str]] = []

    for index, x in enumerate(positions):
        if index > 0:
            candidates.append({"x": x, "side": "left", **current_state})

        action = events[x]
        current_state = {
            "N": current_state["N"] - action["fx"],
            "V": current_state["V"] - action["fy"],
            "M": current_state["M"] - action["mz"],
        }

        if x < model.length - EPSILON:
            candidates.append({"x": x, "side": "right", **current_state})

        if index == len(positions) - 1:
            continue

        next_x = positions[index + 1]
        interval_length = next_x - x
        if interval_length <= EPSILON:
            continue

        coeffs = _interval_load_coefficients(model, x, next_x)
        qx0 = coeffs["ax"] + coeffs["bx"] * x
        qy0 = coeffs["ay"] + coeffs["by"] * x
        mz0 = coeffs["am"] + coeffs["bm"] * x

        for root in _solve_linear_root(qx0, coeffs["bx"], interval_length):
            state = _evaluate_interval_state(current_state, coeffs, x, root)
            candidates.append({"x": x + root, "side": "inside", **state})

        for root in _solve_linear_root(qy0, coeffs["by"], interval_length):
            state = _evaluate_interval_state(current_state, coeffs, x, root)
            candidates.append({"x": x + root, "side": "inside", **state})

        for root in _solve_quadratic_roots(
            0.5 * coeffs["by"],
            qy0 - coeffs["bm"],
            -current_state["V"] - mz0,
            interval_length,
        ):
            state = _evaluate_interval_state(current_state, coeffs, x, root)
            candidates.append({"x": x + root, "side": "inside", **state})

        current_state = _evaluate_interval_state(
            current_state, coeffs, x, interval_length
        )

    if not candidates:
        raise ValueError("No internal force candidates were generated.")

    return {
        "axial_force": _build_extrema_entry(candidates, "N"),
        "shear_force": _build_extrema_entry(candidates, "V"),
        "bending_moment": _build_extrema_entry(candidates, "M"),
    }


def build_station_results(
    model: BeamModel,
    reactions: dict[str, dict[str, float]],
    profile: DeflectionProfile | None,
) -> list[dict[str, float]]:
    """Build the evenly spaced station table returned to the user."""
    station_count = model.stations
    return [
        (
            {
                "x": x,
                **evaluate_internal_actions(model, reactions, x),
                "theta": _lookup_profile_value(
                    profile.x_values, profile.slope_values, x
                ),
                "uy": _lookup_profile_value(
                    profile.x_values, profile.displacement_values, x
                ),
            }
            if profile is not None
            else {
                "x": x,
                **evaluate_internal_actions(model, reactions, x),
            }
        )
        for x in [
            model.length * index / (station_count - 1) for index in range(station_count)
        ]
    ]


def analyze_beam(model: BeamModel) -> dict[str, Any]:
    """Run the full beam analysis and assemble the canonical result payload."""
    reactions = solve_reactions(model)
    profile = None
    try:
        profile = compute_deflection_profile(model, reactions)
    except ValueError as exc:
        raise ValueError(f"Canonical output requires displacement results: {exc}") from exc

    case_type = _infer_canonical_case_type(model)
    force_extrema = compute_extrema(model, reactions)
    displacement_summary = _build_displacement_summary(model, profile)
    extrema = {
        "axial_force": _canonical_force_extreme(force_extrema["axial_force"]),
        "shear_force": _canonical_force_extreme(force_extrema["shear_force"]),
        "bending_moment": _canonical_force_extreme(force_extrema["bending_moment"]),
        "displacement_y": _canonical_profile_extreme(
            displacement_summary["extrema"]["uy"]
        ),
        "rotation_z": _canonical_profile_extreme(
            displacement_summary["extrema"]["theta"]
        ),
    }

    return {
        **({"model_id": model.model_id} if model.model_id is not None else {}),
        "case_type": case_type,
        "backend": {
            "name": BACKEND_NAME,
            "version": BACKEND_VERSION,
        },
        "beam": {
            "length": model.length,
            **(
                {"num_elements": model.num_elements}
                if model.num_elements is not None
                else {}
            ),
        },
        "section": _canonical_section(model.section),
        "material": _canonical_material(model.material),
        "load": _canonical_load_result(model),
        "supports": [
            {
                "name": support.name,
                "type": support.type,
                "x": support.x,
                "reactions": _canonical_reactions(reactions, support.name),
            }
            for support in model.supports
        ],
        "extrema": extrema,
        "response_points": _build_canonical_response_points(
            model, reactions, profile, case_type
        ),
        "stations": _build_canonical_stations(model, reactions, profile),
        "summary_text": _build_summary_text(case_type, model, reactions, extrema),
        "warnings": [],
        "solver_scope": {
            "method": "2D statically determinate beam analysis by equilibrium",
            "supported_reaction_unknowns": 3,
            "note": "Indeterminate support layouts are intentionally rejected in this hand-calculation solver.",
        },
    }


def analyze_beam_file(path: str | Path) -> dict[str, Any]:
    """Analyze one JSON input file."""
    model = load_beam_model(path)
    return analyze_beam(model)


def load_beam_source(path: str | Path) -> BeamModel:
    """Load either a JSON file or a Code_Aster input folder."""
    source_path = Path(path)
    if source_path.is_dir():
        return load_code_aster_model(source_path)
    if source_path.suffix.lower() == ".json":
        return load_beam_model(source_path)
    raise ValueError(
        "Unsupported input source. Provide a JSON file or a folder containing "
        "model.comm, model.export, and beam.geo."
    )


def analyze_beam_source(path: str | Path) -> dict[str, Any]:
    """Analyze either supported source type through one common entry point."""
    model = load_beam_source(path)
    return analyze_beam(model)
