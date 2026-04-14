from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ver_opensees.models import BeamInput


def write_results_json(beam_input: BeamInput, working_dir: Path, target: Path) -> None:
    raw_results = _read_raw_results(beam_input, working_dir)
    summary = _build_canonical_output(beam_input, raw_results)
    report_text = _render_report(beam_input, summary)
    summary["summary_text"] = report_text

    payload = {
        **raw_results,
        "derived_summary": summary,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (working_dir / "summary_results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (working_dir / "report.txt").write_text(report_text, encoding="utf-8")


def _read_raw_results(beam_input: BeamInput, working_dir: Path) -> dict[str, Any]:
    raw_node_displacements = _parse_table(working_dir / "node_displacements.out")
    raw_element_forces = _parse_table(working_dir / "element_forces.out")
    node_displacements = _normalize_recorder_output(
        raw_node_displacements,
        beam_input.num_nodes,
        3,
        "node_displacements.out",
    )
    element_forces = _normalize_recorder_output(
        raw_element_forces,
        beam_input.num_elements,
        6,
        "element_forces.out",
    )

    return {
        "summary": _parse_key_value_file(working_dir / "summary.txt"),
        "node_displacements": node_displacements,
        "node_displacements_description": {
            "format": "one item per node: [ux, uy, rz]",
            "columns": ["ux", "uy", "rz"],
            "node_order": list(range(1, beam_input.num_nodes + 1)),
        },
        "element_forces": element_forces,
        "element_forces_description": {
            "format": "one item per element: [axial_i, shear_i, moment_i, axial_j, shear_j, moment_j]",
            "columns": [
                "axial_i",
                "shear_i",
                "moment_i",
                "axial_j",
                "shear_j",
                "moment_j",
            ],
            "element_order": list(range(1, beam_input.num_elements + 1)),
        },
        "support_reactions": _parse_table(working_dir / "support_reactions.out"),
        "support_reactions_description": {
            "format": "one item per support node: [node, Rx, Ry, Mz]",
            "columns": ["node", "Rx", "Ry", "Mz"],
        },
    }


def _build_canonical_output(
    beam_input: BeamInput, raw_results: dict[str, Any]
) -> dict[str, Any]:
    node_points = _build_node_points(beam_input, raw_results["node_displacements"])
    element_profiles = _build_element_profiles(beam_input, raw_results["element_forces"])
    supports = _build_supports(beam_input, raw_results["support_reactions"])
    response_points = _build_response_points(beam_input, node_points, element_profiles)
    stations = _build_stations(beam_input, node_points, element_profiles)

    payload: dict[str, Any] = {
        "case_type": beam_input.case_type,
        "backend": {
            "name": "OpenSees",
        },
        "beam": {
            "length": beam_input.length,
            "num_elements": beam_input.num_elements,
        },
        "section": {
            "A": beam_input.section.area,
            "Iz": beam_input.section.inertia_z,
        },
        "material": {
            "E": beam_input.material.elastic_modulus,
        },
        "load": {
            "type": beam_input.load.load_type,
            "magnitude": beam_input.load.magnitude,
            "direction": beam_input.load.direction,
            "x": beam_input.load.x,
        },
        "supports": supports,
        "extrema": {
            "axial_force": _extreme_point(stations, "N"),
            "shear_force": _extreme_point(stations, "V"),
            "bending_moment": _extreme_point(stations, "M"),
            "displacement_y": _extreme_point(stations, "uy"),
            "rotation_z": _extreme_point(stations, "rz"),
        },
        "response_points": response_points,
        "stations": stations,
    }
    if beam_input.model_id is not None:
        payload["model_id"] = beam_input.model_id
    if beam_input.section.inertia_y is not None:
        payload["section"]["Iy"] = beam_input.section.inertia_y
    if beam_input.material.poisson_ratio is not None:
        payload["material"]["nu"] = beam_input.material.poisson_ratio
    if beam_input.material.density is not None:
        payload["material"]["rho"] = beam_input.material.density

    warnings = _build_warnings(beam_input)
    if warnings:
        payload["warnings"] = warnings
    return payload


def _build_node_points(
    beam_input: BeamInput, node_displacements: list[list[float]]
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    for node_id, row in enumerate(node_displacements, start=1):
        ux, uy, rz = _pad_row(row, 3)
        points.append(
            {
                "node": node_id,
                "x": _node_x(beam_input, node_id),
                "ux": ux,
                "uy": uy,
                "rz": rz,
            }
        )
    return points


def _build_element_profiles(
    beam_input: BeamInput, element_forces: list[list[float]]
) -> list[dict[str, float | int]]:
    dx = beam_input.length / beam_input.num_elements
    profiles: list[dict[str, float | int]] = []
    for element_id, row in enumerate(element_forces, start=1):
        axial_i, shear_i, moment_i, axial_j, shear_j, moment_j = _pad_row(row, 6)
        profiles.append(
            {
                "element": element_id,
                "x0": (element_id - 1) * dx,
                "x1": element_id * dx,
                "N0": axial_i,
                "N1": -axial_j,
                "V0": shear_i,
                "V1": -shear_j,
                "M0": moment_i,
                "M1": -moment_j,
            }
        )
    return profiles


def _build_supports(
    beam_input: BeamInput, support_reactions: list[list[float]]
) -> list[dict[str, Any]]:
    reactions_by_node = {
        int(row[0]): _pad_row(row[1:], 3) for row in support_reactions if row
    }

    if beam_input.support_type == "cantilever":
        specs = [("A", "fixed", 1, 0.0)]
    else:
        specs = [
            ("A", "pin", 1, 0.0),
            ("B", "roller_y", beam_input.num_nodes, beam_input.length),
        ]

    supports: list[dict[str, Any]] = []
    for name, support_type, node_id, x in specs:
        rx, ry, mz = reactions_by_node.get(node_id, [0.0, 0.0, 0.0])
        supports.append(
            {
                "name": name,
                "type": support_type,
                "x": x,
                "reactions": {
                    "Rx": rx,
                    "Ry": ry,
                    "Mz": mz,
                },
            }
        )
    return supports


def _build_response_points(
    beam_input: BeamInput,
    node_points: list[dict[str, float | int]],
    element_profiles: list[dict[str, float | int]],
) -> list[dict[str, float | str]]:
    response_points: list[dict[str, float | str]] = []
    for index, x in enumerate(beam_input.response_positions, start=1):
        section = _section_response_at_x(x, element_profiles)
        disp = _displacement_response_at_x(x, node_points)
        response_points.append(
            {
                "label": _response_label(index, x, beam_input.load.x),
                "x": x,
                "N": section["N"],
                "V": section["V"],
                "M": section["M"],
                "uy": disp["uy"],
                "rz": disp["rz"],
            }
        )
    return response_points


def _build_stations(
    beam_input: BeamInput,
    node_points: list[dict[str, float | int]],
    element_profiles: list[dict[str, float | int]],
) -> list[dict[str, float]]:
    if beam_input.stations <= 1:
        return []

    positions = [
        beam_input.length * index / (beam_input.stations - 1)
        for index in range(beam_input.stations)
    ]

    stations: list[dict[str, float]] = []
    for x in positions:
        section = _section_response_at_x(x, element_profiles)
        disp = _displacement_response_at_x(x, node_points)
        stations.append(
            {
                "x": x,
                "N": section["N"],
                "V": section["V"],
                "M": section["M"],
                "uy": disp["uy"],
                "rz": disp["rz"],
            }
        )
    return stations


def _section_response_at_x(
    x: float, element_profiles: list[dict[str, float | int]]
) -> dict[str, float]:
    if not element_profiles:
        return {"N": 0.0, "V": 0.0, "M": 0.0}

    for profile in element_profiles:
        x0 = float(profile["x0"])
        x1 = float(profile["x1"])
        if x <= x1 + 1.0e-9:
            return {
                "N": _lerp(float(profile["N0"]), float(profile["N1"]), x, x0, x1),
                "V": _lerp(float(profile["V0"]), float(profile["V1"]), x, x0, x1),
                "M": _lerp(float(profile["M0"]), float(profile["M1"]), x, x0, x1),
            }

    last = element_profiles[-1]
    return {
        "N": float(last["N1"]),
        "V": float(last["V1"]),
        "M": float(last["M1"]),
    }


def _displacement_response_at_x(
    x: float, node_points: list[dict[str, float | int]]
) -> dict[str, float]:
    if not node_points:
        return {"uy": 0.0, "rz": 0.0}

    for left, right in zip(node_points, node_points[1:], strict=False):
        x0 = float(left["x"])
        x1 = float(right["x"])
        if x <= x1 + 1.0e-9:
            return {
                "uy": _lerp(float(left["uy"]), float(right["uy"]), x, x0, x1),
                "rz": _lerp(float(left["rz"]), float(right["rz"]), x, x0, x1),
            }

    last = node_points[-1]
    return {
        "uy": float(last["uy"]),
        "rz": float(last["rz"]),
    }


def _extreme_point(
    items: list[dict[str, float]], key: str
) -> dict[str, float]:
    target = max(items, key=lambda item: abs(float(item[key])))
    return {
        "value": float(target[key]),
        "x": float(target["x"]),
    }


def _build_warnings(beam_input: BeamInput) -> list[str]:
    if beam_input.case_type == "simply_supported_midspan_point_load":
        return [
            "At concentrated-load discontinuities, section forces in response_points and stations are reported from the left side of the load location.",
        ]
    return []


def _render_report(beam_input: BeamInput, summary: dict[str, Any]) -> str:
    template_dir = Path(__file__).resolve().parents[2] / "template_report"
    template_name = (
        "cantilever_beam_report.txt"
        if beam_input.support_type == "cantilever"
        else "simply_supported_beam_report.txt"
    )
    template_text = (template_dir / template_name).read_text(encoding="utf-8")

    support_a = next(
        (support for support in summary["supports"] if support["name"] == "A"),
        None,
    )
    support_b = next(
        (support for support in summary["supports"] if support["name"] == "B"),
        None,
    )
    context = {
        **summary,
        "support_A": support_a,
        "support_B": support_b,
        "primary_response_point": summary["response_points"][0],
    }
    return _render_template(template_text, context)


def _render_template(template_text: str, context: dict[str, Any]) -> str:
    pattern = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        value = _resolve_template_value(context, expression)
        return _stringify_value(value)

    return pattern.sub(replace, template_text)


def _resolve_template_value(context: dict[str, Any], expression: str) -> Any:
    current: Any = context
    for part in expression.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _parse_key_value_file(path: Path) -> dict[str, int | float | str]:
    results: dict[str, int | float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        results[key] = _parse_scalar_value(value)
    return results


def _parse_table(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([float(value) for value in stripped.split()])
    return rows


def _normalize_recorder_output(
    rows: list[list[float]],
    item_count: int,
    item_width: int,
    label: str,
) -> list[list[float]]:
    if not rows:
        return []

    if len(rows) == item_count and all(len(row) == item_width for row in rows):
        return rows

    if len(rows) == 1:
        flat = rows[0]
        expected = item_count * item_width
        if len(flat) != expected:
            raise ValueError(
                f"{label} has {len(flat)} values, expected {expected} "
                f"for {item_count} items with width {item_width}."
            )
        return [
            flat[index : index + item_width]
            for index in range(0, len(flat), item_width)
        ]

    raise ValueError(
        f"Could not interpret {label}. "
        f"Expected either {item_count} rows of width {item_width} or one flat row."
    )


def _parse_scalar_value(value: str) -> int | float | str:
    try:
        number = float(value)
    except ValueError:
        return value

    if number.is_integer():
        return int(number)

    return number


def _node_x(beam_input: BeamInput, node_id: int) -> float:
    return (node_id - 1) * (beam_input.length / beam_input.num_elements)


def _pad_row(row: list[float], size: int) -> list[float]:
    return [*row[:size], *([0.0] * max(0, size - len(row)))]


def _lerp(start: float, end: float, x: float, x0: float, x1: float) -> float:
    if abs(x1 - x0) <= 1.0e-12:
        return start
    ratio = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return start + (end - start) * ratio


def _response_label(index: int, x: float, load_x: float) -> str:
    if abs(x - load_x) <= 1.0e-9:
        return "load_point"
    return f"response_{index}"


def _stringify_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)
