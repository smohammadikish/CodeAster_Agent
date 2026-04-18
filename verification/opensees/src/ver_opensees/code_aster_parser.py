from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ver_opensees.models import (BeamInput, BeamMaterial, BeamSection,
                                 InputSource, PointLoad)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"

_REQUIRED_COMPANION_SUFFIXES = (".comm", ".geo", ".export")
_OPTIONAL_COMPANION_SUFFIXES = (".med",)
N_PER_MM2_TO_PA = 1_000_000.0


@dataclass(slots=True)
class _GeoModel:
    length: float
    num_elements: int
    point_to_node: dict[int, int]
    physical_points: dict[str, list[int]]


def load_code_aster_input(
    path: Path, opensees_executable: str
) -> tuple[BeamInput, InputSource]:
    input_dir, companion_files = _resolve_code_aster_files(path)
    geo_model = _parse_geo_file(companion_files[".geo"])
    comm_data = _parse_comm_file(companion_files[".comm"])
    export_data = _parse_export_file(companion_files[".export"])

    case_type = _infer_case_type(comm_data["constraints"], geo_model)
    load_group = _extract_single_group_name(comm_data["load"]["groups"], "load")
    load_point = _extract_single_point_id(geo_model.physical_points, load_group, "load")
    load_node = geo_model.point_to_node[load_point]
    length_m = geo_model.length
    load_direction, load_magnitude = _extract_point_force(comm_data["load"]["components"])

    # Geometry coordinates are already expressed in model units from the .geo file.
    # Only section dimensions from AFFE_CARA_ELEM need conversion from mm-based inputs.
    beam_input = BeamInput(
        analysis_name=export_data.get("nomjob", input_dir.name),
        case_type=case_type,
        opensees_executable=opensees_executable,
        output_root=DEFAULT_WORKSPACE,
        model_id=export_data.get("nomjob"),
        units={},
        length=length_m,
        num_elements=geo_model.num_elements,
        material=BeamMaterial(
            elastic_modulus=comm_data["material"]["E"],
        ),
        section=BeamSection(
            area=comm_data["section"]["A"] / 1_000_000.0,
            inertia_z=comm_data["section"]["IZ"] / 1_000_000_000_000.0,
        ),
        load=PointLoad(
            load_type="point_force",
            direction=load_direction,
            magnitude=load_magnitude,
            x=(load_node - 1) * (length_m / geo_model.num_elements),
            node=load_node,
        ),
        stations=201,
        response_positions=((load_node - 1) * (length_m / geo_model.num_elements),),
        meta={},
    )

    source = InputSource(
        source_type="code_aster",
        root_path=input_dir,
        snapshot_files=tuple(
            companion_files[suffix]
            for suffix in (*_REQUIRED_COMPANION_SUFFIXES, *_OPTIONAL_COMPANION_SUFFIXES)
            if suffix in companion_files
        ),
    )
    return beam_input, source


def _resolve_code_aster_files(path: Path) -> tuple[Path, dict[str, Path]]:
    input_dir = path if path.is_dir() else path.parent
    companion_files: dict[str, Path] = {}

    for suffix in _REQUIRED_COMPANION_SUFFIXES:
        matches = sorted(input_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '{suffix}' file in '{input_dir}', found {len(matches)}."
            )
        companion_files[suffix] = matches[0]

    for suffix in _OPTIONAL_COMPANION_SUFFIXES:
        matches = sorted(input_dir.glob(f"*{suffix}"))
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Expected at most one '{suffix}' file in '{input_dir}', found {len(matches)}."
            )
        if matches:
            companion_files[suffix] = matches[0]

    return input_dir, companion_files


def _parse_geo_file(path: Path) -> _GeoModel:
    text = path.read_text(encoding="utf-8")
    geometry_in_mm = "(mm)" in text
    points = {
        int(match.group(1)): (
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )
        for match in re.finditer(
            r"Point\((\d+)\)\s*=\s*\{\s*([^,]+),\s*([^,]+),\s*([^,]+)\s*\};",
            text,
        )
    }
    lines = {
        int(match.group(1)): (int(match.group(2)), int(match.group(3)))
        for match in re.finditer(r"Line\((\d+)\)\s*=\s*\{\s*(\d+),\s*(\d+)\s*\};", text)
    }
    transfinite_nodes = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(
            r"Transfinite Line\s*\{\s*(\d+)\s*\}\s*=\s*(\d+)\s*;",
            text,
        )
    }
    physical_points = {
        match.group(1): [
            int(value.strip()) for value in match.group(2).split(",") if value.strip()
        ]
        for match in re.finditer(
            r'Physical Point\("([^"]+)"\)\s*=\s*\{([^}]+)\};',
            text,
        )
    }
    physical_curves = {
        match.group(1): [
            int(value.strip()) for value in match.group(2).split(",") if value.strip()
        ]
        for match in re.finditer(
            r'Physical Curve\("([^"]+)"\)\s*=\s*\{([^}]+)\};',
            text,
        )
    }

    beam_lines = physical_curves.get("BEAM")
    if not beam_lines:
        raise ValueError(f"Could not find Physical Curve('BEAM') in '{path}'.")

    ordered_lines = _order_beam_lines(beam_lines, lines, points)
    current_node = 1
    total_length = 0.0
    point_to_node: dict[int, int] = {}

    first_start, _ = ordered_lines[0]
    point_to_node[first_start] = current_node

    for start_point, line_id in ordered_lines:
        line_start, line_end = lines[line_id]
        next_point = line_end if line_start == start_point else line_start
        total_length += _distance(points[start_point], points[next_point])

        nodes_on_line = transfinite_nodes.get(line_id)
        if nodes_on_line is None or nodes_on_line < 2:
            raise ValueError(
                f"Line {line_id} in '{path}' must have a Transfinite Line definition with at least 2 nodes."
            )

        current_node += nodes_on_line - 1
        point_to_node[next_point] = current_node

    if geometry_in_mm:
        total_length /= 1000.0

    return _GeoModel(
        length=total_length,
        num_elements=current_node - 1,
        point_to_node=point_to_node,
        physical_points=physical_points,
    )


def _order_beam_lines(
    beam_lines: list[int],
    lines: dict[int, tuple[int, int]],
    points: dict[int, tuple[float, float, float]],
) -> list[tuple[int, int]]:
    adjacency: dict[int, list[int]] = {}
    for line_id in beam_lines:
        start, end = lines[line_id]
        adjacency.setdefault(start, []).append(line_id)
        adjacency.setdefault(end, []).append(line_id)

    endpoints = [
        point_id for point_id, connected in adjacency.items() if len(connected) == 1
    ]
    if len(endpoints) != 2:
        raise ValueError(
            "The BEAM geometry must form one continuous chain with two endpoints."
        )

    start_point = min(endpoints, key=lambda point_id: points[point_id][0])
    ordered: list[tuple[int, int]] = []
    visited_lines: set[int] = set()
    current_point = start_point

    while len(visited_lines) < len(beam_lines):
        next_line = next(
            (
                line_id
                for line_id in adjacency[current_point]
                if line_id not in visited_lines
            ),
            None,
        )
        if next_line is None:
            raise ValueError("The BEAM geometry contains a disconnected segment.")

        ordered.append((current_point, next_line))
        visited_lines.add(next_line)

        line_start, line_end = lines[next_line]
        current_point = line_end if line_start == current_point else line_start

    return ordered


def _distance(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> float:
    return math.dist(start, end)


def _parse_comm_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    material_block = _extract_block(text, "DEFI_MATERIAU")
    section_block = _extract_block(text, "AFFE_CARA_ELEM")
    constraints_block = _extract_block(text, "AFFE_CHAR_MECA", "DDL_IMPO")
    load_block = _extract_block(text, "AFFE_CHAR_MECA", "FORCE_NODALE")

    section_names = _parse_tuple_literal(
        _search(section_block, r"CARA\s*=\s*(\([^)]*\))", "CARA")
    )
    section_values = _parse_tuple_literal(
        _search(section_block, r"VALE\s*=\s*(\([^)]*\))", "VALE")
    )
    section = {
        str(name).strip("'\""): float(value)
        for name, value in zip(section_names, section_values, strict=True)
    }
    if "A" not in section and {"HY", "HZ"}.issubset(section):
        hy = section["HY"]
        hz = section["HZ"]
        section["A"] = hy * hz
        section["IZ"] = hz * hy**3 / 12.0

    constraints = []
    for factor in _extract_factor_blocks(constraints_block, "DDL_IMPO"):
        groups = _parse_tuple_literal(
            _search(factor, r"GROUP_NO\s*=\s*(\([^)]*\))", "GROUP_NO")
        )
        constraints.append(
            {
                "groups": [str(group) for group in groups],
                "components": _parse_component_assignments(
                    factor, ("DX", "DY", "DZ", "DRX", "DRY", "DRZ")
                ),
            }
        )

    load_factor = _extract_single_factor_block(load_block, "FORCE_NODALE")
    elastic_modulus = float(
        _search(material_block, r"E\s*=\s*([0-9.eE+-]+)", "E")
    )
    if elastic_modulus < 1e7:
        elastic_modulus *= N_PER_MM2_TO_PA

    return {
        "material": {
            "E": elastic_modulus,
        },
        "section": section,
        "constraints": constraints,
        "load": {
            "groups": [
                str(group)
                for group in _parse_tuple_literal(
                    _search(load_factor, r"GROUP_NO\s*=\s*(\([^)]*\))", "GROUP_NO")
                )
            ],
            "components": _parse_component_assignments(
                load_factor, ("FX", "FY", "FZ", "MX", "MY", "MZ")
            ),
        },
    }


def _parse_export_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "P":
            data[parts[1]] = parts[2]
    return data


def _extract_block(
    text: str, concept_name: str, required_keyword: str | None = None
) -> str:
    start = 0
    while True:
        index = text.find(f"{concept_name}(", start)
        if index == -1:
            target = required_keyword or concept_name
            raise ValueError(f"Could not find '{target}' in Code_Aster input.")

        block = _slice_balanced_parentheses(text, index + len(concept_name))
        if required_keyword is None or required_keyword in block:
            return block
        start = index + len(concept_name)


def _slice_balanced_parentheses(text: str, open_paren_index: int) -> str:
    if text[open_paren_index] != "(":
        raise ValueError("Expected '(' while parsing Code_Aster input.")

    depth = 0
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index + 1 : index]

    raise ValueError("Could not find matching ')' while parsing Code_Aster input.")


def _extract_factor_blocks(block: str, keyword: str) -> list[str]:
    factors: list[str] = []
    start = 0
    token = f"{keyword}=_F("
    while True:
        index = block.find(token, start)
        if index == -1:
            break
        open_index = index + len(keyword) + 3
        factors.append(_slice_balanced_parentheses(block, open_index))
        start = open_index + 1
    if factors:
        return factors

    tuple_token = f"{keyword}=("
    tuple_index = block.find(tuple_token)
    if tuple_index == -1:
        raise ValueError(f"Could not find any '{keyword}=_F(...)' blocks.")

    tuple_open_index = tuple_index + len(keyword) + 1
    tuple_content = _slice_balanced_parentheses(block, tuple_open_index)
    search_start = 0
    while True:
        factor_index = tuple_content.find("_F(", search_start)
        if factor_index == -1:
            break
        factors.append(_slice_balanced_parentheses(tuple_content, factor_index + 2))
        search_start = factor_index + 3
    if not factors:
        raise ValueError(f"Could not find any '{keyword}=_F(...)' blocks.")
    return factors


def _extract_single_factor_block(block: str, keyword: str) -> str:
    factors = _extract_factor_blocks(block, keyword)
    if len(factors) != 1:
        raise ValueError(
            f"Expected exactly one '{keyword}=_F(...)' block, found {len(factors)}."
        )
    return factors[0]


def _parse_tuple_literal(text: str) -> tuple[object, ...]:
    value = ast.literal_eval(text)
    if isinstance(value, tuple):
        return value
    return (value,)


def _parse_component_assignments(text: str, names: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in names:
        match = re.search(rf"\b{name}\s*=\s*([0-9.eE+-]+)", text)
        if match:
            values[name] = float(match.group(1))
    return values


def _search(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find '{label}' while parsing Code_Aster input.")
    return match.group(1)


def _extract_single_group_name(groups: list[str], label: str) -> str:
    if len(groups) != 1:
        raise ValueError(
            f"Expected exactly one group for '{label}', found {len(groups)}."
        )
    return groups[0]


def _extract_single_point_id(
    physical_points: dict[str, list[int]],
    group_name: str,
    label: str,
) -> int:
    point_ids = physical_points.get(group_name)
    if not point_ids:
        raise ValueError(
            f"Could not find a physical point group named '{group_name}' for '{label}'."
        )
    if len(point_ids) != 1:
        raise ValueError(
            f"Expected exactly one point in physical group '{group_name}', found {len(point_ids)}."
        )
    return point_ids[0]


def _infer_case_type(
    constraints: list[dict[str, object]], geo_model: _GeoModel
) -> str:
    support_nodes: list[tuple[int, dict[str, float]]] = []
    for constraint in constraints:
        group_name = _extract_single_group_name(constraint["groups"], "support")
        point_id = _extract_single_point_id(
            geo_model.physical_points, group_name, "support"
        )
        support_nodes.append((geo_model.point_to_node[point_id], constraint["components"]))  # type: ignore[arg-type]

    support_nodes.sort(key=lambda item: item[0])
    if len(support_nodes) == 1:
        node_id, components = support_nodes[0]
        if node_id == 1 and _is_fixed_support(components):
            return "cantilever_tip_load"
        raise ValueError(
            "Unsupported single-support configuration in Code_Aster input."
        )

    if len(support_nodes) == 2:
        first_node, first_components = support_nodes[0]
        second_node, second_components = support_nodes[1]
        if (
            first_node == 1
            and second_node == geo_model.num_elements + 1
            and _is_pinned_support(first_components)
            and _is_roller_support(second_components)
        ):
            return "simply_supported_midspan_point_load"
        raise ValueError("Unsupported two-support configuration in Code_Aster input.")

    raise ValueError(
        "Only cantilever and simply supported beam configurations are supported."
    )


def _is_fixed_support(components: dict[str, float]) -> bool:
    return {"DX", "DY", "DRZ"}.issubset(components)


def _is_pinned_support(components: dict[str, float]) -> bool:
    return {"DX", "DY"}.issubset(components) and "DRZ" not in components


def _is_roller_support(components: dict[str, float]) -> bool:
    return set(components).issubset({"DY", "DZ"}) and "DY" in components and "DX" not in components and "DRZ" not in components


def _extract_point_force(components: dict[str, float]) -> tuple[str, float]:
    active_components = [
        (name, value)
        for name, value in components.items()
        if name in {"FX", "FY"} and abs(value) > 0.0
    ]
    if len(active_components) != 1:
        raise ValueError(
            "Code_Aster input must define exactly one non-zero nodal force component among FX and FY."
        )

    component, value = active_components[0]
    if component == "FX":
        return ("x_positive" if value >= 0.0 else "x_negative", abs(value))
    return ("y_positive" if value >= 0.0 else "y_negative", abs(value))
