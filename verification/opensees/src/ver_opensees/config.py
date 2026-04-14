from __future__ import annotations

import json
import os
from pathlib import Path

from ver_opensees.code_aster_parser import load_code_aster_input
from ver_opensees.models import (
    BeamInput,
    BeamMaterial,
    BeamSection,
    InputSource,
    PointLoad,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"
OPENSEES_ENV_KEYS = ("OPENSEES_EXECUTABLE", "opensees", "OPENSEES")


# Read the input source and convert it into typed Python objects.
def load_beam_input(path: Path) -> tuple[BeamInput, InputSource]:
    input_path = resolve_input_source_path(path)
    _load_env_files(input_path)

    if input_path.is_dir():
        beam_input, source = load_code_aster_input(
            input_path,
            _resolve_opensees_executable({}),
        )
        _validate(beam_input)
        return beam_input, source

    if input_path.suffix.lower() == ".json":
        return _load_json_input(input_path)

    beam_input, source = load_code_aster_input(
        input_path,
        _resolve_opensees_executable({}),
    )
    _validate(beam_input)
    return beam_input, source


def resolve_input_source_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved

    if resolved.suffix.lower() in {".json", ".comm", ".geo", ".export", ".med"}:
        return resolved

    raise ValueError(
        "Input must be a JSON file, a Code_Aster companion file, or a directory containing the Code_Aster input set."
    )


def _load_json_input(path: Path) -> tuple[BeamInput, InputSource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    beam = _require_object(raw, "beam")
    material = _require_object(raw, "material")
    section = _require_object(raw, "section")
    load = _require_object(raw, "load")
    units = raw.get("units", {})
    analysis = raw.get("analysis", {})

    case_type = str(raw["case_type"]).strip()
    num_elements = int(beam.get("num_elements", 20))
    length = float(beam["length"])
    load_x = _case_load_x(case_type, length)
    load_node = _case_load_node(case_type, num_elements)

    beam_input = BeamInput(
        analysis_name=str(raw.get("model_id", path.stem)),
        case_type=case_type,
        opensees_executable=_resolve_opensees_executable(raw),
        output_root=DEFAULT_WORKSPACE,
        model_id=_optional_string(raw.get("model_id")),
        units=_normalize_units(units),
        length=length,
        num_elements=num_elements,
        material=BeamMaterial(
            elastic_modulus=float(material["E"]),
            poisson_ratio=_optional_float(material.get("nu")),
            density=_optional_float(material.get("rho")),
        ),
        section=BeamSection(
            area=float(section["A"]),
            inertia_z=float(section["Iz"]),
            inertia_y=_optional_float(section.get("Iy")),
        ),
        load=PointLoad(
            load_type=str(load["type"]),
            direction=str(load["direction"]),
            magnitude=float(load["magnitude"]),
            x=load_x,
            node=load_node,
        ),
        stations=int(analysis.get("stations", 201)),
        response_positions=_response_positions(analysis, load_x),
        meta=dict(raw.get("meta", {})),
    )

    _validate(beam_input)
    return beam_input, InputSource(
        source_type="json",
        root_path=path,
        snapshot_files=(path,),
    )


# Load repository-wide defaults first, then allow legacy input-adjacent .env files
# to fill in only still-missing variables.
def _load_env_files(input_path: Path) -> None:
    candidate_paths = [REPO_ROOT / ".env"]
    legacy_path = (
        input_path.parent.parent / ".env"
        if input_path.is_file()
        else input_path.parent / ".env"
    )
    if legacy_path not in candidate_paths:
        candidate_paths.append(legacy_path)

    for candidate in candidate_paths:
        _load_dotenv(candidate)


# Read a simple .env file and store values in the process environment.
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

    _apply_env_aliases()


def _apply_env_aliases() -> None:
    canonical_value = next(
        (
            os.environ.get(env_key)
            for env_key in OPENSEES_ENV_KEYS
            if os.environ.get(env_key)
        ),
        None,
    )
    if canonical_value:
        normalized = os.path.expandvars(os.path.expanduser(canonical_value.strip()))
        os.environ.setdefault("OPENSEES_EXECUTABLE", normalized)
        os.environ.setdefault("opensees", normalized)
        os.environ.setdefault("OPENSEES", normalized)


# Choose the OpenSees executable from JSON first, then from environment variables.
def _resolve_opensees_executable(raw: dict) -> str:
    json_value = raw.get("opensees_executable")
    if json_value and json_value != "OpenSees":
        return str(json_value)

    _apply_env_aliases()

    for env_key in OPENSEES_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if env_value:
            return os.path.expandvars(os.path.expanduser(env_value.strip()))

    return str(json_value or "OpenSees")


# Stop early if the input file contains invalid values.
def _validate(beam_input: BeamInput) -> None:
    supported_types = {
        "cantilever_tip_load",
        "simply_supported_midspan_point_load",
    }
    supported_directions = {
        "x_positive",
        "x_negative",
        "y_positive",
        "y_negative",
    }

    if beam_input.length <= 0:
        raise ValueError("beam.length must be greater than zero.")
    if beam_input.num_elements <= 0:
        raise ValueError("beam.num_elements must be greater than zero.")
    if beam_input.case_type not in supported_types:
        raise ValueError(
            "case_type must be one of: cantilever_tip_load, simply_supported_midspan_point_load."
        )
    if beam_input.case_type == "simply_supported_midspan_point_load":
        if beam_input.num_elements % 2 != 0:
            raise ValueError(
                "beam.num_elements must be even for simply_supported_midspan_point_load so the midspan load lands on a node."
            )
    if beam_input.section.area <= 0:
        raise ValueError("section.A must be greater than zero.")
    if beam_input.material.elastic_modulus <= 0:
        raise ValueError("material.E must be greater than zero.")
    if beam_input.section.inertia_z <= 0:
        raise ValueError("section.Iz must be greater than zero.")
    if beam_input.section.inertia_y is not None and beam_input.section.inertia_y <= 0:
        raise ValueError("section.Iy must be greater than zero when provided.")
    if beam_input.load.load_type != "point_force":
        raise ValueError("load.type must be 'point_force'.")
    if beam_input.load.direction not in supported_directions:
        raise ValueError(
            "load.direction must be one of: x_positive, x_negative, y_positive, y_negative."
        )
    if not 1 <= beam_input.load.node <= beam_input.num_nodes:
        raise ValueError("load.node must be between 1 and the last node.")
    expected_x = _case_load_x(beam_input.case_type, beam_input.length)
    expected_node = _case_load_node(beam_input.case_type, beam_input.num_elements)
    if abs(beam_input.load.x - expected_x) > 1.0e-9:
        raise ValueError("load.x does not match the selected canonical case_type.")
    if beam_input.load.node != expected_node:
        raise ValueError("load.node does not match the selected canonical case_type.")
    if beam_input.stations < 2:
        raise ValueError("analysis.stations must be at least 2.")
    for position in beam_input.response_positions:
        if position < -1.0e-9 or position > beam_input.length + 1.0e-9:
            raise ValueError(
                "analysis.response_positions values must be between 0 and beam.length."
            )


def _require_object(raw: dict, key: str) -> dict:
    value = raw[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_units(raw_units: object) -> dict[str, str]:
    if not isinstance(raw_units, dict):
        return {}
    return {
        key: str(value)
        for key, value in raw_units.items()
        if key in {"length", "force", "moment"}
    }


def _response_positions(analysis: object, default_x: float) -> tuple[float, ...]:
    if not isinstance(analysis, dict):
        return (default_x,)

    raw_positions = analysis.get("response_positions")
    if raw_positions is None:
        return (default_x,)
    if not isinstance(raw_positions, list):
        raise ValueError("analysis.response_positions must be an array when provided.")
    return tuple(float(value) for value in raw_positions) or (default_x,)


def _case_load_x(case_type: str, length: float) -> float:
    if case_type == "cantilever_tip_load":
        return length
    if case_type == "simply_supported_midspan_point_load":
        return length / 2.0
    raise ValueError(f"Unsupported case_type '{case_type}'.")


def _case_load_node(case_type: str, num_elements: int) -> int:
    if case_type == "cantilever_tip_load":
        return num_elements + 1
    if case_type == "simply_supported_midspan_point_load":
        return (num_elements // 2) + 1
    raise ValueError(f"Unsupported case_type '{case_type}'.")
