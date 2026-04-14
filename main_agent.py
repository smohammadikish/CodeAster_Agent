# ============================================================
#  main_agentic.py - OpenAI LangChain Agent
#  Beam simulation assistant using Code_Aster & Gmsh
# ============================================================

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# ----------------------------
# Load environment
# ----------------------------
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# LangChain imports
from langchain_openai import ChatOpenAI

from env_loader import load_env  # noqa: E402
from input_parser import (ParameterSheet, build_state_aware_input,
                          parse_beam_request, resolve_analysis_inputs)
from library import (list_materials, list_sections, resolve_material_family_name,
                     resolve_section_family_name)
from storage import (create_run_id, ensure_base_dirs, ensure_run_dirs,
                     finalize_run, get_registered_run_paths, record_artifact,
                     record_tool_event, register_run, resolve_workspace_file,
                     utc_now_iso)
from tools.catalog_tools import (build_parametric_section_catalog_tool,
                                 get_material_catalog_tool,
                                 get_section_catalog_tool,
                                 list_material_catalog_tool,
                                 list_section_catalog_tool)
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.validators import file_exists
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)
from units import DEFAULT_UNIT_POLICY, normalize_user_params_mixed_units

VERIFICATION_ROOT = Path(__file__).resolve().parent / "verification"
if str(VERIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_ROOT))

from orchestrator import run_both_solvers

# ============================================================
#  Setup
# ============================================================
ensure_base_dirs()
gmsh_bin, as_run_bin, _, openai_key, openai_model = load_env()

logging.basicConfig(
    filename=str(Path("workspace/logs/hybrid_pipeline_agentic.log").resolve()),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Agentic pipeline started (LangChain 0.3.x).")

# ============================================================
#  Default Parameters (IN METERS for geometry)
# ============================================================
# NOTE:
# - All geometry-related parameters (length_m, HY, HZ) are in meters when
#   interacting with the user and inside the LLM / agent logic.
# - They are converted to millimeters ONLY inside the deterministic pipeline
#   before calling the tools, because all tools work in millimeters.
DEFAULT_PARAMS: Dict[str, Any] = {
    "length_m": 3.0,  # m
    "n_elems": 15,
    "E": 2.10e11,  # Pa
    "NU": 0.3,
    "RHO": 7850.0,  # kg/m^3
    "HY": 0.12,  # m
    "HZ": 0.10,  # m
    "A_mm2": None,
    "IY_mm4": None,
    "IZ_mm4": None,
    "JX_mm4": None,
    "AY_mm2": None,
    "AZ_mm2": None,
    "FY": -1500.0,  # N (negative = downward)
}
DEFAULT_INPUT_LENGTH_UNIT = DEFAULT_UNIT_POLICY.user_length_unit
DEFAULT_INPUT_FORCE_UNIT = DEFAULT_UNIT_POLICY.user_force_unit


# ============================================================
#  Helpers for params / JSON
# ============================================================


def coerce_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply numeric casting and fill missing keys from DEFAULT_PARAMS.

    IMPORTANT:
    - All geometry parameters are assumed to be in meters here.
    - Conversion to millimeters happens only inside `run_hybrid_pipeline`.
    """
    out = dict(DEFAULT_PARAMS)

    def cast(x, default):
        try:
            if isinstance(default, int):
                return int(float(x))
            return float(x)
        except Exception:
            return default

    for key in DEFAULT_PARAMS:
        if key in raw and raw[key] is not None:
            out[key] = cast(raw[key], DEFAULT_PARAMS[key])
    return out


def dict_to_pretty_json(d: Dict[str, Any]) -> str:
    try:
        return json.dumps(d, indent=2, ensure_ascii=False)
    except Exception:
        return str(d)


def console_print(*parts: object) -> None:
    """
    Print safely even when the terminal encoding cannot represent all characters.
    """
    text = " ".join(str(part) for part in parts)
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))


DIMENSION_WITH_OPTIONAL_UNIT = re.compile(
    r"\b(?P<name>hy|hz|height|width|length|length_m|beam length)\b"
    r"\s*[:=]?\s*(?P<value>-?\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|meter|meters|metre|metres)?\b",
    flags=re.IGNORECASE,
)


def get_ambiguous_dimension_message(user_input: str) -> str | None:
    """
    Reject bare geometry values like 'hy 20' unless the unit is explicit.
    """
    for match in DIMENSION_WITH_OPTIONAL_UNIT.finditer(user_input):
        if not match.group("unit"):
            return (
                "Please specify geometry units explicitly for length, HY, and HZ. "
                "I will not guess units for values like 'hy 20' or 'hz 10'. "
                "Use examples such as 'HY=0.20 m', 'HZ=100 mm', or 'HZ=10 cm'."
            )
    return None


# ============================================================
#  Safe pipeline helper (used internally inside run_hybrid_pipeline)
# ============================================================


def _add_run_log_handler(run_id: str) -> logging.Handler:
    run_log_path = resolve_workspace_file("pipeline.log", "logs", run_id).resolve()
    handler = logging.FileHandler(run_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _remove_run_log_handler(handler: logging.Handler) -> None:
    logger = logging.getLogger()
    logger.removeHandler(handler)
    handler.close()


def safe_call(tool_func, run_id: str, expected_artifacts=None, **kwargs) -> bool:
    """
    Safely run a LangChain tool used in the deterministic pipeline.

    - Prints which tool is running.
    - Logs the call and errors.
    - Records tool events and produced artifacts.
    """
    expected_artifacts = expected_artifacts or []
    console_print(f"\nRunning: {tool_func.name}")
    logging.info(f"Running {tool_func.name} with args={kwargs}")
    started_at = utc_now_iso()
    try:
        result = tool_func.invoke({**kwargs, "run_id": run_id})
    except Exception as e:
        console_print(f"Error in {tool_func.name}: {e}")
        logging.exception(f"{tool_func.name} failed")
        record_tool_event(
            run_id,
            tool_func.name,
            "failed",
            started_at,
            utc_now_iso(),
            str(e),
            kwargs,
        )
        return False

    console_print(result)

    if "Error" in str(result) or "❌" in str(result):
        console_print(f"{tool_func.name} failed.")
        record_tool_event(
            run_id,
            tool_func.name,
            "failed",
            started_at,
            utc_now_iso(),
            str(result),
            kwargs,
        )
        return False

    for artifact_type, artifact_stage, artifact_path in expected_artifacts:
        if not file_exists(str(artifact_path)):
            console_print(f"Expected file missing: {artifact_path}")
            record_tool_event(
                run_id,
                tool_func.name,
                "failed",
                started_at,
                utc_now_iso(),
                f"Expected file missing: {artifact_path}",
                kwargs,
            )
            return False
        record_artifact(run_id, artifact_type, artifact_stage, str(artifact_path))

    console_print(f"Step '{tool_func.name}' OK.\n")
    record_tool_event(
        run_id,
        tool_func.name,
        "completed",
        started_at,
        utc_now_iso(),
        str(result),
        kwargs,
    )
    return True


# ============================================================
#  Deterministic pipeline (INTERNAL, converts m → mm for tools)
# ============================================================


def run_hybrid_pipeline(params: Dict[str, Any]) -> str:
    """
    Run the full deterministic simulation pipeline.

    INPUT:
    - params: dictionary where geometry is in meters (length_m, HY, HZ)
      and load FY is in Newtons (user-level units).

    INTERNAL:
    - This function converts all geometry from meters to millimeters BEFORE
      calling the tools, because all tools are implemented in millimeters.
    """
    console_print("\nStarting deterministic simulation pipeline...\n")

    # -------------------------------------------------
    # Unit conversion: meters → millimeters for tools
    # -------------------------------------------------
    # We keep a local copy so logging and prints remain consistent.
    p = dict(params)

    # Convert geometric dimensions from meters to millimeters
    p["length_m"] = p["length_m"] * 1000.0
    if p.get("HY") is not None:
        p["HY"] = p["HY"] * 1000.0
    if p.get("HZ") is not None:
        p["HZ"] = p["HZ"] * 1000.0

    # Ensure load FY is negative (downward) unless user explicitly gave negative
    if p["FY"] > 0:
        p["FY"] = -abs(p["FY"])

    logging.info(
        "run_hybrid_pipeline with internal (mm) params:\n" + dict_to_pretty_json(p)
    )

    # -------------------------------------------------
    # GEO generation (uses millimeters internally)
    # -------------------------------------------------
    if not safe_call(
        generate_geo_tool,
        "",
        expected_artifacts=[
            ("geo", "workspace", resolve_workspace_file("model.geo", "inputs"))
        ],
        length_mm=p["length_m"],  # NOTE: now explicitly mm
        n_elems=p["n_elems"],
        geo_name="model.geo",
    ):
        return "Failed in generate_geo_tool"

    # -------------------------------------------------
    # Meshing with Gmsh (still mm)
    # -------------------------------------------------
    if not safe_call(
        mesh_with_gmsh_tool,
        "",
        expected_artifacts=[
            ("med", "workspace", resolve_workspace_file("model.med", "inputs"))
        ],
        geo_name="model.geo",
        med_name="model.med",
    ):
        return "Failed in mesh_with_gmsh_tool"

    # -------------------------------------------------
    # COMM generation for Code_Aster (geometry in mm)
    # -------------------------------------------------
    if not safe_call(
        generate_comm_tool,
        "",
        expected_artifacts=[
            ("comm", "workspace", resolve_workspace_file("model.comm", "inputs"))
        ],
        med_name="model.med",
        comm_name="model.comm",
        E=p["E"],
        NU=p["NU"],
        RHO=p["RHO"],
        A_mm2=p.get("A_mm2"),
        IY_mm4=p.get("IY_mm4"),
        IZ_mm4=p.get("IZ_mm4"),
        JX_mm4=p.get("JX_mm4"),
        AY_mm2=p.get("AY_mm2"),
        AZ_mm2=p.get("AZ_mm2"),
        HY_mm=p.get("HY", 120.0),
        HZ_mm=p.get("HZ", 100.0),
        FY_N=p["FY"],
    ):
        return "Failed in generate_comm_tool"

    # -------------------------------------------------
    # EXPORT file generation
    # -------------------------------------------------
    if not safe_call(
        generate_export_tool,
        "",
        expected_artifacts=[
            ("export", "workspace", resolve_workspace_file("model.export", "inputs"))
        ],
        comm_name="model.comm",
        export_name="model.export",
        job_name="beam_job",
        med_name="model.med",
    ):
        return "Failed in generate_export_tool"

    # -------------------------------------------------
    # Run Code_Aster
    # -------------------------------------------------
    if not safe_call(
        run_code_aster_tool,
        "",
        expected_artifacts=[
            ("mess", "workspace", resolve_workspace_file("beam_job.mess", "outputs")),
            ("resu", "workspace", resolve_workspace_file("beam_job.resu", "outputs")),
            ("tab_reac", "workspace", resolve_workspace_file("tab_reac.txt", "outputs")),
            ("tab_disp", "workspace", resolve_workspace_file("tab_disp.txt", "outputs")),
            ("local_axes", "workspace", resolve_workspace_file("local_axes.txt", "outputs")),
            ("tab_sie", "workspace", resolve_workspace_file("tab_sie.txt", "outputs")),
        ],
        export_name="model.export",
    ):
        return "Failed in run_code_aster_tool"

    # -------------------------------------------------
    # Visualization phase
    # -------------------------------------------------
    console_print("\nVisualization phase ...")
    visualization_failures = []

    if not safe_call(
        visualize_resu_tool,
        "",
        expected_artifacts=[
            ("deflection_plot", "workspace", resolve_workspace_file("beam_job_def.png", "viz")),
            ("rotation_plot", "workspace", resolve_workspace_file("beam_job_rot.png", "viz")),
        ],
        resu_name="beam_job.resu",
    ):
        visualization_failures.append("visualize_resu_tool")

    if not safe_call(
        visualize_stress_tool,
        "",
        expected_artifacts=[
            ("shear_plot", "workspace", resolve_workspace_file("shear_diagram.png", "viz")),
            ("moment_plot", "workspace", resolve_workspace_file("moment_diagram.png", "viz")),
        ],
        stress_name="tab_sie.txt",
    ):
        visualization_failures.append("visualize_stress_tool")

    if not safe_call(
        visualize_mfz_contour_tool,
        "",
        expected_artifacts=[
            ("mfz_contour", "workspace", resolve_workspace_file("mfz_contour.png", "viz"))
        ],
        stress_name="tab_sie.txt",
    ):
        visualization_failures.append("visualize_mfz_contour_tool")

    if not safe_call(
        visualize_freebody_tool,
        "",
        expected_artifacts=[
            ("freebody_plot", "workspace", resolve_workspace_file("freebody.png", "viz"))
        ],
        reac_name="tab_reac.txt",
    ):
        visualization_failures.append("visualize_freebody_tool")

    if visualization_failures:
        return (
            "Simulation completed, but some visualization steps failed: "
            + ", ".join(visualization_failures)
        )

    console_print("\nAll steps completed successfully - Simulation finished!\n")
    return "Simulation finished successfully. Results saved in 'workspace/' directory."


def run_hybrid_pipeline_with_run_storage(
    params: Dict[str, Any],
    length_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    hy_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    hz_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    force_unit: str = DEFAULT_INPUT_FORCE_UNIT,
) -> str:
    """
    Run the deterministic simulation pipeline using a dedicated run folder.
    """
    run_id = create_run_id("beam")
    run_paths = ensure_run_dirs(run_id)
    register_run(run_id, params, str(run_paths.root.resolve()))
    run_log_handler = _add_run_log_handler(run_id)

    console_print(f"\nStarting deterministic simulation pipeline...\nRun ID: {run_id}\n")

    try:
        p = dict(params)
        p = normalize_user_params_mixed_units(
            p,
            DEFAULT_UNIT_POLICY,
            length_unit=length_unit,
            hy_unit=hy_unit,
            hz_unit=hz_unit,
            force_unit=force_unit,
        )

        if p["FY"] > 0:
            p["FY"] = -abs(p["FY"])

        logging.info(
            "run_hybrid_pipeline_with_run_storage internal (mm) params:\n"
            + dict_to_pretty_json(p)
        )

        inputs = [
            ("geo", "inputs", resolve_workspace_file("model.geo", "inputs", run_id)),
            ("med", "inputs", resolve_workspace_file("model.med", "inputs", run_id)),
            ("comm", "inputs", resolve_workspace_file("model.comm", "inputs", run_id)),
            ("export", "inputs", resolve_workspace_file("model.export", "inputs", run_id)),
        ]
        outputs = [
            ("mess", "outputs", resolve_workspace_file("beam_job.mess", "outputs", run_id)),
            ("resu", "outputs", resolve_workspace_file("beam_job.resu", "outputs", run_id)),
            ("tab_reac", "outputs", resolve_workspace_file("tab_reac.txt", "outputs", run_id)),
            ("tab_disp", "outputs", resolve_workspace_file("tab_disp.txt", "outputs", run_id)),
            ("local_axes", "outputs", resolve_workspace_file("local_axes.txt", "outputs", run_id)),
            ("tab_sie", "outputs", resolve_workspace_file("tab_sie.txt", "outputs", run_id)),
        ]
        viz = [
            ("deflection_plot", "viz", resolve_workspace_file("beam_job_def.png", "viz", run_id)),
            ("rotation_plot", "viz", resolve_workspace_file("beam_job_rot.png", "viz", run_id)),
            ("shear_plot", "viz", resolve_workspace_file("shear_diagram.png", "viz", run_id)),
            ("moment_plot", "viz", resolve_workspace_file("moment_diagram.png", "viz", run_id)),
            ("mfz_contour", "viz", resolve_workspace_file("mfz_contour.png", "viz", run_id)),
            ("freebody_plot", "viz", resolve_workspace_file("freebody.png", "viz", run_id)),
        ]

        if not safe_call(
            generate_geo_tool,
            run_id,
            expected_artifacts=[inputs[0]],
            length_mm=p["length_m"],
            n_elems=p["n_elems"],
            geo_name="model.geo",
        ):
            finalize_run(run_id, "failed", "Failed in generate_geo_tool")
            return f"Failed in generate_geo_tool (run_id={run_id})"

        if not safe_call(
            mesh_with_gmsh_tool,
            run_id,
            expected_artifacts=[inputs[1]],
            geo_name="model.geo",
            med_name="model.med",
        ):
            finalize_run(run_id, "failed", "Failed in mesh_with_gmsh_tool")
            return f"Failed in mesh_with_gmsh_tool (run_id={run_id})"

        if not safe_call(
            generate_comm_tool,
            run_id,
            expected_artifacts=[inputs[2]],
            med_name="model.med",
            comm_name="model.comm",
            E=p["E"],
            NU=p["NU"],
            RHO=p["RHO"],
            A_mm2=p.get("A_mm2"),
            IY_mm4=p.get("IY_mm4"),
            IZ_mm4=p.get("IZ_mm4"),
            JX_mm4=p.get("JX_mm4"),
            AY_mm2=p.get("AY_mm2"),
            AZ_mm2=p.get("AZ_mm2"),
            HY_mm=p.get("HY", 120.0),
            HZ_mm=p.get("HZ", 100.0),
            FY_N=p["FY"],
        ):
            finalize_run(run_id, "failed", "Failed in generate_comm_tool")
            return f"Failed in generate_comm_tool (run_id={run_id})"

        if not safe_call(
            generate_export_tool,
            run_id,
            expected_artifacts=[inputs[3]],
            comm_name="model.comm",
            export_name="model.export",
            job_name="beam_job",
            med_name="model.med",
        ):
            finalize_run(run_id, "failed", "Failed in generate_export_tool")
            return f"Failed in generate_export_tool (run_id={run_id})"

        if not safe_call(
            run_code_aster_tool,
            run_id,
            expected_artifacts=outputs,
            export_name="model.export",
        ):
            finalize_run(run_id, "failed", "Failed in run_code_aster_tool")
            return f"Failed in run_code_aster_tool (run_id={run_id})"

        console_print("\nVisualization phase ...")
        visualization_failures = []

        if not safe_call(
            visualize_resu_tool,
            run_id,
            expected_artifacts=viz[0:2],
            resu_name="beam_job.resu",
        ):
            visualization_failures.append("visualize_resu_tool")

        if not safe_call(
            visualize_stress_tool,
            run_id,
            expected_artifacts=viz[2:4],
            stress_name="tab_sie.txt",
        ):
            visualization_failures.append("visualize_stress_tool")

        if not safe_call(
            visualize_mfz_contour_tool,
            run_id,
            expected_artifacts=[viz[4]],
            stress_name="tab_sie.txt",
        ):
            visualization_failures.append("visualize_mfz_contour_tool")

        if not safe_call(
            visualize_freebody_tool,
            run_id,
            expected_artifacts=[viz[5]],
            reac_name="tab_reac.txt",
        ):
            visualization_failures.append("visualize_freebody_tool")

        meta_path = resolve_workspace_file("run.json", "meta", run_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "status": "completed_with_warnings"
                    if visualization_failures
                    else "completed",
                    "params_user_units": params,
                    "input_units": {
                        "length": length_unit,
                        "HY": hy_unit,
                        "HZ": hz_unit,
                        "force": force_unit,
                    },
                    "params_internal_mm": p,
                    "unit_policy": DEFAULT_UNIT_POLICY.to_dict(),
                },
                f,
                indent=2,
            )
        record_artifact(run_id, "run_metadata", "meta", str(meta_path))

        if visualization_failures:
            finalize_run(
                run_id,
                "completed_with_warnings",
                "Visualization failures: " + ", ".join(visualization_failures),
            )
            return (
                "Simulation completed, but some visualization steps failed: "
                + ", ".join(visualization_failures)
                + f"\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"
            )

        finalize_run(run_id, "completed")
        console_print("\nAll steps completed successfully - Simulation finished!\n")
        return (
            "Simulation finished successfully.\n"
            f"Run ID: {run_id}\n"
            f"Run folder: {run_paths.root.resolve()}"
        )
    except Exception as exc:
        finalize_run(run_id, "failed", str(exc))
        raise
    finally:
        _remove_run_log_handler(run_log_handler)


# ============================================================
#  High-level tool: run_full_pipeline
# ============================================================


@tool
def run_full_pipeline(
    length_m: float = None,
    n_elems: int = None,
    E: float = None,
    NU: float = None,
    RHO: float = None,
    HY: float = None,
    HZ: float = None,
    A_mm2: float = None,
    IY_mm4: float = None,
    IZ_mm4: float = None,
    JX_mm4: float = None,
    AY_mm2: float = None,
    AZ_mm2: float = None,
    FY: float = None,
    length_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    hy_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    hz_unit: str = DEFAULT_INPUT_LENGTH_UNIT,
    force_unit: str = DEFAULT_INPUT_FORCE_UNIT,
    confirm: bool = False,
) -> str:
    """
    Run the complete simulation pipeline ONLY after explicit user confirmation.

    IMPORTANT UNITS (FOR THE USER / LLM SIDE):
    - length_m: beam length in `length_unit`
    - HY/HZ: optional rectangular dimensions in `hy_unit`/`hz_unit`
    - A_mm2/IY_mm4/IZ_mm4/JX_mm4/AY_mm2/AZ_mm2: preferred section properties for Code_Aster
    - FY: load in `force_unit`, positive or negative
    - E: Young's modulus [Pa]
    - NU: Poisson ratio [-]
    - RHO: density [kg/m^3]
    - n_elems: number of finite elements

    Rules:
    - If ANY required parameter is missing → return an instructional message
      instead of running the pipeline.
    - If confirm=False → return a message asking the agent to request confirmation.
    - If all parameters exist AND confirm=True → run the deterministic pipeline.
    """

    # 1) Check for missing required parameters
    params_raw = {
        "length_m": length_m,
        "n_elems": n_elems,
        "E": E,
        "NU": NU,
        "RHO": RHO,
        "HY": HY,
        "HZ": HZ,
        "A_mm2": A_mm2,
        "IY_mm4": IY_mm4,
        "IZ_mm4": IZ_mm4,
        "JX_mm4": JX_mm4,
        "AY_mm2": AY_mm2,
        "AZ_mm2": AZ_mm2,
        "FY": FY,
        "length_unit": length_unit,
        "hy_unit": hy_unit,
        "hz_unit": hz_unit,
        "force_unit": force_unit,
    }

    missing = [
        name
        for name, val in params_raw.items()
        if name not in {"length_unit", "hy_unit", "hz_unit", "force_unit", "HY", "HZ", "A_mm2", "IY_mm4", "IZ_mm4", "JX_mm4", "AY_mm2", "AZ_mm2"} and val is None
    ]
    has_section = (
        all(value is not None for value in (A_mm2, IY_mm4, IZ_mm4, JX_mm4, AY_mm2, AZ_mm2))
        or (HY is not None and HZ is not None)
    )
    if not has_section:
        missing.append("section_definition")

    if missing:
        return (
            "Missing required parameters: "
            + ", ".join(missing)
            + "\nPlease ask the user to provide these values together with explicit units, "
            "or explicitly approve using default values."
        )

    # 2) Check for confirmation flag
    if confirm is False:
        return (
            "All parameters are provided, but the simulation has NOT been confirmed.\n"
            "Ask the user to confirm the full simulation with the stated units."
        )

    # 3) All good → run pipeline
    params = coerce_params(
        {
            "length_m": length_m,
            "n_elems": n_elems,
            "E": E,
            "NU": NU,
            "RHO": RHO,
            "HY": HY,
            "HZ": HZ,
            "A_mm2": A_mm2,
            "IY_mm4": IY_mm4,
            "IZ_mm4": IZ_mm4,
            "JX_mm4": JX_mm4,
            "AY_mm2": AY_mm2,
            "AZ_mm2": AZ_mm2,
            "FY": FY,
        }
    )
    logging.info(
        "run_full_pipeline confirmed with params (user units):\n"
        + dict_to_pretty_json(params)
    )

    summary = run_hybrid_pipeline_with_run_storage(
        params,
        length_unit=length_unit,
        hy_unit=hy_unit,
        hz_unit=hz_unit,
        force_unit=force_unit,
    )

    return (
        f"{summary}\n\n"
        f"Input units: length={length_unit}, HY={hy_unit}, HZ={hz_unit}, force={force_unit}\n"
        "Used parameters:\n"
        f"{dict_to_pretty_json(params)}"
    )


# ============================================================
#  Agent construction
# ============================================================


def build_system_prompt() -> str:
    return """You are a mechanical simulation assistant specialized in beam analysis using Code_Aster and Gmsh. You support multi-step conversations where the user incrementally provides beam geometry, material properties, cross-section definitions, and loads.

============================================================
UNITS CONVENTION (CRITICAL)
============================================================
- The user may provide mixed units such as m, cm, mm, N, or kN.
- You must accept explicit user units and convert them internally.
- Do not ask the user to manually convert 20 cm to 0.20 m or 1.2 kN to 1200 N.
- Keep every numeric value together with its own unit while collecting data.
- If the user says '5 m long, 20 cm high, 12 cm wide, 1.2 kN load', treat that as valid.
- If the user gives a named material or a named section from the catalog, prefer resolving it instead of asking for raw properties.
- Only ask follow-up questions when a value is missing or a unit is absent and ambiguous.

============================================================
CATALOG CAPABILITIES
============================================================
- Material families in the catalog: steel, aluminum, concrete, wood.
- Section families in the catalog: steel, aluminum, concrete, wood.
- Parametric sections are available for rectangle and circle.
- Catalog materials return solver-ready values such as E, NU, and RHO.
- Catalog sections return Code_Aster-ready values A, IY, IZ, JX, AY, and AZ.
- Concrete can use a formula fallback when the exact class is not listed.

============================================================
BEHAVIOR RULES - FOLLOW THESE STRICTLY
============================================================

RESPONSE STYLE
- Keep normal replies compact and guided.
- Prefer 1 short intro sentence plus at most 4 bullets.
- Ask only for the missing items needed for the next step.
- Do not produce long dumps of solver properties unless the user explicitly asks for them.
- If the user asks for suggestions, give a short recommendation with a brief reason, then ask the next missing question.
- When listing missing inputs, prefer plain user language such as 'material' or 'section' instead of raw solver fields like E, NU, RHO, A, IY, IZ, JX, AY, AZ unless the user asked for raw properties.
- For a partially defined beam, prefer wording like:
  "Please add:
  - elements
  - material, or I can suggest from my Eurocode-based database
  - section, or I can suggest one"
- Never invent catalog suggestions. Any suggested material or section must come from the real local catalog/database or from parametric section formulas.

1) Use catalog tools when the user asks for available materials, available sections, or named catalog items.
   - Use list_material_catalog for material lists.
   - Use get_material_catalog for one named material.
   - Use list_section_catalog for section lists.
   - Use get_section_catalog for one named section.
   - Use build_parametric_section_catalog for simple rectangle or circle sections.
   - Never invent catalog contents from memory.
   - If the user selects only a family such as steel, aluminum, concrete, or wood, list the actual entries available in that family.

2) Do not call run_full_pipeline until all required parameters are known:
   - length_m  (beam length)
   - n_elems   (integer)
   - E, NU, RHO or a resolvable material name
   - A, IY, IZ, JX, AY, AZ or a resolvable section definition
   - FY        (load)

3) When calling run_full_pipeline, pass units explicitly per field:
   - length_unit for length_m
   - hy_unit for HY
   - hz_unit for HZ
   - force_unit for FY
   Allowed length units: m, cm, mm. Allowed force units: N, kN.

4) If any required value is missing:
   - Ask follow-up questions.
   - Do not assume defaults unless the user explicitly says 'Use default values'.

5) When the user gives a known material or section name:
   - Prefer the internal catalog and formulas to resolve properties.
   - Use Eurocode references for catalog materials.
   - For concrete grades not in the table, use the formula fallback.
   - If the user says a shorthand concrete grade like C25, resolve it to the closest standard catalog grade such as C25/30 when available.
   - If the user gives a concrete grade not present in the table, do not ask again; calculate the concrete properties directly with the Eurocode formula from the provided fck.
   - When mentioning the concrete fallback, say clearly that E is calculated with the Eurocode formula based on the user's fck.

5a) When suggesting materials or sections from the catalog:
   - Explicitly say they are suggestions from the local database/catalog.
   - Mention that many catalog items are based on Eurocode references.
   - Make clear they are not the only valid choices.
   - Tell the user they may instead provide their own material properties or section properties/dimensions.
   - Do not promise an internet search unless that capability is actually available in the current app/session.

6) When the user does not know material properties and no catalog match exists:
   - Do not call run_full_pipeline.
   - Ask for either a valid catalog material or explicit values for E, NU, and RHO.

7) When all parameters are known:
   - Summarize them clearly and preserve the user's units.
   - Keep the summary short and user-facing.
   - If a concrete material was resolved by formula instead of the catalog table, mark it with '(calculated by Eurocode formula)'.
   - Then ask:
       "Do you confirm running the simulation with these values?"
   - Never call run_full_pipeline unless the user explicitly confirms with:
       'yes', 'confirm', 'ok run', or similar.

8) If the user responds with 'no' or 'not yet':
   - Do not call any tools.
   - Ask what they want to modify.

9) FINALIZATION / STOP RULES:
   If the user says any of these in the conversation:
       'stop', 'finish', 'end', 'cancel', 'no more', 'terminate', "that's enough", 'done'
   - Immediately stop all simulation workflow.
   - Do not call any tools.
   - Simply acknowledge and wait for the next input (or let the CLI handle exit).
   - Never restart the simulation pipeline unless the user explicitly asks.

10) Never rerun run_full_pipeline automatically.
   - A simulation may run only once per explicit confirmation.
   - Never repeat runs after the user says it is finished.

11) If the user gives a geometry number without an explicit unit (for example 'hy 20' or 'hz 10'):
   - Treat it as ambiguous.
   - Ask the user to restate that dimension with units.
   - Never guess whether it means m, cm, or mm.

12) Maintain conversation memory and be conversational and helpful, but follow all rules above with 100% strictness.

============================================================
YOUR GOAL
============================================================
Guide the user step-by-step, help determine missing values, ensure explicit confirmation before executing any tools, and respect all stop commands."""


def is_confirmation_input(user_input: str) -> bool:
    text = user_input.strip().lower()
    return bool(
        re.search(
            r"\b(yes|confirm|confirmed|i confirm|ok run|run it|yes run it|please run)\b",
            text,
        )
    )


def is_exit_input(user_input: str) -> bool:
    text = user_input.strip().lower()
    return bool(
        re.search(
            r"\b(exit|quit|stop|end|finish|cancel|bye|goodbye|ok bye|thanks|thank you|ok thanks)\b",
            text,
        )
    )


def build_post_run_message() -> str:
    return (
        "The simulation has already been completed for the current parameter sheet.\n"
        "If you want another run, change one or more parameters or start a new analysis."
    )


def build_post_success_message(run_output: str) -> str:
    lines = [line.rstrip() for line in str(run_output).splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        if line.startswith("Input units:") or line.startswith("Used parameters:"):
            break
        kept.append(line)
    if not kept:
        kept = ["Simulation finished successfully."]
    kept.append("Do you want me to run verification for this run as well?")
    return "\n".join(kept)


def build_post_run_idle_message() -> str:
    return "You can change parameters for another run, or finish the process."


def is_negative_response(user_input: str) -> bool:
    text = user_input.strip().lower()
    return bool(re.search(r"\b(no|not now|no thanks|skip|later)\b", text))


def is_verification_input(user_input: str) -> bool:
    text = user_input.strip().lower()
    return bool(re.search(r"\b(verify|verification|verif|validate|check results)\b", text))


def extract_run_id_and_folder(run_output: str) -> tuple[str | None, Path | None]:
    run_id_match = re.search(r"Run ID:\s*(\S+)", run_output)
    run_folder_match = re.search(r"Run folder:\s*(.+)", run_output)
    run_id = run_id_match.group(1).strip() if run_id_match else None
    run_folder = Path(run_folder_match.group(1).strip()) if run_folder_match else None
    return run_id, run_folder


def run_verification_for_run(run_id: str) -> str:
    run_paths = get_registered_run_paths(run_id)
    if run_paths is None or not run_paths.root.exists():
        return f"Verification could not start because run '{run_id}' was not found."

    input_dir = run_paths.inputs
    verification_dir = run_paths.root / "verification"

    if not input_dir.exists():
        return (
            "Verification could not start because the run inputs folder is missing.\n"
            f"Expected folder: {input_dir}"
        )

    required_files = ["model.comm", "model.export", "model.geo"]
    missing_files = [name for name in required_files if not (input_dir / name).exists()]
    if missing_files:
        return (
            "Verification could not start because the run inputs are incomplete.\n"
            f"Missing files in {input_dir}: {', '.join(missing_files)}"
        )

    try:
        _, manual_result, opensees_result, comparison = run_both_solvers(
            input_dir,
            verification_root=verification_dir,
        )
    except Exception as exc:
        logging.exception("Verification failed for run_id=%s", run_id)
        return f"Verification failed for run {run_id}: {exc}"

    if manual_result and opensees_result:
        status_line = "Verification finished successfully."
    elif manual_result or opensees_result:
        status_line = "Verification finished with partial results."
    else:
        status_line = "Verification failed."

    return (
        f"{status_line}\n"
        f"Run ID: {run_id}\n"
        f"Verification folder: {verification_dir.resolve()}\n"
        f"Manual: {(verification_dir / 'manual' / 'result.json').resolve()}\n"
        f"OpenSees: {(verification_dir / 'opensees' / 'results.json').resolve()}\n"
        f"Comparison: {(verification_dir / 'comparison' / 'comparison_report.txt').resolve()}"
    )


def get_sheet_signature(sheet: ParameterSheet) -> str:
    return json.dumps(sheet.to_dict(), sort_keys=True, ensure_ascii=True)


def build_confirmation_summary(sheet: ParameterSheet) -> str:
    resolution = resolve_analysis_inputs(sheet.to_dict())
    values = resolution["resolved"]
    material = values.get("resolved_material")
    section = values.get("resolved_section")
    material_suffix = (
        " (calculated by Eurocode formula)"
        if material and material.get("derived")
        else ""
    )
    material_line = (
        f"- Material: {material.get('name')} ({material.get('reference')}){material_suffix}"
        if material
        else ""
    )
    section_name = section.get("Profile") or section.get("Section") or section.get("shape") if section else None
    section_ref = section.get("reference") if section else None
    section_line = (
        f"- Section: {section_name} ({section_ref})"
        if section_name
        else ""
    )
    lines = [
        "All required parameters are ready.",
        f"- Length: {values['length_m']} {values.get('length_unit', DEFAULT_INPUT_LENGTH_UNIT)}",
        f"- Elements: {values['n_elems']}",
        f"- Load FY: {values['FY']} {values.get('force_unit', DEFAULT_INPUT_FORCE_UNIT)}",
    ]
    if material_line:
        lines.append(material_line)
    if section_line:
        lines.append(section_line)
    if material or section:
        lines.append("- Solver properties were resolved from the selected catalog entries.")
    if resolution["notes"]:
        lines.extend(f"- Note: {note}" for note in resolution["notes"])
    lines.append("Reply with 'yes', 'confirm', or 'run it' to start the simulation.")
    return "\n".join(lines)


def build_pipeline_kwargs_from_sheet(sheet: ParameterSheet) -> dict[str, Any]:
    values = resolve_analysis_inputs(sheet.to_dict())["resolved"]
    return {
        "length_m": values["length_m"],
        "n_elems": values["n_elems"],
        "E": values["E"],
        "NU": values["NU"],
        "RHO": values["RHO"],
        "HY": values.get("HY"),
        "HZ": values.get("HZ"),
        "A_mm2": values["A_mm2"],
        "IY_mm4": values["IY_mm4"],
        "IZ_mm4": values["IZ_mm4"],
        "JX_mm4": values["JX_mm4"],
        "AY_mm2": values["AY_mm2"],
        "AZ_mm2": values["AZ_mm2"],
        "FY": values["FY"],
        "length_unit": values.get("length_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "hy_unit": values.get("hy_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "hz_unit": values.get("hz_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "force_unit": values.get("force_unit", DEFAULT_INPUT_FORCE_UNIT),
        "confirm": True,
    }


def build_catalog_family_guidance(sheet: ParameterSheet) -> str | None:
    values = sheet.to_dict()
    resolution = sheet.resolved_state()
    lines: list[str] = []

    material_hint = values.get("material_hint")
    if material_hint and not resolution["resolved"].get("resolved_material"):
        try:
            material_family = resolve_material_family_name(str(material_hint))
            material_options = list_materials(material_family)
            shown = material_options[:8]
            lines.append(
                f"For `{material_family}` materials in the local Eurocode-based database, available options are:"
            )
            lines.extend(f"- {item}" for item in shown)
            if len(material_options) > len(shown):
                lines.append("- ...")
            lines.append("Choose one, or provide `E`, `NU`, and `RHO`.")
        except Exception:
            pass

    section_hint = values.get("section_hint")
    if section_hint and not resolution["resolved"].get("resolved_section"):
        try:
            section_family = resolve_section_family_name(str(section_hint))
            if section_family == "parametric":
                lines.append(
                    "For the section, you can use a parametric `rectangle` or `circle` and provide the dimensions."
                )
            else:
                section_options = list_sections(section_family)
                shown = section_options[:8]
                lines.append(
                    f"For `{section_family}` sections in the local database, available options are:"
                )
                lines.extend(f"- {item}" for item in shown)
                if len(section_options) > len(shown):
                    lines.append("- ...")
                lines.append("Choose one, or provide your own section properties/dimensions.")
        except Exception:
            pass

    return "\n".join(lines) if lines else None


def build_agent() -> AgentExecutor:
    """
    Build the AgentExecutor that:
    - Acts as a mechanical simulation assistant for beam analysis.
    - Has access to all tools for geometry, meshing, Code_Aster, and visualization.
    - Decides which tools to call during a multi-step conversation.
    """

    # Main LLM
    llm = ChatOpenAI(
        model=openai_model or "gpt-4.1",
        api_key=openai_key,
        temperature=0,
        max_tokens=800,
    )

    # Expose catalog-assistance tools and keep final execution behind the
    # guarded high-level pipeline tool.
    tools = [
        list_material_catalog_tool,
        get_material_catalog_tool,
        list_section_catalog_tool,
        get_section_catalog_tool,
        build_parametric_section_catalog_tool,
        run_full_pipeline,  # High-level pipeline tool
    ]

    # Agent prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                build_system_prompt(),
            ),
            # Chat history buffer
            MessagesPlaceholder("chat_history", optional=True),
            # Current user message
            ("human", "{input}"),
            # Agent scratchpad for tool planning
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Create agent with tools
    agent = create_openai_tools_agent(llm, tools, prompt)

    # AgentExecutor for multi-step reasoning with tools
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
    )
    return agent_executor


def build_state_extractor_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=openai_model or "gpt-4.1",
        api_key=openai_key,
        temperature=0,
        max_tokens=500,
    )


# ============================================================
#  CLI / Entry point
# ============================================================


def run_agent_one_shot(
    agent_executor: AgentExecutor, user_input: str, extractor_llm: Any | None = None
) -> None:
    """
    One-shot execution (mode=agent): LLM decides which tools to use.

    NOTE TO THE USER (printed once before running):
    - Beam dimensions may be provided in m, cm, or mm.
    - Loads may be provided in N or kN.
    - The system converts them internally for the tools.
    """
    console_print("\nAGENT MODE (one-shot with tools)\n")
    console_print("Describe the beam with the values you know, and include units for each dimension or load.")
    console_print(
        "   Examples: 5 m beam with 1.2 kN load | 5 m beam, S355, IPE160, 12 elements, 1.2 kN tip load.\n"
    )

    ambiguous_message = get_ambiguous_dimension_message(user_input)
    if ambiguous_message:
        console_print("\nAgent response:\n")
        console_print(ambiguous_message)
        return

    try:
        sheet = ParameterSheet()
        state_input = build_state_aware_input(user_input, sheet, llm=extractor_llm)
        result = agent_executor.invoke({"input": state_input})
    except Exception as e:
        console_print("\nAgent error:\n")
        console_print(
            "The agent could not reach the language model. "
            "Please check your OPENAI_API_KEY, network access, and local firewall settings."
        )
        console_print(f"Details: {e}")
        return

    console_print("\nAgent response:\n")
    console_print(result.get("output", result))


def run_agent_chat(agent_executor: AgentExecutor, extractor_llm: Any | None = None) -> None:
    """
    Full interactive chat mode:
    - Keeps history between turns.
    - The LLM can decide to call tools at any step.
    - The user can exit with: exit, quit, stop, end, finish, cancel.
    """
    console_print("\nINTERACTIVE CHAT AGENT MODE")
    console_print(
        "Type 'exit', 'quit', 'stop', 'end', 'finish', or 'cancel' to terminate.\n"
    )
    console_print("Describe the beam with the values you know, and include units for each dimension or load.")
    console_print(
        "   Examples: 5 m beam with 1.2 kN load | 5 m beam, S355, IPE160, 12 elements, 1.2 kN tip load.\n"
    )

    chat_history: List[BaseMessage] = []
    parameter_sheet = ParameterSheet()
    simulation_executed = False
    last_executed_signature: str | None = None
    last_run_id: str | None = None
    verification_prompt_pending = False

    from langchain_core.messages import AIMessage, HumanMessage

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console_print("\nBye!")
            break

        if is_exit_input(user_input):
            console_print("\nBye! Chat session terminated.\n")
            break

        ambiguous_message = get_ambiguous_dimension_message(user_input)
        if ambiguous_message:
            console_print("\nAgent:\n" + ambiguous_message + "\n")
            continue

        # Run the agent with history
        try:
            parsed_update = parse_beam_request(user_input)
            state_input = build_state_aware_input(
                user_input, parameter_sheet, llm=extractor_llm
            )
            current_signature = get_sheet_signature(parameter_sheet)
            if last_executed_signature is not None and current_signature != last_executed_signature:
                simulation_executed = False
                verification_prompt_pending = False

            if simulation_executed and verification_prompt_pending and is_confirmation_input(user_input):
                if not last_run_id:
                    output_str = "I could not find a completed run to verify yet."
                else:
                    output_str = (
                        run_verification_for_run(last_run_id)
                        + "\nYou can change parameters for another run, request verification again, or finish the process."
                    )
                verification_prompt_pending = False
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            if simulation_executed and verification_prompt_pending and is_negative_response(user_input):
                output_str = "Okay. You can change parameters for another run, request verification later, or finish the process."
                verification_prompt_pending = False
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            if simulation_executed and is_verification_input(user_input):
                if not last_run_id:
                    output_str = "I could not find a completed run to verify yet."
                else:
                    output_str = (
                        run_verification_for_run(last_run_id)
                        + "\nYou can change parameters for another run, request verification again, or finish the process."
                    )
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            if simulation_executed and not parsed_update:
                output_str = build_post_run_idle_message()
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            if simulation_executed and is_confirmation_input(user_input):
                output_str = build_post_run_message()
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            if not parameter_sheet.missing_required_fields():
                if is_confirmation_input(user_input):
                    run_output = str(
                        run_full_pipeline.invoke(
                            build_pipeline_kwargs_from_sheet(parameter_sheet)
                        )
                    ).strip()
                    last_run_id, _ = extract_run_id_and_folder(run_output)
                    output_str = build_post_success_message(run_output)
                    simulation_executed = True
                    last_executed_signature = get_sheet_signature(parameter_sheet)
                    verification_prompt_pending = True
                    console_print("\nAgent:\n" + output_str + "\n")
                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=output_str))
                    continue

                output_str = build_confirmation_summary(parameter_sheet)
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
                continue

            family_guidance = build_catalog_family_guidance(parameter_sheet)
            if family_guidance:
                console_print("\nAgent:\n" + family_guidance + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=family_guidance))
                continue

            result = agent_executor.invoke(
                {
                    "input": state_input,
                    "chat_history": chat_history,
                }
            )
        except Exception as e:
            console_print(
                "\nAgent:\nThe agent could not reach the language model. "
                "Please check your OPENAI_API_KEY, network access, and local firewall settings."
            )
            console_print(f"Details: {e}\n")
            continue

        output = result.get("output", "")
        output_str = str(output).strip()

        console_print("\nAgent:\n" + output_str + "\n")

        # Append only user and AI messages to history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Beam Simulation Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["agent", "chat"],
        help="agent: one-shot NL prompt with tools; chat: interactive agent chat",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help=(
            "User natural language request "
            "(example: 'simulate a 4m beam with 25 elements and 2000 N'). "
            "All dimensions MUST be provided in meters (m)."
        ),
    )

    args = parser.parse_args()

    agent_executor = build_agent()
    extractor_llm = build_state_extractor_llm()

    if args.mode == "agent":
        if not args.input:
            raise ValueError("--input is required in agent mode.")
        run_agent_one_shot(agent_executor, args.input, extractor_llm=extractor_llm)

    elif args.mode == "chat":
        run_agent_chat(agent_executor, extractor_llm=extractor_llm)
