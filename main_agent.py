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

from dotenv import load_dotenv

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# LangChain imports
from langchain_openai import ChatOpenAI

from env_loader import load_env  # noqa: E402
from input_parser import ParameterSheet, build_state_aware_input, parse_beam_request
from storage import (create_run_id, ensure_base_dirs, ensure_run_dirs,
                     finalize_run, record_artifact, record_tool_event,
                     register_run, resolve_workspace_file, utc_now_iso)
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
    p["length_m"] = p["length_m"] * 1000.0  # m → mm
    p["HY"] = p["HY"] * 1000.0  # m → mm
    p["HZ"] = p["HZ"] * 1000.0  # m → mm

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
        os.path.join(WORKSPACE, "model.geo"),
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
        os.path.join(WORKSPACE, "model.med"),
        geo_name="model.geo",
        med_name="model.med",
    ):
        return "Failed in mesh_with_gmsh_tool"

    # -------------------------------------------------
    # COMM generation for Code_Aster (geometry in mm)
    # -------------------------------------------------
    if not safe_call(
        generate_comm_tool,
        os.path.join(WORKSPACE, "model.comm"),
        med_name="model.med",
        comm_name="model.comm",
        E=p["E"],
        NU=p["NU"],
        RHO=p["RHO"],
        HY_mm=p["HY"],  # mm
        HZ_mm=p["HZ"],  # mm
        FY_N=p["FY"],
    ):
        return "Failed in generate_comm_tool"

    # -------------------------------------------------
    # EXPORT file generation
    # -------------------------------------------------
    if not safe_call(
        generate_export_tool,
        os.path.join(WORKSPACE, "model.export"),
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
        os.path.join(WORKSPACE, "beam_job.mess"),
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
        os.path.join(WORKSPACE, "beam_job.resu"),
        resu_name="beam_job.resu",
    ):
        visualization_failures.append("visualize_resu_tool")

    if not safe_call(
        visualize_stress_tool,
        os.path.join(WORKSPACE, "tab_sie.txt"),
        stress_name="tab_sie.txt",
    ):
        visualization_failures.append("visualize_stress_tool")

    if not safe_call(
        visualize_mfz_contour_tool,
        os.path.join(WORKSPACE, "tab_sie.txt"),
        stress_name="tab_sie.txt",
    ):
        visualization_failures.append("visualize_mfz_contour_tool")

    if not safe_call(
        visualize_freebody_tool,
        os.path.join(WORKSPACE, "tab_reac.txt"),
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
            HY_mm=p["HY"],
            HZ_mm=p["HZ"],
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
    - HY: section height in `hy_unit`
    - HZ: section width in `hz_unit`
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
        "FY": FY,
        "length_unit": length_unit,
        "hy_unit": hy_unit,
        "hz_unit": hz_unit,
        "force_unit": force_unit,
    }

    missing = [
        name
        for name, val in params_raw.items()
        if name not in {"length_unit", "hy_unit", "hz_unit", "force_unit"} and val is None
    ]

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
    return (
        "You are a mechanical simulation assistant specialized in beam analysis "
        "using Code_Aster and Gmsh. You support multi-step conversations where "
        "the user incrementally provides beam geometry, material properties, "
        "cross-section dimensions, and loads.\n\n"
        "============================================================\n"
        "UNITS CONVENTION (CRITICAL)\n"
        "============================================================\n"
        "- The user may provide mixed units such as m, cm, mm, N, or kN.\n"
        "- You MUST accept explicit user units and convert them internally.\n"
        "- Do NOT ask the user to manually convert 20 cm to 0.20 m or 1.2 kN to 1200 N.\n"
        "- Keep every numeric value together with its own unit while collecting data.\n"
        "- If the user says '5 m long, 20 cm high, 12 cm wide, 1.2 kN load', treat that as valid.\n"
        "- Only ask follow-up questions when a value is missing or a unit is absent and ambiguous.\n\n"
        "============================================================\n"
        "BEHAVIOR RULES — FOLLOW THESE STRICTLY\n"
        "============================================================\n\n"
        "1) DO NOT call any tools until ALL required parameters are known:\n"
        "   - length_m  (beam length)\n"
        "   - n_elems   (integer)\n"
        "   - E         (Young's modulus)\n"
        "   - NU        (Poisson ratio)\n"
        "   - RHO       (density)\n"
        "   - HY        (cross-section height)\n"
        "   - HZ        (cross-section width)\n"
        "   - FY        (load)\n\n"
        "2) When calling run_full_pipeline, pass units explicitly per field:\n"
        "   - length_unit for length_m\n"
        "   - hy_unit for HY\n"
        "   - hz_unit for HZ\n"
        "   - force_unit for FY\n"
        "   Allowed length units: m, cm, mm. Allowed force units: N, kN.\n\n"
        "3) If ANY required value is missing:\n"
        "   → Ask follow-up questions.\n"
        "   → DO NOT assume defaults unless the user explicitly says: 'Use default values'.\n\n"
        "4) When the user does not know material properties (E, NU, RHO):\n"
        "   → Do NOT call any tools.\n"
        "   → Offer common approximate engineering values for steel, aluminum, or concrete.\n"
        "   → Clearly label them as suggested defaults and ask the user to confirm one set.\n\n"
        "5) When ALL parameters are known:\n"
        "   → Summarize them clearly and preserve the user's units.\n"
        "   → THEN ask:\n"
        '       "Do you confirm running the simulation with these values?"\n'
        "   → NEVER call run_full_pipeline unless the user explicitly confirms with:\n"
        "       'yes', 'confirm', 'ok run', or similar.\n\n"
        "6) If the user responds with 'no' or 'not yet':\n"
        "   → DO NOT call any tools.\n"
        "   → Ask what they want to modify.\n\n"
        "7) FINALIZATION / STOP RULES:\n"
        "   If the user says ANY of these in the conversation:\n"
        "       'stop', 'finish', 'end', 'cancel', 'no more', 'terminate', 'that's enough', 'done'\n"
        "   → Immediately stop all simulation workflow.\n"
        "   → DO NOT call any tools.\n"
        "   → Simply acknowledge and wait for the next input (or let the CLI handle exit).\n"
        "   → Never restart the simulation pipeline unless the user explicitly asks.\n\n"
        "8) NEVER rerun run_full_pipeline automatically.\n"
        "   → A simulation may run only once per explicit confirmation.\n"
        "   → Never repeat runs after the user says it is finished.\n\n"
        "9) If the user gives a geometry number without an explicit unit "
        "(for example 'hy 20' or 'hz 10'):\n"
        "   → Treat it as ambiguous.\n"
        "   → Ask the user to restate that dimension with units.\n"
        "   → Never guess whether it means m, cm, or mm.\n\n"
        "10) Maintain conversation memory and be conversational and helpful, but "
        "follow ALL rules above with 100% strictness.\n\n"
        "============================================================\n"
        "YOUR GOAL\n"
        "============================================================\n"
        "Guide the user step-by-step, help determine missing values, ensure explicit "
        "confirmation before executing any tools, and respect all stop commands."
    )


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
    kept.append("Would you like to do another analysis or finish the process?")
    return "\n".join(kept)


def build_post_run_idle_message() -> str:
    return "Would you like to do another analysis or finish the process?"


def get_sheet_signature(sheet: ParameterSheet) -> str:
    return json.dumps(sheet.to_dict(), sort_keys=True, ensure_ascii=True)


def build_confirmation_summary(sheet: ParameterSheet) -> str:
    values = sheet.to_dict()
    return (
        "All required parameters are available.\n"
        "Current values:\n"
        f"- Length: {values['length_m']} {values.get('length_unit', DEFAULT_INPUT_LENGTH_UNIT)}\n"
        f"- Elements: {values['n_elems']}\n"
        f"- HY: {values['HY']} {values.get('hy_unit', DEFAULT_INPUT_LENGTH_UNIT)}\n"
        f"- HZ: {values['HZ']} {values.get('hz_unit', DEFAULT_INPUT_LENGTH_UNIT)}\n"
        f"- FY: {values['FY']} {values.get('force_unit', DEFAULT_INPUT_FORCE_UNIT)}\n"
        f"- E: {values['E']}\n"
        f"- NU: {values['NU']}\n"
        f"- RHO: {values['RHO']}\n"
        "Reply with 'yes', 'confirm', or 'run it' to start the simulation."
    )


def build_pipeline_kwargs_from_sheet(sheet: ParameterSheet) -> dict[str, Any]:
    values = sheet.to_dict()
    return {
        "length_m": values["length_m"],
        "n_elems": values["n_elems"],
        "E": values["E"],
        "NU": values["NU"],
        "RHO": values["RHO"],
        "HY": values["HY"],
        "HZ": values["HZ"],
        "FY": values["FY"],
        "length_unit": values.get("length_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "hy_unit": values.get("hy_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "hz_unit": values.get("hz_unit", DEFAULT_INPUT_LENGTH_UNIT),
        "force_unit": values.get("force_unit", DEFAULT_INPUT_FORCE_UNIT),
        "confirm": True,
    }


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

    # Expose only the guarded high-level pipeline tool to the agent.
    tools = [
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
    console_print("Please provide geometry with explicit units such as m, cm, mm, N, or kN.")
    console_print(
        "   Example: 5 m beam, 20 cm height, 12 cm width, 1.2 kN load.\n"
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
    console_print("Please provide geometry and load values with explicit units.")
    console_print(
        "   Example: 5 m beam, 20 cm height, 12 cm width, 1.2 kN load.\n"
    )

    chat_history: List[BaseMessage] = []
    parameter_sheet = ParameterSheet()
    simulation_executed = False
    last_executed_signature: str | None = None

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
                    output_str = build_post_success_message(run_output)
                    simulation_executed = True
                    last_executed_signature = get_sheet_signature(parameter_sheet)
                    console_print("\nAgent:\n" + output_str + "\n")
                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=output_str))
                    continue

                output_str = build_confirmation_summary(parameter_sheet)
                console_print("\nAgent:\n" + output_str + "\n")
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=output_str))
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
