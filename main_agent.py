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
from typing import Any, Dict

# ----------------------------
# Load environment
# ----------------------------
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# LangChain imports
from langchain_openai import ChatOpenAI

from env_loader import load_env  # noqa: E402
from input_parser import ParameterSheet, build_state_aware_input
from storage import (create_run_id, ensure_base_dirs, ensure_run_dirs,
                     finalize_run, record_artifact,
                     record_tool_event, register_run, resolve_workspace_file,
                     utc_now_iso)
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.validators import file_exists
from tools.visualize_tool import visualize_stress_tool
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
SCENARIO_CANTILEVER_TIP_LOAD = "cantilever_tip_load"
SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD = "simply_supported_midpoint_load"
ALLOWED_SCENARIOS = (
    SCENARIO_CANTILEVER_TIP_LOAD,
    SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD,
)
ENABLED_SCENARIOS = {
    SCENARIO_CANTILEVER_TIP_LOAD,
    SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD,
}


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


def extract_scenario_from_text(user_input: str) -> str | None:
    text = user_input.strip().lower()
    for scenario in ALLOWED_SCENARIOS:
        if scenario in text:
            return scenario
    if re.search(r"\bcantilever(?:ed)?\b", text):
        return SCENARIO_CANTILEVER_TIP_LOAD
    if re.search(r"\b(simply supported|simple beam|simple supported|supported beam)\b", text):
        return SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD
    return None


def get_sheet_scenario(sheet: ParameterSheet) -> str | None:
    scenario = sheet.to_dict().get("scenario")
    return str(scenario) if scenario else None


def _tool_result_indicates_failure(result: Any) -> bool:
    text = str(result)
    text_upper = text.upper()
    if "ERROR" in text_upper or "âŒ" in text:
        return True
    if "ABNORMAL_ABORT" in text_upper or "<F>" in text_upper:
        return True
    exit_code_match = re.search(r"EXIT_CODE\s*=\s*(-?\d+)", text_upper)
    if exit_code_match and int(exit_code_match.group(1)) != 0:
        return True
    if "FAILED" in text_upper:
        return True
    if "FATAL" in text_upper:
        return True
    if "TRACEBACK" in text_upper:
        return True
    return False


def validate_cantilever_tip_load_inputs(params: dict[str, Any]) -> str | None:
    has_resolved_section = all(
        params.get(key) is not None
        for key in ("A_mm2", "IY_mm4", "IZ_mm4", "JX_mm4", "AY_mm2", "AZ_mm2")
    )
    validators = [
        ("length_m", params.get("length_m"), lambda value: value > 0, "a positive value in meters"),
        (
            "n_elems",
            params.get("n_elems"),
            lambda value: isinstance(value, int) and value > 0,
            "a positive integer",
        ),
        ("E", params.get("E"), lambda value: value > 0, "a positive value"),
        ("NU", params.get("NU"), lambda value: 0 < value < 0.5, "a value strictly between 0 and 0.5"),
        ("RHO", params.get("RHO"), lambda value: value > 0, "a positive value"),
        ("FY", params.get("FY"), lambda value: value != 0, "a non-zero value"),
    ]
    for field_name, value, predicate, expectation in validators:
        if value is None or not predicate(value):
            return f"Invalid value for {field_name}: {value}. Expected {expectation}."
    if not has_resolved_section:
        section_dimension_validators = [
            ("HY", params.get("HY"), lambda value: value is not None and value > 0, "a positive value in meters"),
            ("HZ", params.get("HZ"), lambda value: value is not None and value > 0, "a positive value in meters"),
        ]
        for field_name, value, predicate, expectation in section_dimension_validators:
            if value is None or not predicate(value):
                return f"Invalid value for {field_name}: {value}. Expected {expectation}."
    return None


def validate_simply_supported_midpoint_load_inputs(
    params: dict[str, Any],
) -> str | None:
    n_elems = params.get("n_elems")
    if not isinstance(n_elems, int) or n_elems <= 0:
        return f"Invalid value for n_elems: {n_elems}. Expected a positive even integer."
    if n_elems % 2 != 0:
        return (
            f"Invalid value for n_elems: {n_elems}. "
            "Expected a positive even integer so the beam has a midpoint node."
        )
    return None


def get_scenario_meaning(scenario: str | None) -> str:
    if scenario == SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD:
        return (
            "simply supported beam, pin at one end, roller at the other, "
            "with a point load at midspan"
        )
    return "cantilever beam, fixed at one end, with a point load at the free tip"


def _step_result(
    ok: bool,
    step: str,
    message: str,
    raw_result: Any = "",
    expected_file: str | None = None,
    file_exists: bool | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "step": step,
        "message": message,
        "expected_file": expected_file,
        "file_exists": file_exists,
        "raw_result": str(raw_result),
    }


def _pipeline_failure_message(
    result: dict[str, Any],
    skip_visualization: bool = False,
) -> str:
    lines = [
        f"Pipeline failed at step: {result['step']}",
        f"Reason: {result['message']}",
    ]
    if result.get("expected_file"):
        lines.append(f"Expected file: {result['expected_file']}")
    if skip_visualization:
        lines.append(
            "Visualization skipped because the run did not complete successfully or expected result files are missing."
        )
    return "\n".join(lines)


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


def safe_call(tool_func, run_id: str, expected_artifacts=None, **kwargs) -> dict[str, Any]:
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
        step_result = _step_result(False, tool_func.name, str(e), raw_result=str(e))
        record_tool_event(
            run_id,
            tool_func.name,
            "failed",
            started_at,
            utc_now_iso(),
            str(e),
            kwargs,
        )
        return step_result

    console_print(result)

    if _tool_result_indicates_failure(result):
        console_print(f"{tool_func.name} failed.")
        step_result = _step_result(
            False,
            tool_func.name,
            "Tool output indicates failure.",
            raw_result=result,
        )
        record_tool_event(
            run_id,
            tool_func.name,
            "failed",
            started_at,
            utc_now_iso(),
            str(result),
            kwargs,
        )
        return step_result

    for artifact_type, artifact_stage, artifact_path in expected_artifacts:
        if not file_exists(str(artifact_path)):
            console_print(f"Expected file missing: {artifact_path}")
            step_result = _step_result(
                False,
                tool_func.name,
                "Expected output file is missing.",
                raw_result=result,
                expected_file=str(artifact_path),
                file_exists=False,
            )
            record_tool_event(
                run_id,
                tool_func.name,
                "failed",
                started_at,
                utc_now_iso(),
                f"Expected file missing: {artifact_path}",
                kwargs,
            )
            return step_result
        record_artifact(run_id, artifact_type, artifact_stage, str(artifact_path))

    console_print(f"Step '{tool_func.name}' OK.\n")
    expected_file = str(expected_artifacts[0][2]) if expected_artifacts else None
    record_tool_event(
        run_id,
        tool_func.name,
        "completed",
        started_at,
        utc_now_iso(),
        str(result),
        kwargs,
    )
    return _step_result(
        True,
        tool_func.name,
        "Step completed successfully.",
        raw_result=result,
        expected_file=expected_file,
        file_exists=True if expected_file else None,
    )


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
    step_results: list[dict[str, Any]] = []

    # -------------------------------------------------
    # GEO generation (uses millimeters internally)
    # -------------------------------------------------
    geo_result = safe_call(
        generate_geo_tool,
        "",
        expected_artifacts=[
            ("geo", "workspace", resolve_workspace_file("model.geo", "inputs"))
        ],
        scenario=params.get("scenario", SCENARIO_CANTILEVER_TIP_LOAD),
        length_mm=p["length_m"],  # NOTE: now explicitly mm
        n_elems=p["n_elems"],
        geo_name="model.geo",
    )
    step_results.append(geo_result)
    if not geo_result["ok"]:
        return _pipeline_failure_message(geo_result)

    # -------------------------------------------------
    # Meshing with Gmsh (still mm)
    # -------------------------------------------------
    mesh_result = safe_call(
        mesh_with_gmsh_tool,
        "",
        expected_artifacts=[
            ("med", "workspace", resolve_workspace_file("model.med", "inputs"))
        ],
        geo_name="model.geo",
        med_name="model.med",
    )
    step_results.append(mesh_result)
    if not mesh_result["ok"]:
        return _pipeline_failure_message(mesh_result)

    # -------------------------------------------------
    # COMM generation for Code_Aster (geometry in mm)
    # -------------------------------------------------
    comm_result = safe_call(
        generate_comm_tool,
        "",
        expected_artifacts=[
            ("comm", "workspace", resolve_workspace_file("model.comm", "inputs"))
        ],
        scenario=params.get("scenario", SCENARIO_CANTILEVER_TIP_LOAD),
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
    )
    step_results.append(comm_result)
    if not comm_result["ok"]:
        return _pipeline_failure_message(comm_result)

    # -------------------------------------------------
    # EXPORT file generation
    # -------------------------------------------------
    export_result = safe_call(
        generate_export_tool,
        "",
        expected_artifacts=[
            ("export", "workspace", resolve_workspace_file("model.export", "inputs"))
        ],
        comm_name="model.comm",
        export_name="model.export",
        job_name="beam_job",
        med_name="model.med",
    )
    step_results.append(export_result)
    if not export_result["ok"]:
        return _pipeline_failure_message(export_result)

    # -------------------------------------------------
    # Run Code_Aster
    # -------------------------------------------------
    solver_result = safe_call(
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
    )
    step_results.append(solver_result)
    if not solver_result["ok"]:
        return _pipeline_failure_message(solver_result, skip_visualization=True)

    # -------------------------------------------------
    # Visualization phase
    # -------------------------------------------------
    console_print("\nVisualization phase ...")
    visualization_failures = []

    visualize_stress_result = safe_call(
        visualize_stress_tool,
        "",
        expected_artifacts=[
            ("shear_plot", "workspace", resolve_workspace_file("shear_diagram.png", "viz")),
            ("moment_plot", "workspace", resolve_workspace_file("moment_diagram.png", "viz")),
        ],
        stress_name="tab_sie.txt",
    )
    step_results.append(visualize_stress_result)
    if not visualize_stress_result["ok"]:
        visualization_failures.append("visualize_stress_tool")

    if visualization_failures:
        return (
            "Simulation completed successfully, but some visualization steps failed: "
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
        step_results: list[dict[str, Any]] = []

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
            ("shear_plot", "viz", resolve_workspace_file("shear_diagram.png", "viz", run_id)),
            ("moment_plot", "viz", resolve_workspace_file("moment_diagram.png", "viz", run_id)),
        ]

        geo_result = safe_call(
            generate_geo_tool,
            run_id,
            expected_artifacts=[inputs[0]],
            scenario=params.get("scenario", SCENARIO_CANTILEVER_TIP_LOAD),
            length_mm=p["length_m"],
            n_elems=p["n_elems"],
            geo_name="model.geo",
        )
        step_results.append(geo_result)
        if not geo_result["ok"]:
            failure_message = _pipeline_failure_message(geo_result)
            finalize_run(run_id, "failed", failure_message)
            return f"{failure_message}\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"

        mesh_result = safe_call(
            mesh_with_gmsh_tool,
            run_id,
            expected_artifacts=[inputs[1]],
            geo_name="model.geo",
            med_name="model.med",
        )
        step_results.append(mesh_result)
        if not mesh_result["ok"]:
            failure_message = _pipeline_failure_message(mesh_result)
            finalize_run(run_id, "failed", failure_message)
            return f"{failure_message}\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"

        comm_result = safe_call(
            generate_comm_tool,
            run_id,
            expected_artifacts=[inputs[2]],
            scenario=params.get("scenario", SCENARIO_CANTILEVER_TIP_LOAD),
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
        )
        step_results.append(comm_result)
        if not comm_result["ok"]:
            failure_message = _pipeline_failure_message(comm_result)
            finalize_run(run_id, "failed", failure_message)
            return f"{failure_message}\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"

        export_result = safe_call(
            generate_export_tool,
            run_id,
            expected_artifacts=[inputs[3]],
            comm_name="model.comm",
            export_name="model.export",
            job_name="beam_job",
            med_name="model.med",
        )
        step_results.append(export_result)
        if not export_result["ok"]:
            failure_message = _pipeline_failure_message(export_result)
            finalize_run(run_id, "failed", failure_message)
            return f"{failure_message}\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"

        solver_result = safe_call(
            run_code_aster_tool,
            run_id,
            expected_artifacts=outputs,
            export_name="model.export",
        )
        step_results.append(solver_result)
        if not solver_result["ok"]:
            failure_message = _pipeline_failure_message(
                solver_result,
                skip_visualization=True,
            )
            finalize_run(run_id, "failed", failure_message)
            return f"{failure_message}\nRun ID: {run_id}\nRun folder: {run_paths.root.resolve()}"

        console_print("\nVisualization phase ...")
        visualization_failures = []

        visualize_stress_result = safe_call(
            visualize_stress_tool,
            run_id,
            expected_artifacts=viz,
            stress_name="tab_sie.txt",
        )
        step_results.append(visualize_stress_result)
        if not visualize_stress_result["ok"]:
            visualization_failures.append("visualize_stress_tool")

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
                "Simulation completed successfully, but some visualization steps failed: "
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
    scenario: str = None,
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
) -> str:
    """
    Run the complete simulation pipeline when required inputs are available.

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
    - If any required parameter is missing, return an instructional message.
    - If all parameters exist, run the deterministic pipeline.
    """

    if scenario is None:
        return (
            "Missing required parameter: scenario.\n"
            "Please ask the user to choose one of these scenarios:\n"
            f"- {SCENARIO_CANTILEVER_TIP_LOAD}\n"
            f"- {SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD}"
        )

    if scenario not in ALLOWED_SCENARIOS:
        return (
            f"Invalid scenario '{scenario}'.\n"
            "Allowed scenario names are:\n"
            f"- {SCENARIO_CANTILEVER_TIP_LOAD}\n"
            f"- {SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD}"
        )

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

    if scenario == SCENARIO_CANTILEVER_TIP_LOAD:
        validation_message = validate_cantilever_tip_load_inputs(
            {
                "length_m": length_m,
                "n_elems": n_elems,
                "HY": HY,
                "HZ": HZ,
                "A_mm2": A_mm2,
                "IY_mm4": IY_mm4,
                "IZ_mm4": IZ_mm4,
                "JX_mm4": JX_mm4,
                "AY_mm2": AY_mm2,
                "AZ_mm2": AZ_mm2,
                "E": E,
                "NU": NU,
                "RHO": RHO,
                "FY": FY,
            }
        )
        if validation_message:
            return validation_message
    elif scenario == SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD:
        validation_message = validate_simply_supported_midpoint_load_inputs(
            {
                "n_elems": n_elems,
            }
        )
        if validation_message:
            return validation_message

    # 2) All good -> run pipeline
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
    params["scenario"] = scenario
    logging.info(
        f"run_full_pipeline started for scenario={scenario} with params (user units):\n"
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
        f"Scenario: {scenario}\n"
        f"Input units: length={length_unit}, HY={hy_unit}, HZ={hz_unit}, force={force_unit}\n"
        "Used parameters:\n"
        f"{dict_to_pretty_json(params)}"
    )


# ============================================================
#  Agent construction
# ============================================================


def build_system_prompt() -> str:
    return """You are a mechanical simulation assistant specialized in beam analysis using Code_Aster and Gmsh.

You support exactly these two scenario names:
- cantilever_tip_load
- simply_supported_midpoint_load

Do not call run_full_pipeline until all required inputs are known:
- scenario
- length_m
- n_elems
- E, NU, RHO
- section definition (A/IY/IZ/JX/AY/AZ, or HY and HZ)
- FY

When calling run_full_pipeline, pass explicit units:
- length_unit for length_m
- hy_unit for HY
- hz_unit for HZ
- force_unit for FY

If a required value is missing, ask only for the missing value. Keep responses compact.
Never request confirmation before running."""


def is_exit_input(user_input: str) -> bool:
    text = user_input.strip().lower()
    return bool(
        re.search(
            r"\b(exit|quit|stop|end|finish|cancel|bye|goodbye|ok bye|thanks|thank you|ok thanks)\b",
            text,
        )
    )


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

    # Expose only the high-level deterministic execution tool.
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
    console_print(
        "Supported scenarios: "
        f"{SCENARIO_CANTILEVER_TIP_LOAD}, {SCENARIO_SIMPLY_SUPPORTED_MIDPOINT_LOAD}."
    )
    console_print("Both supported scenarios can proceed through execution.")
    console_print("All geometric dimensions must be provided in meters.")
    console_print("Describe the beam with the values you know, and include units for each dimension or load.")
    console_print(
        "   Example: cantilever beam, 5 m length, 12 elements, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 0.2 m, HZ 0.1 m, 1.2 kN tip load.\n"
    )

    ambiguous_message = get_ambiguous_dimension_message(user_input)
    if ambiguous_message:
        console_print("\nAgent response:\n")
        console_print(ambiguous_message)
        return

    try:
        sheet = ParameterSheet()
        turn_update = sheet.update_from_text(user_input, llm=extractor_llm)
        state_input = build_state_aware_input(
            user_input,
            sheet,
            turn_update=turn_update,
        )
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


def print_agent_input_help() -> None:
    console_print("\nAGENT MODE INPUT GUIDE\n")
    console_print("Required core parameters:")
    console_print(
        "- scenario: cantilever_tip_load or simply_supported_midpoint_load"
    )
    console_print("- length_m with length_unit (m/cm/mm)")
    console_print("- n_elems")
    console_print("- material: E, NU, RHO")
    console_print("- section: HY/HZ (or A_mm2, IY_mm4, IZ_mm4, JX_mm4, AY_mm2, AZ_mm2)")
    console_print("- FY with force_unit (N/kN)")
    console_print("\nExample:")
    console_print(
        'cantilever beam, length 4 m, 25 elements, E 210e9 Pa, NU 0.3, '
        'RHO 7850 kg/m3, HY 0.2 m, HZ 0.1 m, load 2000 N'
    )
    console_print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Beam Simulation Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["agent"],
        help="agent: one-shot NL prompt with tools",
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
            print_agent_input_help()
            sys.exit(0)
        run_agent_one_shot(agent_executor, args.input, extractor_llm=extractor_llm)
