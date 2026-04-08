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
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.validators import file_exists
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)

# ============================================================
#  Setup
# ============================================================
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)
gmsh_bin, as_run_bin, _, openai_key, openai_model = load_env()

logging.basicConfig(
    filename=os.path.join(WORKSPACE, "hybrid_pipeline_agentic.log"),
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


def safe_call(tool_func, expected_file=None, **kwargs) -> bool:
    """
    Safely run a LangChain tool used in the deterministic pipeline.

    - Prints which tool is running.
    - Logs the call and errors.
    - Optionally checks for an expected output file.
    """
    console_print(f"\nRunning: {tool_func.name}")
    logging.info(f"Running {tool_func.name} with args={kwargs}")
    try:
        # All these are LangChain tools, so we call invoke()
        result = tool_func.invoke(kwargs)
    except Exception as e:
        console_print(f"Error in {tool_func.name}: {e}")
        logging.exception(f"{tool_func.name} failed")
        return False

    console_print(result)

    if "Error" in str(result) or "❌" in str(result):
        console_print(f"{tool_func.name} failed.")
        return False

    if expected_file and not file_exists(expected_file):
        console_print(f"Expected file missing: {expected_file}")
        return False

    console_print(f"Step '{tool_func.name}' OK.\n")
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
    confirm: bool = False,
) -> str:
    """
    Run the complete simulation pipeline ONLY after explicit user confirmation.

    IMPORTANT UNITS (FOR THE USER / LLM SIDE):
    - length_m: beam length in meters (m)
    - HY: section height in meters (m)
    - HZ: section width in meters (m)
    - FY: load in Newtons (N), positive or negative (negative = downward)
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
    }

    missing = [name for name, val in params_raw.items() if val is None]

    if missing:
        return (
            "Missing required parameters: "
            + ", ".join(missing)
            + "\nPlease ask the user to provide these values in **meters (m)** for all dimensions, "
            "or explicitly approve using default values."
        )

    # 2) Check for confirmation flag
    if confirm is False:
        return (
            "All parameters are provided, but the simulation has NOT been confirmed.\n"
            "Ask the user: 'Do you confirm running the full simulation with these values (all dimensions in meters)?'"
        )

    # 3) All good → run pipeline
    params = coerce_params(params_raw)
    logging.info(
        "run_full_pipeline confirmed with params (in meters):\n"
        + dict_to_pretty_json(params)
    )

    summary = run_hybrid_pipeline(params)

    return (
        f"{summary}\n\n"
        "Used parameters (geometry in meters as seen by the user):\n"
        f"{dict_to_pretty_json(params)}"
    )


# ============================================================
#  Agent construction
# ============================================================


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
                (
                    "You are a mechanical simulation assistant specialized in beam analysis "
                    "using Code_Aster and Gmsh. You support multi-step conversations where "
                    "the user incrementally provides beam geometry, material properties, "
                    "cross-section dimensions, and loads.\n\n"
                    "============================================================\n"
                    "UNITS CONVENTION (CRITICAL)\n"
                    "============================================================\n"
                    "- All dimensions that the USER provides MUST be in **meters (m)**.\n"
                    "- When you summarize or discuss parameters with the user, ALWAYS keep them in meters.\n"
                    "- The internal deterministic pipeline will convert meters to millimeters for the tools.\n"
                    "- DO NOT convert meters to millimeters yourself when talking to the user.\n"
                    "- DO NOT show millimeters to the user. Stay in meters at the conversation level.\n\n"
                    "============================================================\n"
                    "BEHAVIOR RULES — FOLLOW THESE STRICTLY\n"
                    "============================================================\n\n"
                    "1) DO NOT call any tools until ALL required parameters are known:\n"
                    "   - length_m  (beam length in meters)\n"
                    "   - n_elems   (integer)\n"
                    "   - E         (Young's modulus)\n"
                    "   - NU        (Poisson ratio)\n"
                    "   - RHO       (density)\n"
                    "   - HY        (cross-section height in meters)\n"
                    "   - HZ        (cross-section width in meters)\n"
                    "   - FY        (load in Newtons)\n\n"
                    "2) If ANY of these parameters are missing:\n"
                    "   → Ask follow-up questions.\n"
                    "   → DO NOT assume defaults unless the user explicitly says: 'Use default values'.\n\n"
                    "3) When the user does not know material properties (E, NU, RHO):\n"
                    "   → Do NOT call any tools.\n"
                    "   → Offer common approximate engineering values for steel, aluminum, or concrete.\n"
                    "   → Clearly label them as suggested defaults and ask the user to confirm one set.\n\n"
                    "4) When ALL parameters are known:\n"
                    "   → ALWAYS summarize them clearly in meters and Newtons.\n"
                    "   → THEN ask:\n"
                    '       "Do you confirm running the simulation with these values?"\n'
                    "   → NEVER call run_full_pipeline unless the user explicitly confirms with:\n"
                    "       'yes', 'confirm', 'ok run', or similar.\n\n"
                    "5) If the user responds with 'no' or 'not yet':\n"
                    "   → DO NOT call any tools.\n"
                    "   → Ask what they want to modify.\n\n"
                    "6) FINALIZATION / STOP RULES:\n"
                    "   If the user says ANY of these in the conversation:\n"
                    "       'stop', 'finish', 'end', 'cancel', 'no more', 'terminate', 'that's enough', 'done'\n"
                    "   → Immediately stop all simulation workflow.\n"
                    "   → DO NOT call any tools.\n"
                    "   → Simply acknowledge and wait for the next input (or let the CLI handle exit).\n"
                    "   → Never restart the simulation pipeline unless the user explicitly asks.\n\n"
                    "7) NEVER rerun run_full_pipeline automatically.\n"
                    "   → A simulation may run only once per explicit confirmation.\n"
                    "   → Never repeat runs after the user says it is finished.\n\n"
                    "8) If the user gives a geometry number without an explicit unit "
                    "(for example 'hy 20' or 'hz 10'):\n"
                    "   → Treat it as ambiguous.\n"
                    "   → Ask the user to restate that dimension with units.\n"
                    "   → Never guess whether it means m, cm, or mm.\n\n"
                    "9) Maintain conversation memory and be conversational and helpful, but "
                    "follow ALL rules above with 100% strictness.\n\n"
                    "============================================================\n"
                    "YOUR GOAL\n"
                    "============================================================\n"
                    "Guide the user step-by-step, help determine missing values (including via web "
                    "search), ensure explicit confirmation before executing any tools, and respect "
                    "all stop commands."
                ),
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


# ============================================================
#  CLI / Entry point
# ============================================================


def run_agent_one_shot(agent_executor: AgentExecutor, user_input: str) -> None:
    """
    One-shot execution (mode=agent): LLM decides which tools to use.

    NOTE TO THE USER (printed once before running):
    - All beam dimensions MUST be provided in meters (m).
      The system converts them internally to millimeters (mm) for the tools.
    """
    console_print("\nAGENT MODE (one-shot with tools)\n")
    console_print("Please provide ALL geometry dimensions in **meters (m)**.")
    console_print(
        "   Internally, they will be converted to millimeters (mm) for the simulation.\n"
    )

    ambiguous_message = get_ambiguous_dimension_message(user_input)
    if ambiguous_message:
        console_print("\nAgent response:\n")
        console_print(ambiguous_message)
        return

    try:
        result = agent_executor.invoke({"input": user_input})
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


def run_agent_chat(agent_executor: AgentExecutor) -> None:
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
    console_print("Please provide ALL geometry dimensions in **meters (m)**.")
    console_print(
        "   Internally, they will be converted to millimeters (mm) for the tools.\n"
    )

    chat_history: List[BaseMessage] = []

    from langchain_core.messages import AIMessage, HumanMessage

    exit_commands = {"exit", "quit", "stop", "end", "finish", "cancel"}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console_print("\nBye!")
            break

        if user_input.lower() in exit_commands:
            console_print("\nBye! Chat session terminated.\n")
            break

        ambiguous_message = get_ambiguous_dimension_message(user_input)
        if ambiguous_message:
            console_print("\nAgent:\n" + ambiguous_message + "\n")
            continue

        # Run the agent with history
        try:
            result = agent_executor.invoke(
                {
                    "input": user_input,
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

    if args.mode == "agent":
        if not args.input:
            raise ValueError("--input is required in agent mode.")
        run_agent_one_shot(agent_executor, args.input)

    elif args.mode == "chat":
        run_agent_chat(agent_executor)
