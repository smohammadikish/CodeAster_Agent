# ============================================================
#  main_agent_hybrid.py — OpenAI + LangChain 0.3.x Compatible
# ============================================================

import argparse
import json
import logging
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
from env_loader import load_env
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
    filename=os.path.join(WORKSPACE, "hybrid_pipeline_openai.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Hybrid pipeline started (LangChain 0.3.x).")

# ============================================================
#  Default Parameters
# ============================================================
DEFAULT_PARAMS: Dict[str, Any] = {
    "length_m": 3.0,
    "n_elems": 15,
    "E": 2.10e11,
    "NU": 0.3,
    "RHO": 7850.0,
    "HY": 0.12,
    "HZ": 0.10,
    "FY": -1500.0,
}


# ============================================================
#  JSON extraction helpers
# ============================================================
def robust_extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON safely from text."""
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}


def coerce_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure numeric values and apply defaults for missing keys."""
    out = dict(DEFAULT_PARAMS)

    def f(x, default):
        try:
            if isinstance(default, int):
                return int(float(x))
            return float(x)
        except Exception:
            return default

    for key in DEFAULT_PARAMS:
        if key in raw and raw[key] is not None:
            out[key] = f(raw[key], DEFAULT_PARAMS[key])
    return out


# ============================================================
#  Safe tool execution
# ============================================================
def safe_call(tool_func, expected_file=None, **kwargs):
    """Run a LangChain tool safely."""
    print(f"\n🚀 Running: {tool_func.name}")
    logging.info(f"Running {tool_func.name} with args={kwargs}")
    try:
        result = tool_func.invoke(kwargs)
    except Exception as e:
        print(f"❌ Error in {tool_func.name}: {e}")
        logging.exception(f"{tool_func.name} failed")
        return False
    print(result)
    if "Error" in str(result) or "❌" in str(result):
        print(f"⛔ {tool_func.name} failed.")
        return False
    if expected_file and not file_exists(expected_file):
        print(f"⚠️ Expected file missing: {expected_file}")
        return False
    print(f"✅ Step '{tool_func.name}' OK.\n")
    return True


# ============================================================
#  Parameter extraction (non-interactive, one-shot)
# ============================================================
def extract_params_with_openai(user_prompt: str) -> Dict[str, Any]:
    """
    Use OpenAI model (via LangChain 0.3.x) to extract ONLY the parameters
    explicitly mentioned by the user.

    IMPORTANT RULE:
    - The LLM MUST NOT invent values.
    - The LLM MUST NOT return parameters the user didn't mention.
    - Missing parameters will automatically fall back to defaults via `coerce_params()`.
    """
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=openai_key,
        temperature=0,
        max_tokens=800,
        extra_headers={"x-openai-web-search": "true"},
    )

    system_instruction = (
        "You are a mechanical simulation assistant.\n"
        "Extract ONLY the numeric parameters explicitly mentioned by the user.\n"
        "Your output MUST be valid JSON.\n\n"
        "RULES:\n"
        "- DO NOT invent or guess missing values.\n"
        "- DO NOT include keys not mentioned by the user.\n"
        "- If a parameter is not mentioned → OMIT IT.\n"
        "- Output ONLY JSON with no explanation.\n\n"
        "Allowed keys:\n"
        "length_m, n_elems, E, NU, RHO, HY, HZ, FY"
    )

    full_prompt = f"{system_instruction}\n\nUSER REQUEST:\n{user_prompt}"
    print("🧩 Asking ChatGPT for parameters ...")

    response = llm.invoke([HumanMessage(content=full_prompt)])
    text = getattr(response, "content", "") or str(response)
    logging.info("Raw LLM response:\n" + text)

    raw = robust_extract_json(text)
    if not raw:
        print("⚠️ Could not parse JSON — using defaults from DEFAULT_PARAMS.")
        return dict(DEFAULT_PARAMS)

    params = coerce_params(raw)
    print("📦 Parameters going into pipeline:", params)
    return params


# ============================================================
#  INTERACTIVE PARAMETER EXTRACTION (chat mode)
# ============================================================
def _parse_value_with_unit(answer: str, expected_unit: str) -> float:
    """
    Parse a numeric value + unit from user answer.
    expected_unit:
      - 'm'  → meters
      - 'N'  → Newtons (accepts N or kN and converts)
    """
    text = answer.strip().lower()
    # extract numeric part
    m = re.search(r"(-?\d+(\.\d+)?)", text)
    if not m:
        raise ValueError("no numeric value found")
    value = float(m.group(1))

    # extract unit substring (letters)
    u = re.findall(r"[a-zA-Z]+", text)
    unit_str = "".join(u).lower() if u else ""

    if expected_unit == "m":
        # require 'm' in the unit string
        if "m" not in unit_str:
            raise ValueError("unit must be meters (m)")
        return value

    if expected_unit == "N":
        # accept N or kN
        if "kn" in unit_str:
            return value * 1000.0
        if "n" in unit_str:
            return value
        raise ValueError("unit must be N or kN")

    # fallback (should not normally reach)
    return value


def interactive_extract_params():
    """
    Interactive parameter collector:
    - First, user provides a full beam description in one line.
    - LLM extracts ONLY the parameters explicitly mentioned.
    - Missing parameters are then asked one-by-one.
    - Material is checked; if unknown → user chooses manual or web-search.
    - Units are validated.
    """

    required = ["length_m", "n_elems", "HY", "HZ", "FY", "material"]
    known = {}

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    # LLM instance with web-search enabled
    llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=openai_key,
        temperature=0,
        max_tokens=800,
        extra_headers={"x-openai-web-search": "true"},
    )

    # -------------------------------
    # 1) Ask for initial full sentence
    # -------------------------------
    print("🤖 Please describe your beam and loading in one sentence.")
    print(
        "   Example: '6m steel beam, 20 elements, height 0.2m, width 0.1m, load 1500 N'\n"
    )

    first_input = input("👤 You: ").strip()

    # -------------------------------------------
    # 2) Extract known values directly from LLM
    # -------------------------------------------
    extract_prompt = f"""
Extract ONLY the parameters explicitly mentioned by the user.

USER INPUT:
"{first_input}"

Allowed keys:
- length_m
- n_elems
- HY
- HZ
- FY
- material

Rules:
- Do NOT invent any value.
- Do NOT include keys the user did not mention.
- Units MUST NOT be converted; keep numeric values but DO NOT guess units.
- Output JSON only, with no explanation.
"""

    resp = llm.invoke([HumanMessage(content=extract_prompt)])
    raw = robust_extract_json(resp.content)

    if raw:
        for k, v in raw.items():
            known[k] = v

    # -----------------------------------------------------------
    # Helper functions for validating numeric-values + units
    # -----------------------------------------------------------
    def parse_float_with_unit(text, unit="m"):
        t = text.strip().lower()
        t = t.replace("meters", "m").replace("meter", "m")
        if t.endswith(unit):
            try:
                return float(t.replace(unit, "").strip())
            except:
                return None
        return None

    def parse_force(text):
        t = text.lower().strip()
        if "kn" in t:
            try:
                return float(t.replace("kn", "").strip()) * 1000.0
            except:
                return None
        if "n" in t:
            try:
                return float(t.replace("n", "").strip())
            except:
                return None
        return None

    # -----------------------------------------------------------
    # 3) Now fill missing parameters by asking user interactively
    # -----------------------------------------------------------
    while True:
        missing = [p for p in required if p not in known]

        if not missing:
            # ========== MATERIAL CHECK ==========
            material = known.get("material", "").lower()

            if material and material not in ["steel", "aluminum", "aluminium"]:
                print(f"\n⚠ Material '{material}' is not in the local database.")

                choice = (
                    input(
                        "❓ Enter material properties manually or search online? (manual/search): "
                    )
                    .strip()
                    .lower()
                )

                if choice.startswith("s"):
                    # ---------------------------
                    # Online search query
                    # ---------------------------
                    query = (
                        f"Search the web and retrieve accurate standard mechanical properties "
                        f"for {material}. Return Young's modulus E (Pa), Poisson ratio NU, "
                        f"and density RHO (kg/m^3)."
                    )
                    print("\n🌐 Searching online…")

                    resp = llm.invoke([HumanMessage(content=query)])
                    print("\n🔎 Search result summary:\n", resp.content)

                    print("\nPlease extract numeric values from the above result:")
                    known["E"] = float(input("Enter E (Pa): "))
                    known["NU"] = float(input("Enter NU: "))
                    known["RHO"] = float(input("Enter RHO (kg/m³): "))

                else:
                    # Manual entry
                    known["E"] = float(input("Enter E (Pa): "))
                    known["NU"] = float(input("Enter NU: "))
                    known["RHO"] = float(input("Enter RHO (kg/m³): "))

            print("\n🧩 Final parameters (raw):", known)
            return known

        # --------------------------
        # Ask for next missing value
        # --------------------------
        next_param = missing[0]

        # LENGTH
        if next_param == "length_m":
            print("🤖 What is the beam length in meters?")
            ans = input("👤 You: ")
            val = parse_float_with_unit(ans, "m")
            if val is None:
                print("⚠ Length must be in meters (e.g., '3.5 m')")
                continue
            known["length_m"] = val
            continue

        # ELEMENT COUNT
        if next_param == "n_elems":
            print("🤖 What is the number of elements along the beam (integer)?")
            ans = input("👤 You: ").strip()
            if not ans.isdigit():
                print("⚠ Must be an integer (e.g., '25')")
                continue
            known["n_elems"] = int(ans)
            continue

        # HEIGHT
        if next_param == "HY":
            print("🤖 What is the beam cross-section height HY in meters?")
            ans = input("👤 You: ")
            val = parse_float_with_unit(ans, "m")
            if val is None:
                print("⚠ Must be in meters (e.g., '0.2 m')")
                continue
            known["HY"] = val
            continue

        # WIDTH
        if next_param == "HZ":
            print("🤖 What is the beam cross-section width HZ in meters?")
            ans = input("👤 You: ")
            val = parse_float_with_unit(ans, "m")
            if val is None:
                print("⚠ Must be in meters (e.g., '0.1 m')")
                continue
            known["HZ"] = val
            continue

        # FORCE
        if next_param == "FY":
            print("🤖 What is the vertical load FY (N or kN)?")
            ans = input("👤 You: ")
            val = parse_force(ans)
            if val is None:
                print("⚠ Must be like '1500 N' or '1.5 kN'")
                continue
            known["FY"] = val
            continue

        # MATERIAL
        if next_param == "material":
            print("🤖 What is the beam material?")
            ans = input("👤 You: ").strip().lower()
            known["material"] = ans
            continue


# ============================================================
#  Deterministic pipeline
# ============================================================
def run_hybrid_pipeline(params: Dict[str, Any]) -> None:
    print("\n⚙️ Starting simulation pipeline...\n")

    if not safe_call(
        generate_geo_tool,
        os.path.join(WORKSPACE, "model.geo"),
        length_m=params["length_m"],
        n_elems=params["n_elems"],
        geo_name="model.geo",
    ):
        return

    if not safe_call(
        mesh_with_gmsh_tool,
        os.path.join(WORKSPACE, "model.med"),
        geo_name="model.geo",
        med_name="model.med",
    ):
        return

    if not safe_call(
        generate_comm_tool,
        os.path.join(WORKSPACE, "model.comm"),
        med_name="model.med",
        comm_name="model.comm",
        E=params["E"],
        NU=params["NU"],
        RHO=params["RHO"],
        HY=params["HY"],
        HZ=params["HZ"],
        FY=params["FY"],
    ):
        return

    if not safe_call(
        generate_export_tool,
        os.path.join(WORKSPACE, "model.export"),
        comm_name="model.comm",
        export_name="model.export",
        job_name="beam_job",
        med_name="model.med",
    ):
        return

    if not safe_call(
        run_code_aster_tool,
        os.path.join(WORKSPACE, "beam_job.mess"),
        export_name="model.export",
    ):
        return

    print("\n📊 Visualization phase ...")
    safe_call(
        visualize_resu_tool,
        os.path.join(WORKSPACE, "beam_job.resu"),
        resu_name="beam_job.resu",
    )
    safe_call(
        visualize_stress_tool,
        os.path.join(WORKSPACE, "tab_sie.txt"),
        stress_name="tab_sie.txt",
    )
    safe_call(
        visualize_mfz_contour_tool,
        os.path.join(WORKSPACE, "tab_sie.txt"),
        stress_name="tab_sie.txt",
    )
    safe_call(
        visualize_freebody_tool,
        os.path.join(WORKSPACE, "tab_reac.txt"),
        reac_name="tab_reac.txt",
    )

    print("\n🎯 All steps completed successfully — Simulation finished!\n")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Agent Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["agent", "tool", "chat"],
        help="agent: one-shot NL prompt; tool: run with DEFAULT_PARAMS; chat: interactive Q&A with LLM",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="User natural language request (example: 'simulate a 4m beam with 25 elements and 2000 N')",
    )

    args = parser.parse_args()

    if args.mode == "agent":
        print(f"\n🟢 AGENT MODE STARTED")
        print(f"🔹 Input: {args.input}\n")
        params = extract_params_with_openai(args.input or "")
        run_hybrid_pipeline(params)

    elif args.mode == "tool":
        print("\n🟡 TOOL MODE EXECUTION (using DEFAULT_PARAMS)\n")
        run_hybrid_pipeline(DEFAULT_PARAMS)

    elif args.mode == "chat":
        print("\n🟢 INTERACTIVE CHAT MODE")
        # 1) interactive collection (no defaults)
        raw_params = interactive_extract_params()

        # 2) derive E, NU, RHO from material or defaults
        material = raw_params.get("material", "default").lower()

        if material == "steel":
            E = 210e9
            NU = 0.3
            RHO = 7850.0
        elif material in ("aluminium", "aluminum"):
            E = 69e9
            NU = 0.33
            RHO = 2700.0
        else:
            # fallback to defaults for unknown material
            E = DEFAULT_PARAMS["E"]
            NU = DEFAULT_PARAMS["NU"]
            RHO = DEFAULT_PARAMS["RHO"]

        # 3) build full params dict for pipeline
        full_params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        full_params.update(
            {
                "length_m": float(raw_params["length_m"]),
                "n_elems": int(raw_params["n_elems"]),
                "HY": float(raw_params["HY"]),
                "HZ": float(raw_params["HZ"]),
                "FY": float(raw_params["FY"]),
                "E": E,
                "NU": NU,
                "RHO": RHO,
            }
        )

        print("📦 Final parameters going into pipeline:", full_params)
        run_hybrid_pipeline(full_params)
