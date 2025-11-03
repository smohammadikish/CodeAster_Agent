# ----------------------------
#  main_agent.py
# ----------------------------
import argparse
import logging
import os

from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY, etc.)
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
#  Global setup
# ============================================================
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

gmsh_bin, as_run_bin, _ = load_env()

# Logging setup
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started.")


# ============================================================
#  Helper function
# ============================================================
def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """Safely execute a tool and verify expected file existence."""
    print(f"\n🚀 Running: {tool_func.name}")
    try:
        result = tool_func.invoke(kwargs)
    except Exception as e:
        print(f"❌ Error running {tool_func.name}: {e}")
        return False

    print(result)
    logging.info(f"Executed tool: {tool_func.name}")
    logging.info(result)

    if "Error" in str(result) or "❌" in str(result):
        print(f"❌ {tool_func.name} failed — stopping pipeline.")
        return False

    if expected_file and not file_exists(expected_file):
        print(f"⚠️ Expected file not found: {expected_file}")
        return False

    print(f"✅ Step '{tool_func.name}' completed successfully.\n")
    return True


# ============================================================
#  Manual Mode
# ============================================================
def run_pipeline():
    """Execute full Code_Aster pipeline manually."""
    print("\n=== 🧠 Starting Code_Aster pipeline (manual mode) ===\n")

    # --- Preprocessing ---
    if not safe_call(
        generate_geo_tool, expected_file=os.path.join(WORKSPACE, "model.geo")
    ):
        return
    if not safe_call(
        mesh_with_gmsh_tool, expected_file=os.path.join(WORKSPACE, "model.med")
    ):
        return
    if not safe_call(
        generate_comm_tool, expected_file=os.path.join(WORKSPACE, "model.comm")
    ):
        return
    if not safe_call(
        generate_export_tool, expected_file=os.path.join(WORKSPACE, "model.export")
    ):
        return

    # --- Run Code_Aster ---
    if not safe_call(
        run_code_aster_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.mess")
    ):
        return

    # --- Visualization ---
    print("\n=== 📊 Visualizing Results ===")

    vis_tools = [
        (
            "Deflection/Rotation",
            visualize_resu_tool,
            {"resu_name": "beam_analysis.resu"},
        ),
        ("Internal Forces", visualize_stress_tool, {"stress_name": "tab_sie.txt"}),
        ("Moment Contour", visualize_mfz_contour_tool, {"stress_name": "tab_sie.txt"}),
        ("Free Body", visualize_freebody_tool, {"reac_name": "tab_reac.txt"}),
    ]

    for title, func, args in vis_tools:
        print(f"\n🖼 {title}")
        try:
            print(func.invoke(args))
        except Exception as e:
            print(f"⚠️ Visualization '{title}' failed: {e}")

    print("\n🎯 Pipeline finished successfully!")


# ============================================================
#  Agent Mode (Gemini + LangGraph)
# ============================================================
def run_agent_mode():
    """Run the Code_Aster agent built with LangGraph + Gemini."""
    from langchain_core.messages import HumanMessage

    from agents.codeaster_agentـopenai import build_codeaster_agent_gemini

    print("\n=== 🤖 Running Code_Aster Agent (Gemini) ===\n")

    # Initialize agent
    agent = build_codeaster_agent_gemini()
    user_prompt = (
        "Generate a 3-meter cantilever beam, mesh it, run Code_Aster, "
        "and visualize deflection, shear, and moment diagrams."
    )

    print(f"👤 User: {user_prompt}\n")
    response = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    print("🤖 Agent Response:\n", response)


# ============================================================
#  Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "manual"], default="agent")
    args = parser.parse_args()

    if args.mode == "agent":
        run_agent_mode()
    else:
        run_pipeline()
