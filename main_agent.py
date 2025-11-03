# main_agent.py — Compatible with LangChain 0.2.14 & langchain-ollama 0.1.1

import argparse
import logging
import os

from env_loader import load_env
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.visualize_tool import (
    visualize_freebody_tool,
    visualize_mfz_contour_tool,
    visualize_resu_tool,
    visualize_stress_tool,
)
from tools.validators import file_exists

# ----------------------------
# Environment setup
# ----------------------------
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)
gmsh_bin, as_run_bin = load_env()

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started.")


# ----------------------------
# Helper
# ----------------------------
def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """Safely execute a tool, handle errors, and verify expected file."""
    print(f"\n🚀 Running: {tool_func.name}")
    try:
        tool_input = kwargs.pop("tool_input", {})
        result = tool_func.invoke(tool_input)
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


# ----------------------------
# Manual pipeline
# ----------------------------
def run_pipeline():
    print("\n=== 🧠 Starting Code_Aster pipeline (manual mode) ===\n")

    if not safe_call(generate_geo_tool, expected_file=os.path.join(WORKSPACE, "model.geo")):
        return
    if not safe_call(mesh_with_gmsh_tool, expected_file=os.path.join(WORKSPACE, "model.med")):
        return
    if not safe_call(generate_comm_tool, expected_file=os.path.join(WORKSPACE, "model.comm")):
        return
    if not safe_call(generate_export_tool, expected_file=os.path.join(WORKSPACE, "model.export")):
        return
    if not safe_call(run_code_aster_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.mess")):
        return

    print("\n=== 📊 Visualizing ===")
    vis_tools = [
        ("Deflection/Rotation", visualize_resu_tool, {"resu_name": "beam_analysis.resu"}),
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


# ----------------------------
# Agent (LLM) pipeline
# ----------------------------
def run_agent_mode(llm):
    """LLM-CoPilot mode: lightweight, text-driven automation."""
    print("\n=== 🤖 Running LLM-CoPilot Mode (no LangGraph) ===\n")

    # Step 1 — We ask the model to give a work plan
    user_prompt = (
        "Generate a short step-by-step plan to simulate a 3-meter steel beam with 15 elements "
        "using Code_Aster. Mention which files should be created (.geo, .med, .comm, .export), "
        "and then suggest when to run Code_Aster and visualize results."
    )

    try:
        response = llm.invoke(user_prompt)
        plan_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"❌ Error invoking LLM: {e}")
        return

    print("\n=== 🧠 LLM PLAN ===")
    print(plan_text)

# Step 2 — Save the program to the workspace file
    output_path = os.path.join(WORKSPACE, "llm_plan.txt")
    with open(output_path, "w") as f:
        f.write(plan_text)
    print(f"\n📝 Plan saved to: {output_path}")

# Step 3 — Auto-run based on keywords
    steps = plan_text.lower()

    def run_step_if(keyword, tool, expected_file):
        if keyword in steps:
            print(f"\n⚙️  Detected '{keyword}' → Running {tool.name}")
            safe_call(tool, expected_file=expected_file)
        else:
            print(f"⏩ Skipping '{keyword}' (not found in LLM plan).")

# Matching keywords to tools
    run_step_if(".geo", generate_geo_tool, os.path.join(WORKSPACE, "model.geo"))
    run_step_if(".med", mesh_with_gmsh_tool, os.path.join(WORKSPACE, "model.med"))
    run_step_if(".comm", generate_comm_tool, os.path.join(WORKSPACE, "model.comm"))
    run_step_if(".export", generate_export_tool, os.path.join(WORKSPACE, "model.export"))
    run_step_if("run", run_code_aster_tool, os.path.join(WORKSPACE, "beam_analysis.mess"))
    run_step_if("visual", visualize_resu_tool, os.path.join(WORKSPACE, "beam_analysis.resu"))

    print("\n🎯 LLM-CoPilot pipeline finished successfully!\n")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "manual", "auto"], default="auto")
    args = parser.parse_args()

    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="tinyllama:latest", temperature=0)
        online = True
        print("🧠 LLM model loaded (Ollama: tinyllama)")
    except Exception as e:
        from langchain_community.chat_models import FakeListChatModel

        llm = FakeListChatModel(
            responses=[
                "OK. Let's start the analysis.",
                "Geometry created.",
                "Mesh done.",
                "Simulation completed.",
                "Visualization ready.",
            ]
        )
        online = False
        print(f"⚠️ Ollama not available ({e}). Using FakeListChatModel fallback.")

    if args.mode == "agent" or (args.mode == "auto" and online):
        run_agent_mode(llm)
    else:
        run_pipeline()
