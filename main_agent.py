# main_agent.py — LLM-CoPilot v2 (Guided + Semantic)
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
    safe_call(visualize_resu_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.resu"))
    print("\n🎯 Pipeline finished successfully!")


# ----------------------------
# LLM-CoPilot v2 (Guided + Semantic)
# ----------------------------
def run_agent_mode(llm):
    """LLM-CoPilot v2: guided plan generation + semantic stage detection."""
    print("\n=== 🤖 Running LLM-CoPilot Mode (Guided + Semantic) ===\n")

    # 1️⃣ System-level guidance
    system_prompt = """
You are a structural engineering assistant that plans and controls a Code_Aster simulation pipeline.

The correct workflow always follows these ordered stages:
1. Generate geometry (.geo) using Gmsh.
2. Mesh the geometry to produce (.med).
3. Create a solver input file (.comm) for Code_Aster.
4. Generate an export file (.export) that defines run parameters.
5. Run Code_Aster simulation.
6. Visualize the results (.resu, .txt).

Your goal is to output a short structured plan mentioning these steps clearly.
Avoid GUI or button instructions — only mention file-based workflow.
"""

    user_prompt = (
        "Simulate a 3-meter steel beam with 15 elements using Code_Aster. "
        "Please provide a short plan based on the pipeline stages described above."
    )

    # 2️⃣ Combine both prompts
    full_prompt = system_prompt + "\nUser request:\n" + user_prompt

    try:
        response = llm.invoke(full_prompt)
        plan_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"❌ Error invoking LLM: {e}")
        return

    print("\n=== 🧠 LLM PLAN (Guided) ===")
    print(plan_text)

    # 3️⃣ Save plan to file
    output_path = os.path.join(WORKSPACE, "llm_plan.txt")
    with open(output_path, "w") as f:
        f.write(plan_text)
    print(f"\n📝 Plan saved to: {output_path}")

    # 4️⃣ Semantic keyword mapping
    steps = plan_text.lower()

    semantic_map = {
        "geo": ["geo", "geometry", "shape", "create beam"],
        "med": ["med", "mesh", "meshing", "gmsh"],
        "comm": ["comm", "command", "solver", "input file"],
        "export": ["export", "as_run", "run file"],
        "run": ["run", "execute", "analyze", "simulation"],
        "visual": ["visual", "plot", "diagram", "deflection", "result", "output"],
    }

    def detect(keyword_group):
        for kw in keyword_group:
            if kw in steps:
                return True
        return False

    # 5️⃣ Execute tools intelligently
    if detect(semantic_map["geo"]):
        safe_call(generate_geo_tool, expected_file=os.path.join(WORKSPACE, "model.geo"))
    else:
        print("⏩ Skipping geometry generation.")

    if detect(semantic_map["med"]):
        safe_call(mesh_with_gmsh_tool, expected_file=os.path.join(WORKSPACE, "model.med"))
    else:
        print("⏩ Skipping mesh generation.")

    if detect(semantic_map["comm"]):
        safe_call(generate_comm_tool, expected_file=os.path.join(WORKSPACE, "model.comm"))
    else:
        print("⏩ Skipping .comm creation.")

    if detect(semantic_map["export"]):
        safe_call(generate_export_tool, expected_file=os.path.join(WORKSPACE, "model.export"))
    else:
        print("⏩ Skipping .export creation.")

    if detect(semantic_map["run"]):
        safe_call(run_code_aster_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.mess"))
    else:
        print("⏩ Skipping Code_Aster run.")

    if detect(semantic_map["visual"]):
        safe_call(visualize_resu_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.resu"))
    else:
        print("⏩ Skipping visualization.")

    print("\n🎯 Guided LLM-CoPilot pipeline completed!\n")


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
                "Step 1: Create geometry (.geo)\nStep 2: Mesh (.med)\nStep 3: Run Code_Aster\nStep 4: Visualize results",
            ]
        )
        online = False
        print(f"⚠️ Ollama not available ({e}). Using FakeListChatModel fallback.")

    if args.mode == "agent" or (args.mode == "auto" and online):
        run_agent_mode(llm)
    else:
        run_pipeline()