# --- main_agent.py ---
import argparse
import logging
import os

from langchain_community.chat_models import FakeListChatModel

from env_loader import load_env
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.summarize_results_tool import summarize_results_tool
from tools.validators import file_exists
from tools.visualize_tool import visualize_resu_tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

gmsh_bin, as_run_bin, _ = load_env()

# Setup logging
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started.")


# ------------------ Helper function ------------------
def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """Safely execute a tool, handle errors, and verify expected file."""
    print(f"\n🚀 Running: {tool_func.name}")
    result = tool_func.run(**kwargs)
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


# ------------------ Manual mode ------------------
def run_pipeline():
    """Run Code_Aster workflow step-by-step (manual mode)."""
    print("\n=== 🧠 Starting Code_Aster pipeline (manual mode) ===\n")

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
        generate_export_tool, expected_file=os.path.join(WORKSPACE, "job.export")
    ):
        return
    if not safe_call(
        run_code_aster_tool, expected_file=os.path.join(WORKSPACE, "result.mess")
    ):
        return

    print("\n=== 📄 Summarizing results ===")
    print(summarize_results_tool.run())

    print("\n=== 📊 Visualizing ===")
    print(visualize_resu_tool.run())

    print("\n🎯 Pipeline finished successfully!")


# ------------------ Agent mode ------------------
def run_agent_mode(llm):
    """Run Agent-based Code_Aster workflow."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    print("\n=== 🤖 Running LLM-driven Agent Mode ===\n")

    tools = [
        generate_geo_tool,
        mesh_with_gmsh_tool,
        generate_comm_tool,
        generate_export_tool,
        run_code_aster_tool,
        summarize_results_tool,
        visualize_resu_tool,
    ]

    agent = create_tool_calling_agent(llm, tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_prompt = (
        "Simulate a 3-meter steel beam with 15 elements and visualize the results."
    )
    result = executor.invoke({"input": user_prompt})
    print("\n=== 🧠 AGENT RESULT ===")
    print(result["output"])


# ------------------ Entry point ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "manual", "auto"], default="auto")
    args = parser.parse_args()

    # Try loading LLM (Ollama or fallback)
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="mistral", temperature=0)
        online = True
        print("🧠 LLM model loaded (Ollama: mistral)")

    except Exception as e:
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
        print(f"⚠️ No local model found ({e}). Using FakeListChatModel (offline mode).")

    # Decide execution mode
    mode = args.mode
    if mode == "agent" or (mode == "auto" and online):
        run_agent_mode(llm)
    else:
        run_pipeline()
