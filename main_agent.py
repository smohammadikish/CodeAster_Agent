# main_agent.py
import argparse
import logging
import os

from env_loader import load_env
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.summarize_results_tool import summarize_results_tool
from tools.validators import file_exists
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# ✅ Environment
gmsh_bin, as_run_bin = load_env()


# ------------------ Logging setup ------------------
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started.")


# ------------------ Helper ------------------
def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """Safely execute a tool, handle errors, and verify expected file."""
    print(f"\n🚀 Running: {tool_func.name}")
    try:
        tool_input = kwargs.pop("tool_input", {})
        result = tool_func.run(tool_input=tool_input, **kwargs)
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


# ------------------ Manual mode ------------------
def run_pipeline():
    """Run Code_Aster workflow step-by-step (manual mode)."""
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

    # --- Summarize ---
    # print("\n=== 📄 Summarizing results ===")
    # print(summarize_results_tool.run())

    # --- Visualizations ---
    print("\n=== 📊 Visualizing ===")

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
            print(func.run(tool_input=args))
        except Exception as e:
            print(f"⚠️ Visualization '{title}' failed: {e}")

    print("\n🎯 Pipeline finished successfully!")


# ------------------ Agent mode ------------------
def run_agent_mode(llm):
    """Run Agent-based Code_Aster workflow (LangChain 1.x)."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.prompts import ChatPromptTemplate

    print("\n=== 🤖 Running LLM-driven Agent Mode ===\n")

    tools = [
        generate_geo_tool,
        mesh_with_gmsh_tool,
        generate_comm_tool,
        generate_export_tool,
        run_code_aster_tool,
        summarize_results_tool,
        visualize_resu_tool,
        visualize_stress_tool,
        visualize_mfz_contour_tool,
        visualize_freebody_tool,
    ]

    prompt = ChatPromptTemplate.from_template(
        "You are a mechanical engineering assistant that can use the provided tools "
        "to simulate, analyze, and visualize beam models using Code_Aster.\n\n"
        "User input: {input}\n"
        "{agent_scratchpad}"
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_prompt = (
        "Simulate a 3-meter steel beam with 15 elements and visualize all results "
        "(deflection, moment, shear, and free-body)."
    )
    result = executor.invoke({"input": user_prompt})

    print("\n=== 🧠 AGENT RESULT ===")
    print(result["output"])


# ------------------ Entry point ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "manual", "auto"], default="auto")
    args = parser.parse_args()

    # ✅ Try to load MockLLM (offline mode)
    try:
        from llm.mock_llm import MockLLM

        llm = MockLLM()
        print("🧠 Using local MockLLM (offline testing mode).")
        online = False
    except Exception as e:
        print(f"⚠️ Could not load MockLLM: {e}. Trying Ollama or FakeListChatModel.")
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model="mistral", temperature=0)
            online = True
            print("🧠 LLM model loaded (Ollama: mistral)")
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
            print(
                f"⚠️ No local model found ({e}). Using FakeListChatModel (offline fallback)."
            )

    # ✅ Choose execution mode
    mode = args.mode
    if mode == "agent" or (mode == "auto" and online):
        run_agent_mode(llm)
    else:
        run_pipeline()
