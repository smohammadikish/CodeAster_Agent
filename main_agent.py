# main_agent.py
import logging
import os
import sys

from env_loader import load_env
from llm.mock_llm import MockLLM
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.summarize_results_tool import summarize_results_tool
from tools.validators import file_exists
from tools.visualize_tool import visualize_resu_tool

# from llm.openai_llm import get_openai_llm
# from llm.local_llm import get_local_llm




# Choose model
llm = MockLLM()  # Offline mode (default for development)
# llm = get_openai_llm()        # Online (OpenAI API)
# llm = get_local_llm("llama3") # Local LLM (Ollama / LM Studio)


# Workspace setup & logging
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)
gmsh_bin, as_run_bin, _ = load_env()

# Setup logging (writes to workspace/pipeline.log)
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started.")


# Safe tool execution helper
def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """
    Safely execute a tool, handle errors, and verify expected file.
    """
    print(f"\n🚀 Running: {tool_func.name}")
    result = tool_func.run(**kwargs)
    print(result)
    logging.info(f"Executed tool: {tool_func.name}")
    logging.info(result)

    if "Error" in str(result) or "❌" in str(result):
        print(f"❌ {tool_func.name} failed — stopping pipeline.")
        logging.error(f"{tool_func.name} failed.")
        return False

    if expected_file and not file_exists(expected_file):
        print(f"⚠️ Expected file not found: {expected_file}")
        logging.warning(f"Missing file: {expected_file}")
        return False

    print(f"✅ Step '{tool_func.name}' completed successfully.\n")
    logging.info(f"✅ Step '{tool_func.name}' completed successfully.")
    return True


# Sequential (manual) pipeline
def run_pipeline():
    """Run Code_Aster workflow step-by-step (manual mode)."""
    print("\n=== 🧠 Starting Code_Aster pipeline (manual mode) ===\n")
    logging.info("Sequential pipeline started.")

    # 1. GEO
    geo_path = os.path.join(WORKSPACE, "model.geo")
    if not safe_call(generate_geo_tool, expected_file=geo_path):
        return

    # 2. MESH
    med_path = os.path.join(WORKSPACE, "model.med")
    if not safe_call(mesh_with_gmsh_tool, expected_file=med_path):
        return

    # 3. COMM
    comm_path = os.path.join(WORKSPACE, "model.comm")
    if not safe_call(generate_comm_tool, expected_file=comm_path):
        return

    # 4. EXPORT
    export_path = os.path.join(WORKSPACE, "job.export")
    if not safe_call(generate_export_tool, expected_file=export_path):
        return

    # 5. RUN ASTER
    mess_path = os.path.join(WORKSPACE, "result.mess")
    if not safe_call(run_code_aster_tool, expected_file=mess_path):
        return

    # 6. SUMMARY
    print("\n=== 📄 Summarizing analysis results ===")
    summary = summarize_results_tool.run()
    print(summary)
    logging.info("Summary generated successfully.")

    # 7. VISUALIZATION
    print("\n=== 📊 Generating visualization ===")
    visualization = visualize_resu_tool.run()
    print(visualization)
    logging.info("Visualization completed successfully.")

    print("\n🎯 Pipeline finished successfully! Results saved in /workspace")
    logging.info("Pipeline finished successfully.")


# 🤖 LangChain Agent
from langchain_core.agents import AgentExecutor, create_react_agent


def run_agent_mode():
    print("\n=== 🤖 Running LLM-driven Agent Mode ===\n")
    logging.info("Starting LLM-driven agent mode (v1.0.2).")

    tools = [
        generate_geo_tool,
        mesh_with_gmsh_tool,
        generate_comm_tool,
        generate_export_tool,
        run_code_aster_tool,
        summarize_results_tool,
        visualize_resu_tool,
    ]

    agent = create_react_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_prompt = "Simulate a 3-meter beam with 15 elements and visualize the results."
    result = agent_executor.invoke({"input": user_prompt})

    print("\n=== 🧠 AGENT RESULT ===")
    print(result["output"])
    logging.info(f"Agent result: {result['output']}")
