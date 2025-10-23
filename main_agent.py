# main_agent.py
import os
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesher_gmsh_tool
from tools.comm_generator import comm_generator_tool
from tools.export_generator import export_generator_tool
from tools.run_aster_tool import run_aster_tool
from tools.visualize_tool import visualize_resu_tool
from tools.summarize_results_tool import summarize_results_tool
from tools.validators import file_exists
from env_loader import load_env

# ==============================
# 🧠 LLM Configuration
# ==============================
# Choose which LLM backend to use

from llm.mock_llm import MockLLM
# from llm.openai_llm import get_openai_llm
# from llm.local_llm import get_local_llm

# Select model
llm = MockLLM()                 # Offline mode (default for development)
# llm = get_openai_llm()        # When company wants to connect to OpenAI
# llm = get_local_llm("llama3") # When testing with local model like Ollama

# ==============================
# ⚙️ Global Configuration
# ==============================
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)
gmsh_bin, as_run_bin = load_env()


def safe_call(tool_func, expected_file: str | None = None, **kwargs) -> bool:
    """
    Safely execute a tool step, handle errors, and verify output file.
    Stops the pipeline if something goes wrong.
    """
    print(f"\n🚀 Running: {tool_func.name}")
    result = tool_func.run(**kwargs)
    print(result)

    # Detect errors in tool result
    if "Error" in str(result) or "❌" in str(result):
        print(f"❌ {tool_func.name} failed — stopping pipeline.")
        return False

    # Check expected output file
    if expected_file and not file_exists(expected_file):
        print(f"⚠️ Expected file not found: {expected_file}")
        return False

    print(f"✅ Step '{tool_func.name}' completed successfully.\n")
    return True


def run_pipeline():
    """Main Code_Aster pipeline controller."""
    print("\n=== 🧠 Starting Code_Aster pipeline ===\n")

    # 1️⃣ GEO generation
    geo_path = os.path.join(WORKSPACE, "model.geo")
    if not safe_call(generate_geo_tool, expected_file=geo_path):
        return

    # 2️⃣ Meshing
    med_path = os.path.join(WORKSPACE, "model.med")
    if not safe_call(mesher_gmsh_tool, expected_file=med_path):
        return

    # 3️⃣ COMM file
    comm_path = os.path.join(WORKSPACE, "model.comm")
    if not safe_call(comm_generator_tool, expected_file=comm_path):
        return

    # 4️⃣ EXPORT file
    export_path = os.path.join(WORKSPACE, "job.export")
    if not safe_call(export_generator_tool, expected_file=export_path):
        return

    # 5️⃣ Run Code_Aster solver
    mess_path = os.path.join(WORKSPACE, "result.mess")
    if not safe_call(run_aster_tool, expected_file=mess_path):
        return

    # 6️⃣ Summarize results (.mess)
    print("\n=== 📄 Summarizing analysis results ===")
    summary = summarize_results_tool.run()
    print(summary)

    # 7️⃣ Visualization (.resu)
    print("\n=== 📊 Generating visualization ===")
    visualization = visualize_resu_tool.run()
    print(visualization)

    print("\n🎯 Pipeline finished successfully! All results stored in /workspace")


if __name__ == "__main__":
    run_pipeline()
