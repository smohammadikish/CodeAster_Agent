# main_agent.py — Real LangChain Agent (ReAct via LangGraph)
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
gmsh_bin, as_run_bin, ollama_url = load_env()


# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    filename=os.path.join(WORKSPACE, "agent_pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("Pipeline started (LangChain Agent mode).")




# ----------------------------
# Safe execution wrapper
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
# Manual pipeline (for fallback)
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
    if not safe_call(visualize_resu_tool, expected_file=os.path.join(WORKSPACE, "beam_analysis.resu")):
        return


    print("\n🎯 Manual pipeline finished successfully!\n")




# ----------------------------
# LangChain Agent Mode (ReAct via LangGraph)
# ----------------------------
def run_agent_mode(llm):
    """Run a real LangChain ReAct agent that uses the Code_Aster tools autonomously."""
    from langgraph.prebuilt import create_react_agent


    print("\n=== 🤖 Running REAL LangChain Agent (ReAct via LangGraph) ===\n")


    # 1️⃣ Register tools (already built with LangChain tool interface)
    tools = [
        generate_geo_tool,
        mesh_with_gmsh_tool,
        generate_comm_tool,
        generate_export_tool,
        run_code_aster_tool,
        visualize_resu_tool,
        visualize_stress_tool,
        visualize_mfz_contour_tool,
        visualize_freebody_tool,
    ]


    # 2️⃣ Create the ReAct agent graph
    agent = create_react_agent(llm, tools)


    # 3️⃣ Define high-level workflow guidance
    workflow_rules = """
You are a structural engineering assistant controlling a Code_Aster simulation pipeline.
Follow this exact workflow:
1. Generate geometry (.geo)
2. Mesh the geometry (.med)
3. Create solver input (.comm)
4. Create export file (.export)
5. Run Code_Aster simulation
6. Visualize results (.resu, .txt)


Use ONLY the provided tools, in the given order.
If one tool fails, stop and report the cause clearly.
Do not suggest GUI steps or manual edits — only file-based commands.
If visualization tools are available, produce deflection and moment plots.
"""


    # 4️⃣ User task
    user_task = (
        "Simulate a 3-meter steel beam with 15 elements using Code_Aster, "
        "and visualize deflection, moment, and free-body diagrams."
    )


    # 5️⃣ Compose agent input
    full_input = f"{workflow_rules}\n\nUser request:\n{user_task}"


    # 6️⃣ Run the agent (the agent will choose tools autonomously)
    print("🤔 Agent reasoning and tool selection in progress...\n")
    result = agent.invoke({"messages": [("user", full_input)]})


    # 7️⃣ Display final message
    try:
        print("\n=== 🧠 AGENT FINAL MESSAGE ===")
        print(result["messages"][-1].content)
    except Exception:
        print(result)


    # 8️⃣ Save reasoning trace
    trace_path = os.path.join(WORKSPACE, "agent_trace.log")
    with open(trace_path, "w") as f:
        f.write(str(result))
    print(f"\n📄 Agent reasoning trace saved to: {trace_path}")




# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "manual", "auto"], default="auto")
    args = parser.parse_args()


    try:
        from langchain_ollama import ChatOllama


        # Use a chat-capable model for agent (e.g., llama3)
        llm = ChatOllama(model="mistral:latest", temperature=0, base_url=ollama_url)
        print(f"🔗 Connected to Ollama at {ollama_url}")
        online = True
        print("🧠 LLM model loaded (Ollama: mistral:latest)")
    except Exception as e:
        from langchain_community.chat_models import FakeListChatModel


        llm = FakeListChatModel(
            responses=[
                "Starting simulation.",
                "Geometry generated.",
                "Mesh created.",
                "Simulation done.",
                "Visualization ready.",
            ]
        )
        online = False
        print(f"⚠️ Ollama not available ({e}). Using FakeListChatModel fallback.")


    # Run in chosen mode
    if args.mode == "agent" or (args.mode == "auto" and online):
        run_agent_mode(llm)
    else:
        run_pipeline()