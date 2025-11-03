# ----------------------------
#  agents/codeaster_agent.py
# ----------------------------

"""
Code_Aster intelligent agent using OpenAI GPT (LangChain 1.x + LangGraph).
"""

from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY, etc.)
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
# === Import project tools ===
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)


# ============================================================
# 🧠 Build Code_Aster Agent (OpenAI version)
# ============================================================
def build_codeaster_agent_openai(
    model_name: str = "gpt-4o-mini", temperature: float = 0.2
):
    """
    Create a LangGraph-based ReAct agent using OpenAI GPT models.
    Compatible with LangChain 1.x and all Code_Aster tools.
    """

    # --- Initialize OpenAI LLM ---
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # --- Register project tools ---
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

    # --- Build ReAct agent with LangGraph ---
    agent = create_react_agent(llm, tools)
    return agent


# ============================================================
# 🧪 Manual test (CLI)
# ============================================================
if __name__ == "__main__":
    print("=== 🤖 Initializing Code_Aster Agent (OpenAI) ===")
    agent = build_codeaster_agent_openai()

    print("✅ Agent ready. Example commands:")
    print("   - 'Generate a 3m beam, mesh it, run Code_Aster, and show the deflection.'\n")

    user_prompt = (
        "Generate a 2m cantilever beam, run Code_Aster, "
        "and plot shear and moment diagrams."
    )
    print(f"👤 User: {user_prompt}\n")

    response = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    print("🤖 Agent response:\n", response)
