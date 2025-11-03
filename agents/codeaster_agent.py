"""
Simplified Local Code_Aster Agent
✅ Compatible with LangChain 1.x + LangChain-Ollama 1.x (2025)
💻 Runs fully offline on macOS M1 with TinyLlama.
"""

from langchain.schema import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
# === Import Tools ===
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)


# ============================================================
# 🧠 Initialize Local Model (Ollama)
# ============================================================
def build_codeaster_agent_ollama(model_name="tinyllama", temperature=0.3):
    """Return a simple callable agent backed by a local Ollama model."""
    llm = ChatOllama(model=model_name, temperature=temperature)

    def agent(prompt: str):
        """Simple function-based local agent using Ollama."""
        print("👤 User:", prompt)
        response = llm.invoke([HumanMessage(content=prompt)])
        print("🤖 AI:", response.content)
        return response.content

    return agent


# ============================================================
# 🧪 Manual Te
