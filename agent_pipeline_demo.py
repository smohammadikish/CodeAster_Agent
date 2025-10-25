# agent_pipeline.py
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
# ---- Import all 6 tools ----
from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.visualize_tool import visualize_resu_tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# ---- Step 1. Choose your LLM ----
try:
    # ✅ Option A: Use Ollama (local mistral / llama3)
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="mistral", temperature=0)
    print("🧠 Using local Ollama model: mistral")

except Exception as e:
    # ⚙️ Option B: fallback if Ollama not installed
    print(f"⚠️ Ollama not found ({e}), switching to FakeListChatModel (offline mode).")
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

# ---- Step 2. Define all tools ----
tools = [
    generate_geo_tool,
    mesh_with_gmsh_tool,
    generate_comm_tool,
    generate_export_tool,
    run_code_aster_tool,
    visualize_resu_tool,
]

# ---- Step 3. Define the system prompt ----
system_prompt = """
You are an intelligent structural analysis assistant using Code_Aster 14 and Gmsh.
Your goal is to perform beam analyses for the user automatically.

Workflow you MUST follow (in correct order):
1 Ask the user for missing parameters: beam length (m), number of elements, section HY/HZ, material (E, NU, RHO), and load FY.
2 Then call these tools IN ORDER:
   - generate_geo
   - mesh_with_gmsh
   - generate_comm
   - generate_export
   - run_code_aster
   - visualize_resu
3 Summarize results (e.g., "Analysis completed, max displacement shown in beam_job.png").
4 Be concise and clear in explanations. Never show code; just explain steps logically.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

# ---- Step 4. Create the Agent ----
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---- Step 5. Interactive run ----
if __name__ == "__main__":
    print("\n🤖 Structural Analysis Agent is ready!")
    print("Type your request (or 'exit' to quit):")

    while True:
        user_input = input("\n🧱 You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting agent. Goodbye!")
            break

        try:
            result = executor.invoke({"input": user_input})
            print("\nAgent:", result["output"])
        except Exception as e:
            print(f"❌ Error during execution: {e}")
