# server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from main_agent import run_agent_mode  # your existing function

app = FastAPI()

# Allow frontend (Next.js) to connect to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "✅ FastAPI backend is running."}


@app.post("/copilot")
async def copilot_endpoint(request: Request):
    data = await request.json()
    user_prompt = data.get("input", "")
    print(f"🧠 Received prompt from frontend: {user_prompt}")

    # Run your main LangChain Agent (or MockLLM)
    result = run_agent_mode()
    return {"output": f"Simulation complete for: {user_prompt}\n\n{result}"}
