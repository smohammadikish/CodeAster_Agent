# llm/openai_llm.py
from langchain.chat_models import ChatOpenAI

from app.env_loader import load_env


def get_openai_llm():
    """Return a real OpenAI LLM (for production use)."""
    gmsh_bin, as_run_bin, openai_key = load_env()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key,
    )
