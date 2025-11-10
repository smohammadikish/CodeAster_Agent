# env_loader.py
import os

from dotenv import load_dotenv


def load_env():
    load_dotenv()
    gmsh_bin = os.getenv("GMSH_BIN", "gmsh")
    as_run_bin = os.getenv("AS_RUN_BIN", "as_run")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return gmsh_bin, as_run_bin, ollama_url, openai_key, openai_model
