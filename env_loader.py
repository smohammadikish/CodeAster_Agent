# env_loader.py
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env():
    _load_repo_env()
    gmsh_bin = os.getenv("GMSH_BIN", "gmsh")
    as_run_bin = os.getenv("AS_RUN_BIN", "as_run")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return gmsh_bin, as_run_bin, ollama_url, openai_key, openai_model
