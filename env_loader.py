# env_loader.py
import os

from dotenv import load_dotenv


def load_env():
    load_dotenv()
    gmsh_bin = os.getenv("GMSH_BIN", "gmsh")
    as_run_bin = os.getenv("AS_RUN_BIN", "as_run")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return gmsh_bin, as_run_bin, google_api_key
