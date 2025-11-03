# ----------------------------
#  tools/mesher_gmsh.py
# ----------------------------

import os
import shutil
import subprocess

from langchain_core.tools import tool  # ✅ updated import for LangChain 1.x

from env_loader import load_env  # Reads GMSH_BIN path from .env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("mesh_with_gmsh", return_direct=False)
def mesh_with_gmsh_tool(
    geo_name: str = "model.geo",
    med_name: str = "model.med",
) -> str:
    """
    Create a 1D mesh (.med file) from the given geometry using Gmsh.
    Ensures MED version 4.0.0 compatibility for Code_Aster.
    """

    # --- Load environment configuration ---
    gmsh_bin, _ = load_env()  # returns (gmsh_path, as_run_path)
    gmsh_path = shutil.which(gmsh_bin)

    if gmsh_path is None:
        return f"❌ Error: '{gmsh_bin}' not found. Please install Gmsh or set GMSH_BIN in .env."

    # --- Define input/output file paths ---
    geo_path = os.path.join(WORKSPACE, geo_name)
    med_path = os.path.join(WORKSPACE, med_name)

    if not os.path.exists(geo_path):
        return f"❌ Error: input file '{geo_path}' not found."

    # --- Gmsh options to enforce MED v4.0.0 compatibility ---
    gmsh_opts = [
        "-setnumber",
        "Mesh.MedFileMinorVersion",
        "0",  # MED 4.0.0 compatibility
        "-setnumber",
        "Mesh.SaveWithoutOrphans",
        "1",  # clean mesh
        "-v",
        "2",  # verbosity level
    ]

    # --- Build full command for subprocess ---
    cmd = [gmsh_path, geo_path, "-1", "-format", "med", "-o", med_path] + gmsh_opts

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"⚠️ Gmsh failed with error:\n{result.stderr}\n{result.stdout}"

        if not os.path.exists(med_path):
            return f"⚠️ Gmsh finished but output file '{med_path}' not found."

        return (
            f"✅ Meshing completed successfully.\n"
            f"MED file saved at: {os.path.abspath(med_path)}\n\n"
            f"🔧 MED version forced to 4.0.0 for Code_Aster compatibility.\n\n"
            f"Gmsh output:\n{result.stdout}"
        )

    except Exception as e:
        return f"❌ Exception while running Gmsh: {e}"
