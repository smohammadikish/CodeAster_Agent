# ----------------------------
#  tools/mesher_gmsh.py (fixed order)
# ----------------------------

import os
import shutil
import subprocess

from langchain_core.tools import tool  # ✅ Compatible with LangChain 1.x

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

    # --- Load environment configuration FIRST ---
    gmsh_bin, *_ = load_env()
    gmsh_path = shutil.which(gmsh_bin)

    if gmsh_path is None:
        return (
            f"❌ Error: '{gmsh_bin}' not found.\n"
            "Please install Gmsh or set GMSH_BIN correctly in your .env file.\n"
            "Example:\nGMSH_BIN=C:\\Program Files\\Gmsh\\gmsh.exe"
        )

    # --- Define input/output file paths ---
    geo_path = os.path.join(WORKSPACE, geo_name)
    med_path = os.path.join(WORKSPACE, med_name)

    if not os.path.exists(geo_path):
        return f"❌ Error: input file '{geo_path}' not found."

    # --- Gmsh options for MED v4.0.0 compatibility ---
    gmsh_opts = [
        "-setnumber",
        "Mesh.MedFileMinorVersion",
        "0",  # MED 4.0.0
        "-setnumber",
        "Mesh.SaveWithoutOrphans",
        "1",  # clean orphan nodes
        "-v",
        "2",  # verbosity level
    ]

    # --- Full Gmsh command ---
    cmd = [gmsh_path, geo_path, "-1", "-format", "med", "-o", med_path] + gmsh_opts

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return (
                "⚠️ Gmsh failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout}"
            )

        if not os.path.exists(med_path):
            return f"⚠️ Gmsh finished but output file '{med_path}' not found."

        return (
            f"✅ Meshing completed successfully.\n"
            f"MED file saved at: {os.path.abspath(med_path)}\n"
            f"🔧 MED version forced to 4.0.0 for Code_Aster compatibility.\n\n"
            f"--- Gmsh Output (first 500 chars) ---\n{result.stdout[:500]}..."
        )

    except Exception as e:
        return f"❌ Exception while running Gmsh: {e}"
