# tools/mesher_gmsh.py
import os
import shutil
import subprocess

from langchain.tools import tool

from env_loader import load_env  # To read the GMSH_BIN path from .env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("mesh_with_gmsh", return_direct=False)
def mesh_with_gmsh_tool(
    geo_name: str = "model.geo",
    med_name: str = "model.med",
) -> str:
    """
    Create a mesh (.med file) from the geometry file using Gmsh.
    Ensures MED version compatibility (MED 4.0.0) for Code_Aster 4.0.0.
    """

    gmsh_bin, _ = load_env()  # Only returns the gmsh path, as_run is not used for now
    gmsh_path = shutil.which(gmsh_bin)

    if gmsh_path is None:
        return f"❌ Error: '{gmsh_bin}' not found. Please install Gmsh or set GMSH_BIN in .env."

    geo_path = os.path.join(WORKSPACE, geo_name)
    med_path = os.path.join(WORKSPACE, med_name)

    if not os.path.exists(geo_path):
        return f"❌ Error: input file '{geo_path}' not found."

    # --- Force MED 4.0.0 compatibility for Code_Aster 4.0.0 ---
    gmsh_opts = [
        "-setnumber",
        "Mesh.MedFileMinorVersion",
        "0",  # <= Key fix line
        "-setnumber",
        "Mesh.SaveWithoutOrphans",
        "1",  # keep mesh clean
        "-v",
        "2",  # verbosity level
    ]

    # --- Build Gmsh command (1D mesh + MED output) ---
    cmd = [gmsh_path, geo_path, "-1", "-format", "med", "-o", med_path] + gmsh_opts

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"⚠️ Gmsh failed with error:\n{result.stderr}\n{result.stdout}"

        if not os.path.exists(med_path):
            return f"⚠️ Gmsh finished but output file '{med_path}' not found."

        return (
            f"✅ Meshing completed successfully.\n"
            f"MED file saved at: {med_path}\n\n"
            f"🔧 MED version forced to 4.0.0 for Code_Aster 4.0.0 compatibility.\n\n"
            f"Gmsh output:\n{result.stdout}"
        )

    except Exception as e:
        return f"❌ Exception while running Gmsh: {e}"
