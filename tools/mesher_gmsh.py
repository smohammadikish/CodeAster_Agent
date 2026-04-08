# ----------------------------
#  tools/mesher_gmsh.py
# ----------------------------

import os
import shutil
import subprocess

from langchain_core.tools import tool

from env_loader import load_env
from storage import resolve_workspace_file


@tool("mesh_with_gmsh", return_direct=False)
def mesh_with_gmsh_tool(
    geo_name: str = "model.geo",
    med_name: str = "model.med",
    run_id: str = "",
) -> str:
    """
    Create a 1D mesh (.med file) from the given geometry using Gmsh.
    Ensures MED version 4.0.0 compatibility for Code_Aster.
    """

    gmsh_bin, *_ = load_env()
    gmsh_path = shutil.which(gmsh_bin)

    if gmsh_path is None:
        return (
            f"âŒ Error: '{gmsh_bin}' not found.\n"
            "Please install Gmsh or set GMSH_BIN correctly in your .env file.\n"
            "Example:\nGMSH_BIN=C:\\Program Files\\Gmsh\\gmsh.exe"
        )

    geo_path = resolve_workspace_file(geo_name, "inputs", run_id or None)
    med_path = resolve_workspace_file(med_name, "inputs", run_id or None)

    if not os.path.exists(geo_path):
        return f"âŒ Error: input file '{geo_path}' not found."

    gmsh_opts = [
        "-setnumber",
        "Mesh.MedFileMinorVersion",
        "0",
        "-setnumber",
        "Mesh.SaveWithoutOrphans",
        "1",
        "-v",
        "2",
    ]

    cmd = [gmsh_path, str(geo_path), "-1", "-format", "med", "-o", str(med_path)] + gmsh_opts

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return (
                "âš ï¸ Gmsh failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout}"
            )

        if not os.path.exists(med_path):
            return f"âš ï¸ Gmsh finished but output file '{med_path}' not found."

        return (
            f"âœ… Meshing completed successfully.\n"
            f"MED file saved at: {med_path.resolve()}\n"
            f"ðŸ”§ MED version forced to 4.0.0 for Code_Aster compatibility.\n\n"
            f"--- Gmsh Output (first 500 chars) ---\n{result.stdout[:500]}..."
        )

    except Exception as e:
        return f"âŒ Exception while running Gmsh: {e}"
