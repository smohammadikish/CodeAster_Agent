# ----------------------------
#  tools/mesher_gmsh.py
# ----------------------------

import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from env_loader import load_env
from storage import resolve_workspace_file


def _resolve_executable(bin_value: str) -> str | None:
    """
    Resolve an executable from either:
    - an absolute/relative file path
    - a command name on PATH
    """
    candidate = (bin_value or "").strip().strip('"').strip("'")
    if not candidate:
        return None

    direct = Path(candidate)
    if direct.is_file():
        return str(direct)

    return shutil.which(candidate)


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
    gmsh_path = _resolve_executable(gmsh_bin)

    if gmsh_path is None:
        return (
            f"ERROR: Gmsh executable not found: '{gmsh_bin}'.\n"
            "Please install Gmsh or set GMSH_BIN correctly in your .env file.\n"
            "Example:\nGMSH_BIN=C:\\Program Files\\Gmsh\\gmsh.exe"
        )

    geo_path = resolve_workspace_file(geo_name, "inputs", run_id or None)
    med_path = resolve_workspace_file(med_name, "inputs", run_id or None)

    if not os.path.exists(geo_path):
        return f"ERROR: Input file '{geo_path}' not found."

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
                "ERROR: Gmsh command failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout}"
            )

        if not os.path.exists(med_path):
            return f"ERROR: Gmsh finished but output file '{med_path}' was not created."

        return (
            "Meshing completed successfully.\n"
            f"MED file saved at: {med_path.resolve()}\n"
            "MED version forced to 4.0.0 for Code_Aster compatibility.\n\n"
            f"--- Gmsh Output (first 500 chars) ---\n{result.stdout[:500]}..."
        )

    except Exception as e:
        return f"ERROR: Exception while running Gmsh: {e}"
