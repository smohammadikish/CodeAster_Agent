# tools/mesher_gmsh.py
import os
import subprocess
import shutil
from langchain.tools import tool
from env_loader import load_env  # برای خواندن مسیر GMSH_BIN از .env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("mesh_with_gmsh", return_direct=False)
def mesh_with_gmsh_tool(
    geo_name: str = "model.geo",
    med_name: str = "model.med",
) -> str:
    """
    Use Gmsh CLI to mesh a 1D geometry (.geo) and export it to MED format (.med).
    Requires Gmsh to be installed and accessible via PATH or set in .env as GMSH_BIN.
    """
    gmsh_bin, _ = load_env()  # فقط مسیر gmsh برمی‌گرده، as_run فعلاً استفاده نمی‌شه
    gmsh_path = shutil.which(gmsh_bin)

    if gmsh_path is None:
        return f"❌ Error: '{gmsh_bin}' not found. Please install Gmsh or set GMSH_BIN in .env."

    geo_path = os.path.join(WORKSPACE, geo_name)
    med_path = os.path.join(WORKSPACE, med_name)

    if not os.path.exists(geo_path):
        return f"❌ Error: input file '{geo_path}' not found."

    # دستور Gmsh برای تولید مش 1D و خروجی MED
    cmd = [gmsh_path, geo_path, "-1", "-format", "med", "-o", med_path, "-v", "2"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # بررسی خطا
        if result.returncode != 0:
            return f"⚠️ Gmsh failed with error:\n{result.stderr}\n{result.stdout}"

        if not os.path.exists(med_path):
            return f"⚠️ Gmsh finished but output file '{med_path}' not found."

        return (
            f"✅ Meshing completed successfully.\n"
            f"MED file saved at: {med_path}\n"
            f"Gmsh output:\n{result.stdout}"
        )

    except Exception as e:
        return f"❌ Exception while running Gmsh: {e}"