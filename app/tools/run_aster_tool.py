# tools/run_aster_tool.py

import os
import subprocess

from langchain_core.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# Name of the solver image (can also be read from .env if you prefer)
SOLVER_IMAGE = os.getenv("SOLVER_IMAGE", "solver-image")

# Host path to the shared workspace directory.
# This is needed because `docker run -v <host_path>:/workspace ...`
HOST_WORKSPACE = os.getenv("HOST_WORKSPACE")  # e.g. /Users/you/project/workspace


@tool("run_code_aster", return_direct=True)
def run_code_aster_tool(
    geo_name: str = "model.geo",
    export_name: str = "model.export",
) -> str:
    """
    Run Gmsh + Code_Aster inside the `solver-image` Docker container.

    This tool assumes:
      - `workspace/` is a shared folder between the app container and host.
      - The host path to `workspace/` is provided via the HOST_WORKSPACE env var.
      - The solver Docker image name is given by SOLVER_IMAGE (default: solver-image).
    """

    # Paths inside the app container
    geo_path = os.path.join(WORKSPACE, geo_name)
    export_path = os.path.join(WORKSPACE, export_name)

    if not os.path.exists(geo_path):
        return f"❌ Error: geometry file not found:\n{os.path.abspath(geo_path)}"

    if not os.path.exists(export_path):
        return f"❌ Error: export file not found:\n{os.path.abspath(export_path)}"

    if not HOST_WORKSPACE:
        return (
            "❌ Error: HOST_WORKSPACE environment variable is not set.\n"
            "It must point to the *host* path of the shared `workspace` directory.\n"
            "Example (host):\n"
            '  export HOST_WORKSPACE="/Users/you/training/workspace"'
        )

    # Build the docker command that will run in the host Docker daemon
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{HOST_WORKSPACE}:/workspace",
        SOLVER_IMAGE,
        geo_name,
        export_name,
    ]

    try:
        print(f"🔧 Running solver container: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return (
                "⚠️ Solver (Gmsh + Code_Aster) failed.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        # At this point, the solver container should have created MED + Aster outputs
        job_name = os.path.splitext(os.path.basename(export_name))[0]
        mess = os.path.join(WORKSPACE, f"{job_name}.mess")
        resu = os.path.join(WORKSPACE, f"{job_name}.resu")
        med = os.path.join(WORKSPACE, "model.med")

        msg = [
            "✅ Solver container completed successfully (Gmsh + Code_Aster).",
            f"Geo:  {os.path.abspath(geo_path)}",
            (
                f"MED:  {os.path.abspath(med)}"
                if os.path.exists(med)
                else "MED: (not found)"
            ),
            (
                f"MESS: {os.path.abspath(mess)}"
                if os.path.exists(mess)
                else "MESS: (not found)"
            ),
            (
                f"RESU: {os.path.abspath(resu)}"
                if os.path.exists(resu)
                else "RESU: (not found)"
            ),
            "\n--- Solver STDOUT (first 500 chars) ---\n",
            result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""),
        ]
        return "\n".join(msg)

    except FileNotFoundError:
        return (
            "❌ Error: `docker` CLI not found inside app container.\n"
            "You need to install docker-cli in the app image and mount the Docker socket.\n"
        )

    except Exception as e:
        return f"❌ Unexpected error while running solver container: {e}"
