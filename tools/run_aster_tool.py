# ----------------------------
#  tools/run_aster_tool.py
# ----------------------------

import os
import platform
import subprocess

from langchain_core.tools import tool  # ✅ Updated for LangChain 1.x

from env_loader import load_env  # Reads AS_RUN_BIN from .env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("run_code_aster", return_direct=True)
def run_code_aster_tool(export_name: str = "model.export") -> str:
    """
    Run the Code_Aster solver using the generated export file.
    Compatible with Code_Aster 14 (Windows/Linux/macOS).
    Produces .mess (log) and .resu (results) files.
    """

    # --- Read configuration from environment ---
    _, as_run_bin = load_env()
    export_path = os.path.abspath(os.path.join(WORKSPACE, export_name))

    # --- Validate that export file exists ---
    if not os.path.exists(export_path):
        return f"❌ Error: Export file not found:\n{export_path}"

    # --- Detect platform and choose proper command ---
    system = platform.system().lower()
    if system.startswith("win"):
        # Windows uses .bat wrapper
        cmd = [as_run_bin or "as_run.bat", "--run", export_path]
    else:
        # Linux / macOS direct execution
        cmd = [as_run_bin or "as_run", export_path]

    try:
        # --- Execute Code_Aster process ---
        result = subprocess.run(cmd, capture_output=True, text=True)

        # --- Handle errors ---
        if result.returncode != 0:
            return (
                "⚠️ Code_Aster run failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout}"
            )

        # --- Build paths for generated output files ---
        job_base = os.path.splitext(os.path.basename(export_name))[0]
        log_file = os.path.abspath(os.path.join(WORKSPACE, f"{job_base}.mess"))
        resu_file = os.path.abspath(os.path.join(WORKSPACE, f"{job_base}.resu"))

        # --- Prepare success message with brief output snippet ---
        msg = "✅ Code_Aster run completed successfully.\n"
        msg += f"Export file: {export_path}\n"
        msg += f"Results file: {resu_file}\n"
        msg += f"Log file: {log_file}\n\n"
        msg += f"--- First lines of output ---\n{result.stdout[:600]}..."
        return msg

    except FileNotFoundError:
        # --- Handle missing Code_Aster binary ---
        return (
            f"❌ Error: '{cmd[0]}' not found.\n"
            "Please check that Code_Aster is installed and AS_RUN_BIN in .env is correct.\n"
            "Example:\nAS_RUN_BIN=C:\\Program Files\\Code_Aster\\bin\\as_run.bat"
        )

    except Exception as e:
        # --- Generic unexpected exception ---
        return f"❌ Unexpected exception while running Code_Aster: {e}"
