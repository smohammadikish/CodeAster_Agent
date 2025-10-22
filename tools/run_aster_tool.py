# tools/run_aster_tool.py
import os
import platform
import subprocess

from langchain.tools import tool

from env_loader import \
    load_env  # To read the path to as_run or as_run.bat from .env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("run_code_aster", return_direct=True)
def run_code_aster_tool(export_name: str = "model.export") -> str:
    """
    Run Code_Aster analysis using an .export file.
    Automatically detects OS and runs 'as_run' (Linux/macOS)
    or 'as_run.bat --run' (Windows).

    Requires Code_Aster installed and its executable path in .env as AS_RUN_BIN (optional).
    """
    # Load path from .env (as_run / as_run.bat)
    _, as_run_bin = load_env()
    export_path = os.path.join(WORKSPACE, export_name)

    if not os.path.exists(export_path):
        return f"❌ Error: Export file '{export_path}' not found."

    # Detect platform
    system = platform.system().lower()  # 'windows', 'linux', 'darwin'
    if system.startswith("win"):
        # On Windows, Code_Aster runs via .bat
        cmd = [as_run_bin or "as_run.bat", "--run", export_path]
    else:
        cmd = [as_run_bin or "as_run", export_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE)

        if result.returncode != 0:
            return (
                "⚠️ Code_Aster run failed.\n"
                f"stderr:\n{result.stderr}\n"
                f"stdout:\n{result.stdout}"
            )

        # if job succeeded
        log_file = os.path.join(WORKSPACE, "beam_job.mess")
        resu_file = os.path.join(WORKSPACE, "beam_job.resu")

        msg = "✅ Code_Aster run completed successfully.\n"
        msg += f"Export file: {export_path}\n"
        msg += f"Check results in: {resu_file}\n"
        msg += f"Log file: {log_file}\n\n"
        msg += f"Output:\n{result.stdout[:600]}..."  # truncate long logs
        return msg

    except FileNotFoundError:
        return f"❌ Error: '{cmd[0]}' not found. Please check AS_RUN_BIN in .env."
    except Exception as e:
        return f"❌ Exception while running Code_Aster: {e}"
