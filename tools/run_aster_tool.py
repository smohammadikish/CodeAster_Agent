# ----------------------------
#  tools/run_aster_tool.py (final fixed)
# ----------------------------

import os
import platform
import shutil  # ✅ added for which()
import subprocess

from langchain_core.tools import tool

from env_loader import load_env

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("run_code_aster", return_direct=True)
def run_code_aster_tool(export_name: str = "model.export") -> str:
    """
    Run the Code_Aster solver using the generated export file.
    Works on Windows (.bat) and Linux (as_run).
    Produces .mess and .resu inside workspace folder.
    """

    # --- Load environment ---
    _, as_run_bin, *_ = load_env()
    export_path = os.path.abspath(os.path.join(WORKSPACE, export_name))

    if not os.path.exists(export_path):
        return f"❌ Error: Export file not found:\n{export_path}"

    # --- Platform command ---
    system = platform.system().lower()
    if system.startswith("win"):
        as_run_bin = as_run_bin or "as_run.bat"
        if not as_run_bin.lower().endswith(".bat"):
            as_run_bin += ".bat"
        cmd = [as_run_bin, export_path]
    else:
        cmd = [as_run_bin or "as_run", export_path]

    # --- Verify binary existence ---
    if shutil.which(cmd[0]) is None:
        return (
            f"❌ Code_Aster binary not found: {cmd[0]}\n"
            "Please check AS_RUN_BIN in your .env file.\n"
            "Example:\n"
            "  AS_RUN_BIN=C:\\Program Files\\Code_Aster\\bin\\as_run.bat\n"
            "  or /usr/bin/as_run"
        )

    try:
        print(f"🔧 Running Code_Aster: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            shell=system.startswith("win"),
        )

        if result.returncode != 0:
            return (
                f"⚠️ Code_Aster run failed (exit {result.returncode}).\n"
                f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
            )

        # --- Output files ---
        job_name = os.path.splitext(os.path.basename(export_name))[0]
        mess = os.path.join(WORKSPACE, f"{job_name}.mess")
        resu = os.path.join(WORKSPACE, f"{job_name}.resu")

        msg = [
            "✅ Code_Aster run completed successfully.",
            f"Export: {export_path}",
            f"MESS: {mess if os.path.exists(mess) else '(missing)'}",
            f"RESU: {resu if os.path.exists(resu) else '(missing)'}",
            "\n--- First lines of output ---\n",
            result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""),
        ]
        return "\n".join(msg)

    except FileNotFoundError:
        return f"❌ as_run not found: {cmd[0]} — check AS_RUN_BIN in .env"

    except Exception as e:
        return f"❌ Unexpected error running Code_Aster: {e}"
