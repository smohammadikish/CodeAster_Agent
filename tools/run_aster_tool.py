# ----------------------------
#  tools/run_aster_tool.py
# ----------------------------

import os
import platform
import shutil
import subprocess

from langchain_core.tools import tool

from env_loader import load_env
from storage import ensure_run_dirs, resolve_workspace_file


def _read_job_name(export_path: str) -> str:
    """
    Read the Code_Aster job name from the export file when available.
    """
    try:
        with open(export_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == "P" and parts[1] == "nomjob":
                    return parts[2]
    except OSError:
        pass

    return os.path.splitext(os.path.basename(export_path))[0]


@tool("run_code_aster", return_direct=True)
def run_code_aster_tool(export_name: str = "model.export", run_id: str = "") -> str:
    """
    Run the Code_Aster solver using the generated export file.
    Works on Windows (.bat) and Linux (as_run).
    Produces .mess and .resu inside the run outputs folder.
    """

    _, as_run_bin, *_ = load_env()
    export_path = resolve_workspace_file(export_name, "inputs", run_id or None).resolve()

    if not os.path.exists(export_path):
        return f"âŒ Error: Export file not found:\n{export_path}"

    system = platform.system().lower()
    if system.startswith("win"):
        as_run_bin = as_run_bin or "as_run.bat"
        if not as_run_bin.lower().endswith(".bat"):
            as_run_bin += ".bat"
        cmd = [as_run_bin, str(export_path)]
    else:
        cmd = [as_run_bin or "as_run", str(export_path)]

    if shutil.which(cmd[0]) is None:
        return (
            f"âŒ Code_Aster binary not found: {cmd[0]}\n"
            "Please check AS_RUN_BIN in your .env file.\n"
            "Example:\n"
            "  AS_RUN_BIN=C:\\Program Files\\Code_Aster\\bin\\as_run.bat\n"
            "  or /usr/bin/as_run"
        )

    try:
        print(f"Running Code_Aster: {' '.join(cmd)}")

        run_root = ensure_run_dirs(run_id).root if run_id else export_path.parent
        result = subprocess.run(
            cmd,
            cwd=str(run_root),
            capture_output=True,
            text=True,
            shell=system.startswith("win"),
        )

        if result.returncode != 0:
            return (
                f"âš ï¸ Code_Aster run failed (exit {result.returncode}).\n"
                f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
            )

        job_name = _read_job_name(str(export_path))
        mess = resolve_workspace_file(f"{job_name}.mess", "outputs", run_id or None)
        resu = resolve_workspace_file(f"{job_name}.resu", "outputs", run_id or None)

        if not os.path.exists(mess) or not os.path.exists(resu):
            return (
                "âŒ Code_Aster finished but expected result files are missing.\n"
                f"Export: {export_path}\n"
                f"Expected MESS: {mess}\n"
                f"Expected RESU: {resu}\n"
                "\n--- First lines of output ---\n"
                + result.stdout[:500]
                + ("..." if len(result.stdout) > 500 else "")
            )

        msg = [
            "âœ… Code_Aster run completed successfully.",
            f"Export: {export_path}",
            f"MESS: {mess.resolve()}",
            f"RESU: {resu.resolve()}",
            "\n--- First lines of output ---\n",
            result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""),
        ]
        return "\n".join(msg)

    except FileNotFoundError:
        return f"âŒ as_run not found: {cmd[0]} â€” check AS_RUN_BIN in .env"

    except Exception as e:
        return f"âŒ Unexpected error running Code_Aster: {e}"
