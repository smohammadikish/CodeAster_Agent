# tools/visualize_tool.py
import os
import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# ==============================================================
# Tool 1️⃣ : Visualization of numeric .resu results
# ==============================================================

@tool("visualize_resu", return_direct=False)
def visualize_resu_tool(resu_name: str = "beam_job.resu", n_points: int = 100) -> str:
    """
    Read a .resu file and create a simple line plot of numeric results.
    The plot is saved as a .png file in the same workspace.
    """
    resu_path = os.path.join(WORKSPACE, resu_name)
    if not os.path.exists(resu_path):
        return f"❌ Error: result file '{resu_path}' not found."

    # Read all lines
    with open(resu_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Try to extract numeric values from each line
    data = []
    for line in lines:
        numbers = [float(s) for s in line.strip().split() if is_float(s)]
        if numbers:
            data.append(numbers)

    if len(data) == 0:
        return f"⚠️ No numeric data found in {resu_name}. Check if Code_Aster output is text format."

    data = np.array(data)
    y = data[:n_points, 0] if data.shape[0] > n_points else data[:, 0]
    x = np.arange(len(y))

    plt.figure(figsize=(9, 4))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.title(f"Preview of {resu_name}")
    plt.xlabel("Data point index")
    plt.ylabel("Value (e.g., displacement)")
    plt.grid(True)
    plt.tight_layout()

    png_name = os.path.splitext(resu_name)[0] + ".png"
    png_path = os.path.join(WORKSPACE, png_name)
    plt.savefig(png_path, dpi=150)
    plt.close()

    return f"✅ Visualization saved: {png_path} ({len(y)} points plotted)"


# Helper function to check if a string can be converted to float
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ==============================================================
# Tool 2️⃣ : Summarize .mess results (text-based summary)
# ==============================================================

@tool("summarize_results", return_direct=True)
def summarize_results_tool(mess_name: str = "result.mess") -> str:
    """
    Read and summarize key information from the Code_Aster .mess file.
    This step must be executed after the solver has finished running.
    It extracts convergence status, maximum stress, displacement, and key solver info.
    """
    """
    Read the .mess file and extract a short summary of main analysis results
    (e.g., max displacement, stress, or solver convergence info).
    """
    mess_path = os.path.join(WORKSPACE, mess_name)
    if not os.path.exists(mess_path):
        return f"❌ Error: message file '{mess_path}' not found."

    with open(mess_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Extract lines containing physical quantities or solver info
    summary = []
    for line in lines:
        if any(keyword in line.upper() for keyword in ["SIGM", "DEPL", "ERROR", "CONVERGENCE", "MAX"]):
            summary.append(line.strip())

    if not summary:
        return f"⚠️ No key results found in {mess_name}."

    # Return the last 10 relevant lines as a compact summary
    return "📄 Simulation summary:\n" + "\n".join(summary[-10:])
