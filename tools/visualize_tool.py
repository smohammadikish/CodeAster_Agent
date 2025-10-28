# tools/visualize_tool.py
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool("visualize_resu", return_direct=False)
def visualize_resu_tool(resu_name: str = "beam_analysis.resu") -> str:
    """
    Parse a Code_Aster .resu file and visualize nodal displacements (DY).
    Automatically reorders nodes so the fixed end (BASE) is first.
    """
    resu_path = os.path.join(WORKSPACE, resu_name)
    if not os.path.exists(resu_path):
        return f"❌ Error: result file '{resu_path}' not found."

    with open(resu_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    node_pattern = re.compile(r"^\s*N\d+\s+")
    data = []
    for line in lines:
        if node_pattern.match(line):
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    node_id = int(parts[0][1:])  # remove 'N'
                    dx, dy, dz, drx, dry, drz = map(float, parts[1:7])
                    data.append((node_id, dy))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No nodal displacement data (N1, N2, ...) found in '{resu_name}'."

    # Sort numerically
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    dy_values = np.array([d[1] for d in data])

    # If the first node has smaller deflection than the last, reverse order
    if abs(dy_values[0]) > abs(dy_values[-1]):
        node_ids = node_ids[::-1]
        dy_values = dy_values[::-1]

    # Plot beam deflection
    plt.figure(figsize=(9, 4))
    plt.plot(node_ids, dy_values * 1000, marker="o", linestyle="-", linewidth=1.4)
    plt.title("Beam Deflection (Vertical Displacement DY)")
    plt.xlabel("Node number (from fixed base to free tip)")
    plt.ylabel("Displacement DY [mm]")
    plt.grid(True)
    plt.tight_layout()

    png_name = os.path.splitext(resu_name)[0] + ".png"
    png_path = os.path.join(WORKSPACE, png_name)
    plt.savefig(png_path, dpi=150)
    plt.close()

    max_disp = np.min(dy_values)
    tip_node = node_ids[np.argmin(dy_values)]
    return (
        f"✅ Visualization saved: {png_path}\n"
        f"📊 Nodes: {len(node_ids)} | Max DY = {max_disp*1000:.3f} mm (Node N{tip_node})"
    )
