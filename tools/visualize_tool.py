# tools/visualize_tool.py
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# ============================================================
# 🟦 TOOL 1 — DEFLECTION + ROTATION
# ============================================================
@tool("visualize_resu", return_direct=False)
def visualize_resu_tool(resu_name: str = "beam_analysis.resu") -> str:
    """
    Parse a Code_Aster .resu file and visualize:
      1) Beam deflection shape (DY)
      2) Beam rotation curve (DRZ)
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
                    node_id = int(parts[0][1:])
                    dx, dy, dz, drx, dry, drz = map(float, parts[1:7])
                    data.append((node_id, dy, drz))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No nodal displacement data found in '{resu_name}'."

    # Sort numerically by node ID
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    dy_values = np.array([d[1] for d in data])
    drz_values = np.array([d[2] for d in data])

    # Reorder (move node 2 to end)
    if 2 in node_ids:
        idx2 = np.where(node_ids == 2)[0][0]
        node_ids = np.concatenate((np.delete(node_ids, idx2), [2]))
        dy_values = np.concatenate((np.delete(dy_values, idx2), [dy_values[idx2]]))
        drz_values = np.concatenate((np.delete(drz_values, idx2), [drz_values[idx2]]))

    x_positions = np.linspace(0, 1, len(node_ids))

    # === Plot 1: Deflection ===
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(9, 3))
    plt.plot(x_positions, dy_values * 1000, "o-", lw=2.2)
    plt.title("Beam Deflected Shape (DY)", fontsize=13, weight="bold")
    plt.xlabel("Normalized Beam Length")
    plt.ylabel("Vertical Deflection [mm]")
    plt.grid(True, linestyle="--", alpha=0.7)
    max_disp = np.min(dy_values) * 1000
    tip_idx = np.argmin(dy_values)
    plt.scatter(x_positions[tip_idx], dy_values[tip_idx] * 1000, color="red", zorder=5)
    plt.text(
        x_positions[tip_idx],
        dy_values[tip_idx] * 1000 - 5,
        f"{max_disp:.1f} mm",
        color="red",
        va="top",
        fontsize=9,
    )
    plt.tight_layout()
    def_path = os.path.join(WORKSPACE, os.path.splitext(resu_name)[0] + "_def.png")
    plt.savefig(def_path, dpi=200)
    plt.close()

    # === Plot 2: Rotation ===
    plt.figure(figsize=(9, 3))
    plt.plot(x_positions, drz_values, "o-", lw=2.2, color="tab:red")
    plt.title("Beam Rotation Curve (DRZ)", fontsize=13, weight="bold")
    plt.xlabel("Normalized Beam Length")
    plt.ylabel("Rotation [rad]")
    plt.grid(True, linestyle="--", alpha=0.7)
    max_rot = np.min(drz_values)
    tip_idx = np.argmin(drz_values)
    plt.scatter(x_positions[tip_idx], drz_values[tip_idx], color="red", zorder=5)
    plt.text(
        x_positions[tip_idx],
        drz_values[tip_idx] - 0.01,
        f"{max_rot:.3f} rad",
        color="red",
        va="top",
        fontsize=9,
    )
    plt.tight_layout()
    rot_path = os.path.join(WORKSPACE, os.path.splitext(resu_name)[0] + "_rot.png")
    plt.savefig(rot_path, dpi=200)
    plt.close()

    return (
        f"✅ Visualizations saved:\n"
        f"📉 Deflection: {def_path}\n"
        f"📈 Rotation: {rot_path}\n"
        f"📊 Nodes: {len(node_ids)} | Max DY = {max_disp:.3f} mm | Max DRZ = {max_rot:.4f} rad"
    )


# ============================================================
# 🟥 TOOL 2 — STRESS CONTOUR MAP
# ============================================================
@tool("visualize_stress", return_direct=False)
def visualize_stress_tool(stress_name: str = "tab_stress.txt") -> str:
    """
    Parse Code_Aster stress table (IMPR_TABLE output)
    and visualize stress distribution (σxx) along beam length.
    """
    stress_path = os.path.join(WORKSPACE, stress_name)
    if not os.path.exists(stress_path):
        return f"❌ Error: file '{stress_path}' not found."

    node_pattern = re.compile(r"^\s*N\d+\s+")
    data = []
    with open(stress_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if node_pattern.match(line):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0][1:])
                        sxx = float(parts[1])  # assuming SIXX is 2nd column
                        data.append((node_id, sxx))
                    except Exception:
                        continue

    if len(data) == 0:
        return f"⚠️ No stress data found in '{stress_name}'."

    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    sxx = np.array([d[1] for d in data])
    x = np.linspace(0, 1, len(node_ids))

    # Color mapping
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=np.min(sxx), vmax=np.max(sxx))
    colors = cmap(norm(sxx))

    plt.figure(figsize=(8, 1.2))
    for i in range(len(x) - 1):
        plt.fill_between([x[i], x[i + 1]], 0, 1, color=colors[i])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.title("Stress Distribution (σxx)", fontsize=12, weight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="horizontal", pad=0.2)
    cbar.set_label("σxx [Pa]")

    out_path = os.path.join(WORKSPACE, "stress_map.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return (
        f"✅ Stress contour saved: {out_path}\n"
        f"📊 σxx range: {np.min(sxx):.2e} → {np.max(sxx):.2e} Pa"
    )
