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
def visualize_stress_tool(stress_name: str = "tab_sie.txt") -> str:
    """
    Parse Code_Aster table from IMPR_TABLE (POST_RELEVE_T) output
    and visualize internal shear force (VY) and bending moment (MFZ) along the beam.
    Handles correct node order (N1 → N2 at the end).
    """
    stress_path = os.path.join(WORKSPACE, stress_name)
    if not os.path.exists(stress_path):
        return f"❌ Error: file '{stress_path}' not found."

    data = []
    with open(stress_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or "NOM_CHAM" in line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 16 and parts[1].startswith("N"):
                try:
                    node_id = int(parts[1][1:])  # remove "N"
                    vy = float(parts[11])  # shear force
                    mfz = float(parts[15])  # bending moment
                    data.append((node_id, vy, mfz))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No internal force data found in '{stress_name}'."

    # Sort by node ID
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    vy = np.array([d[1] for d in data])
    mfz = np.array([d[2] for d in data])

    # 🧩 Fix: Move node 2 (tip) to the end of arrays
    if 2 in node_ids:
        idx2 = np.where(node_ids == 2)[0][0]
        node_ids = np.concatenate((np.delete(node_ids, idx2), [2]))
        vy = np.concatenate((np.delete(vy, idx2), [vy[idx2]]))
        mfz = np.concatenate((np.delete(mfz, idx2), [mfz[idx2]]))

    x = np.linspace(0, 1, len(node_ids))

    # === Plot 1: Shear Force Diagram ===
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 2.2))
    plt.plot(x, vy / 1000, "o-", lw=2.2, color="tab:orange")
    plt.fill_between(x, vy / 1000, 0, color="tab:orange", alpha=0.3)
    plt.title("Shear Force Diagram (VY)", fontsize=13, weight="bold")
    plt.xlabel("Normalized Beam Length")
    plt.ylabel("VY [kN]")
    plt.grid(True, linestyle="--", alpha=0.7)
    v_path = os.path.join(WORKSPACE, "shear_diagram.png")
    plt.tight_layout()
    plt.savefig(v_path, dpi=200)
    plt.close()

    # === Plot 2: Bending Moment Diagram ===
    plt.figure(figsize=(8, 2.2))
    plt.plot(x, mfz / 1000, "o-", lw=2.2, color="tab:blue")
    plt.fill_between(x, mfz / 1000, 0, color="tab:blue", alpha=0.3)
    plt.title("Bending Moment Diagram (MFZ)", fontsize=13, weight="bold")
    plt.xlabel("Normalized Beam Length")
    plt.ylabel("MZ [kN·m]")
    plt.grid(True, linestyle="--", alpha=0.7)
    m_path = os.path.join(WORKSPACE, "moment_diagram.png")
    plt.tight_layout()
    plt.savefig(m_path, dpi=200)
    plt.close()

    return (
        f"✅ Internal force plots generated:\n"
        f"🟠 Shear: {v_path}\n"
        f"🔵 Moment: {m_path}\n"
        f"📊 Nodes: {len(node_ids)} | "
        f"VY range: {np.min(vy):.1f} → {np.max(vy):.1f} N | "
        f"MFZ range: {np.min(mfz):.1f} → {np.max(mfz):.1f} N·mm"
    )


# ============================================================
# 🟩 TOOL 3 — BENDING MOMENT MFY COLOR CONTOUR
# ============================================================
@tool("visualize_mfz_contour", return_direct=False)
def visualize_mfz_contour_tool(stress_name: str = "tab_sie.txt") -> str:
    """
    Visualize bending moment MFZ along beam length as a color contour.
    Reads POST_RELEVE_T (SIEF_ELNO) output table.
    """
    stress_path = os.path.join(WORKSPACE, stress_name)
    if not os.path.exists(stress_path):
        return f"❌ Error: file '{stress_path}' not found."

    data = []
    with open(stress_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or "NOM_CHAM" in line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 16 and parts[1].startswith("N"):
                try:
                    node_id = int(parts[1][1:])
                    x = float(parts[7])  # COOR_X (column 8)
                    mfz = float(parts[15])  # MFZ bending moment
                    data.append((node_id, x, mfz))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No MFZ data found in '{stress_name}'."

    # Sort by node ID
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    x = np.array([d[1] for d in data])
    mfz = np.array([d[2] for d in data])

    # ✅ Fix beam length based on min/max x-coordinates
    beam_length = np.max(x) - np.min(x)
    if beam_length < 0.1:  # fallback (if X are corrupted)
        beam_length = 4.0

    # Fix: move node 2 (tip) to the end
    if 2 in node_ids:
        idx2 = np.where(node_ids == 2)[0][0]
        node_ids = np.concatenate((np.delete(node_ids, idx2), [2]))
        x = np.concatenate((np.delete(x, idx2), [x[idx2]]))
        mfz = np.concatenate((np.delete(mfz, idx2), [mfz[idx2]]))

    # ✅ Convert from N·mm → kN·m
    mfz_kNm = mfz / 1e6 if np.max(np.abs(mfz)) > 1e5 else mfz / 1e3

    # Colormap
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=np.min(mfz_kNm), vmax=np.max(mfz_kNm))
    colors = cmap(norm(mfz_kNm))

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 1.0))
    for i in range(len(x) - 1):
        ax.fill_between([x[i], x[i + 1]], 0, 0.5, color=colors[i], linewidth=0)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, 0.5)
    ax.axis("off")
    ax.set_title("Bending Moment Contour (MFZ)", fontsize=13, weight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.25)
    cbar.set_label("MFZ [kN·m]", fontsize=10)

    out_path = os.path.join(WORKSPACE, "mfz_contour.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return (
        f"✅ MFZ contour saved: {out_path}\n"
        f"📊 MFZ range: {np.min(mfz_kNm):.2f} → {np.max(mfz_kNm):.2f} kN·m\n"
        f"📏 Beam length: {beam_length:.2f} m | Nodes: {len(node_ids)}"
    )
