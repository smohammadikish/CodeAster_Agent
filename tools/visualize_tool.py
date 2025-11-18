# ----------------------------
#  tools/visualize_tool.py
# ----------------------------


import os
import re

import matplotlib.pyplot as plt
import numpy as np
from langchain_core.tools import tool  # ✅ Updated for LangChain 1.x

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

    # Extract nodal displacement data (N#, DX DY DZ DRX DRY DRZ)
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

    # Sort by node ID
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    dy_values = np.array([d[1] for d in data])
    drz_values = np.array([d[2] for d in data])

    # Fix order: move node 2 to the end (beam tip)
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
    and visualize internal shear force (VY) and bending moment (MFZ).
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
                    vy = float(parts[11])
                    mfz = float(parts[15])
                    data.append((node_id, vy, mfz))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No internal force data found in '{stress_name}'."

    # Sort nodes numerically
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    vy = np.array([d[1] for d in data])
    mfz = np.array([d[2] for d in data])

    # Move node 2 (tip) to the end
    if 2 in node_ids:
        idx2 = np.where(node_ids == 2)[0][0]
        node_ids = np.concatenate((np.delete(node_ids, idx2), [2]))
        vy = np.concatenate((np.delete(vy, idx2), [vy[idx2]]))
        mfz = np.concatenate((np.delete(mfz, idx2), [mfz[idx2]]))

    x = np.linspace(0, 1, len(node_ids))

    # === Shear Force Diagram ===
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

    # === Bending Moment Diagram ===
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
# 🟩 TOOL 3 — BENDING MOMENT MFZ COLOR CONTOUR
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
                    x = float(parts[7])
                    mfz = float(parts[15])
                    data.append((node_id, x, mfz))
                except Exception:
                    continue

    if len(data) == 0:
        return f"⚠️ No MFZ data found in '{stress_name}'."

    # Sort nodes numerically
    data.sort(key=lambda x: x[0])
    node_ids = np.array([d[0] for d in data])
    x = np.array([d[1] for d in data])
    mfz = np.array([d[2] for d in data])

    # Beam length check
    beam_length = np.max(x) - np.min(x)
    if beam_length < 0.1:
        beam_length = 4.0

    # Move node 2 to end
    if 2 in node_ids:
        idx2 = np.where(node_ids == 2)[0][0]
        node_ids = np.concatenate((np.delete(node_ids, idx2), [2]))
        x = np.concatenate((np.delete(x, idx2), [x[idx2]]))
        mfz = np.concatenate((np.delete(mfz, idx2), [mfz[idx2]]))

    # Convert N·mm → kN·m
    mfz_kNm = mfz / 1e6 if np.max(np.abs(mfz)) > 1e5 else mfz / 1e3

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=np.min(mfz_kNm), vmax=np.max(mfz_kNm))
    colors = cmap(norm(mfz_kNm))

    # Plot color contour
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


# ============================================================
# 🟧 TOOL 4 — FREE-BODY DIAGRAM
# ============================================================


# ------------------------------------------------------------
#  1. Parse reaction file (tab_reac.txt)
# ------------------------------------------------------------
def _read_reactions(reac_path):
    """Parse tab_reac.txt from IMPR_TABLE output (Code_Aster)."""
    if not os.path.exists(reac_path):
        return None

    with open(reac_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.search(r"\bN\d+\b", line):
                parts = line.strip().split()
                try:
                    node_index = next(
                        i for i, p in enumerate(parts) if re.match(r"N\d+", p)
                    )
                    node = parts[node_index]
                    fy = float(parts[node_index + 4])  # DY component
                    mz = float(parts[-1])  # Moment (DYZ column)
                    fx = (
                        float(parts[node_index + 3])
                        if len(parts) > node_index + 3
                        else 0.0
                    )
                    return {
                        "node": node,
                        "Fx": fx,
                        "Fy": fy,
                        "Fz": 0.0,
                        "Mx": 0.0,
                        "My": 0.0,
                        "Mz": mz,
                    }
                except Exception:
                    continue
    return None


# ------------------------------------------------------------
#  2. Parse beam length from .geo file
# ------------------------------------------------------------
def _read_beam_length(geo_path):
    """Read beam length from Point(1) and Point(2) in .geo file."""
    if not os.path.exists(geo_path):
        return None

    coords = {}
    point_pattern = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\};")
    with open(geo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = point_pattern.search(line)
            if m:
                pid = int(m.group(1))
                xyz = [float(v.strip()) for v in m.group(2).split(",")]
                coords[pid] = xyz

    if 1 in coords and 2 in coords:
        x1, y1, z1 = coords[1]
        x2, y2, z2 = coords[2]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return None


# ------------------------------------------------------------
#  3. Parse loads from .comm file
# ------------------------------------------------------------
def _read_loads(comm_path: str):
    """
    Parse Code_Aster .comm file and extract applied loads.
    Detects FORCE_NODALE or FORCE_POUTRE blocks and extracts FY values.
    """
    if not os.path.exists(comm_path):
        return []

    with open(comm_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    loads = []
    inside_force_nodale = False
    inside_force_poutre = False
    current_block = []

    for line in lines:
        u = line.upper()

        # Detect start of load blocks
        if "FORCE_NODALE" in u:
            inside_force_nodale = True
            inside_force_poutre = False
            current_block = [u]
            continue
        if "FORCE_POUTRE" in u:
            inside_force_poutre = True
            inside_force_nodale = False
            current_block = [u]
            continue

        # Collect block lines until closing parenthesis and semicolon
        if inside_force_nodale or inside_force_poutre:
            current_block.append(u)
            if ")" in u and ";" in u:
                block_text = " ".join(current_block)
                fy_match = re.search(
                    r"FY\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", block_text
                )
                if fy_match:
                    val = float(fy_match.group(1))
                    if inside_force_nodale:
                        loads.append({"type": "point", "Fy": val})
                    elif inside_force_poutre:
                        loads.append({"type": "uniform", "qy": val})
                inside_force_nodale = False
                inside_force_poutre = False
                current_block = []

    return loads


# ------------------------------------------------------------
#  4. Drawing utilities
# ------------------------------------------------------------
def _draw_support(ax, x0, y0, support_type="fixed"):
    """Draw a support symbol at (x0, y0)."""
    if support_type == "fixed":
        ax.add_patch(
            plt.Rectangle((x0 - 0.02, y0 - 0.1), 0.02, 0.2, color="black", alpha=0.6)
        )
        for yy in [y0 - 0.08, y0 - 0.04, y0, y0 + 0.04, y0 + 0.08]:
            ax.plot([x0 - 0.02, x0 - 0.04], [yy, yy - 0.02], color="black", lw=1)

    elif support_type == "pinned":
        tri_x = [x0, x0 - 0.05, x0 + 0.05]
        tri_y = [y0 - 0.05, y0 - 0.15, y0 - 0.15]
        ax.fill(tri_x, tri_y, color="black", alpha=0.6)

    elif support_type == "roller":
        tri_x = [x0, x0 - 0.05, x0 + 0.05]
        tri_y = [y0 - 0.05, y0 - 0.15, y0 - 0.15]
        ax.fill(tri_x, tri_y, color="black", alpha=0.6)
        circ_y = y0 - 0.18
        for cx in [x0 - 0.03, x0 + 0.03]:
            roller = plt.Circle((cx, circ_y), 0.015, color="black", alpha=0.6)
            ax.add_patch(roller)


def _draw_vertical_arrow(ax, x, y, magnitude, label, color="tab:blue"):
    """Draw vertical reaction arrow with label."""
    direction = 1 if magnitude >= 0 else -1
    dy = 0.25 * direction
    ax.arrow(
        x,
        y,
        0,
        dy,
        head_width=0.03,
        head_length=0.05,
        fc=color,
        ec=color,
        lw=2,
        length_includes_head=True,
    )
    ax.text(
        x + 0.05,
        y + dy * 0.6,
        label,
        color=color,
        fontsize=9,
        va="bottom" if direction > 0 else "top",
        ha="left",
    )


def _draw_load(ax, beam_length, loads):
    """Draw applied loads (in red)."""
    for load in loads:
        if load["type"] == "point":
            Fy = load["Fy"]
            ax.arrow(
                beam_length,
                0.0,
                0,
                -0.25,
                head_width=0.03,
                head_length=0.05,
                fc="tab:red",
                ec="tab:red",
                lw=2,
                length_includes_head=True,
            )
            ax.text(
                beam_length - 0.05,
                -0.35,
                f"{Fy/1000:.1f} kN",
                color="tab:red",
                fontsize=9,
                ha="right",
                va="top",
                fontweight="bold",
            )

        elif load["type"] == "uniform":
            qy = load["qy"]
            n_arrows = 6
            for i in range(1, n_arrows):
                x = beam_length * i / n_arrows
                ax.arrow(
                    x,
                    0.05,
                    0,
                    -0.2,
                    head_width=0.02,
                    head_length=0.03,
                    fc="tab:red",
                    ec="tab:red",
                    lw=1.8,
                    length_includes_head=True,
                )
            ax.text(
                beam_length / 2,
                -0.35,
                f"q = {qy/1000:.1f} kN/m",
                color="tab:red",
                fontsize=9,
                ha="center",
                va="top",
                fontweight="bold",
            )


# ------------------------------------------------------------
#  5. Main visualization tool
# ------------------------------------------------------------
# ============================================================
# 🟧 TOOL 4 — FREE-BODY DIAGRAM (FIXED FOR mm → m SCALING)
# ============================================================


@tool("visualize_freebody", return_direct=False)
def visualize_freebody_tool(
    reac_name: str = "tab_reac.txt",
    support_type: str = "fixed",
    geo_name: str = "model.geo",
    comm_name: str = "model.comm",
    unit: str = "m",
) -> str:
    """Draw full free-body diagram including supports, loads, and reactions."""

    # --- Read beam length in mm from .geo ---
    geo_path = os.path.join(WORKSPACE, geo_name)
    beam_length_mm = _read_beam_length(geo_path)
    if beam_length_mm is None:
        beam_length_mm = 1000.0  # fallback = 1 meter

    # Convert mm → m for plotting
    beam_length = beam_length_mm / 1000.0

    # --- Read reactions ---
    reac_path = os.path.join(WORKSPACE, reac_name)
    reac = _read_reactions(reac_path)
    if reac is None:
        return f"❌ Error: could not parse reactions from '{reac_path}'."

    # --- Read loads ---
    comm_path = os.path.join(WORKSPACE, comm_name)
    loads = _read_loads(comm_path)

    # --- Figure setup ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.set_facecolor("white")

    # Draw beam axis (in meters)
    ax.plot([0, beam_length], [0, 0], color="black", lw=3, solid_capstyle="round")

    # Draw support at x=0
    _draw_support(ax, x0=0.0, y0=0.0, support_type=support_type)

    # Draw reaction Fy
    Fy = reac["Fy"]
    if abs(Fy) > 1e-9:
        _draw_vertical_arrow(
            ax,
            x=-0.05,
            y=0.0,
            magnitude=Fy,
            label=f"Fy={Fy/1000:.1f} kN",
        )

    # Draw reaction moment Mz (fixed support only)
    if support_type == "fixed":
        Mz = reac["Mz"]
        if abs(Mz) > 1e-9:
            theta = np.linspace(np.deg2rad(85), np.deg2rad(275), 100)
            r = 0.18
            cx, cy = -0.10, 0.0

            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax.plot(arc_x, arc_y, color="tab:blue", lw=2.4)

            # Arrowhead for moment
            ax.arrow(
                arc_x[1],
                arc_y[1],
                arc_x[0] - arc_x[1],
                arc_y[0] - arc_y[1],
                head_width=0.04,
                head_length=0.05,
                fc="tab:blue",
                ec="tab:blue",
                lw=0,
                length_includes_head=True,
            )

            M_value = Mz / 1000.0  # convert N·mm → kN·mm
            ax.text(
                cx - r - 0.12,
                cy + 0.02,
                f"M = {M_value:.1f} N·mm",
                color="tab:blue",
                fontsize=10,
                va="center",
                ha="right",
                fontweight="bold",
            )

    # Draw loads (in meters)
    if loads:
        _draw_load(ax, beam_length, loads)

    # Dimension line under the beam
    arrow_y = -0.25
    ax.annotate(
        "",
        xy=(0, arrow_y),
        xytext=(beam_length, arrow_y),
        arrowprops=dict(
            arrowstyle="<->",
            lw=1.5,
            color="gray",
        ),
    )

    ax.text(
        beam_length / 2,
        arrow_y - 0.05,
        f"L = {beam_length:.2f} m",
        fontsize=10,
        color="gray",
        ha="center",
        va="top",
    )

    # Final plot settings
    ax.set_xlim(-0.35, beam_length + 0.25)
    ax.set_ylim(-0.45, 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(WORKSPACE, "freebody.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    # Return summary
    return (
        f"✅ Free-body diagram generated.\n"
        f"🖼 Image: {out_path}\n"
        f"📏 Beam length: {beam_length:.2f} m\n"
        f"📍 Support: {support_type}\n"
        f"🔁 Reactions at {reac['node']}: "
        f"Fx={reac['Fx']:.2f} N, Fy={reac['Fy']:.2f} N, Mz={reac['Mz']:.2f} Nmm\n"
        f"🎯 Loads: {len(loads)} found from {comm_name}"
    )
