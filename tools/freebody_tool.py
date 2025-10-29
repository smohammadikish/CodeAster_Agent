# tools/freebody_tool.py
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# =============================================================
#                      freebody
# ------------------------------------------------------------
#  1. Parse reaction file (tab_reac.txt)
# ------------------------------------------------------------
def _read_reactions(reac_path):
    """Parse tab_reac.txt from IMPR_TABLE output (Code_Aster)"""
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
                    fy = float(parts[node_index + 4])  # DY
                    mz = float(parts[-1])  # the last column DYZ
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
                except Exception as e:
                    continue
    return None


# ------------------------------------------------------------
#  2. Parse beam length from .geo file
# ------------------------------------------------------------
def _read_beam_length(geo_path):
    """Read beam length from Point(1) and Point(2) in .geo file"""
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
# 📙 3. Parse loads from .comm file
# ------------------------------------------------------------
def _read_loads(comm_path: str):
    """
    Parse Code_Aster .comm file and extract applied loads.
    Looks for 'FORCE_NODALE' or 'FORCE_POUTRE' blocks
    and finds FY=... values inside, even if multiline.
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

        # Start detection
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

        # Collect lines until block ends
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
# 📘 4. Drawing utilities
# ------------------------------------------------------------
def _draw_support(ax, x0, y0, support_type="fixed"):
    """Draw a support symbol at (x0,y0)"""
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
    """Draw vertical arrow"""
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
    """Draw applied loads (in red)"""
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
# 📗 5. Main visualization tool
# ------------------------------------------------------------
@tool("visualize_freebody", return_direct=False)
def visualize_freebody_tool(
    reac_name: str = "tab_reac.txt",
    support_type: str = "fixed",
    geo_name: str = "model.geo",
    comm_name: str = "model.comm",
    unit: str = "m",
) -> str:
    """Draw full free-body diagram including loads and reactions"""

    # --- Read beam length ---
    geo_path = os.path.join(WORKSPACE, geo_name)
    beam_length = _read_beam_length(geo_path)
    if beam_length is None:
        beam_length = 1.0

    # --- Read reactions and loads ---
    reac_path = os.path.join(WORKSPACE, reac_name)
    reac = _read_reactions(reac_path)
    if reac is None:
        return f"❌ Error: could not parse reactions from '{reac_path}'."

    comm_path = os.path.join(WORKSPACE, comm_name)
    loads = _read_loads(comm_path)

    # --- Start plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.set_facecolor("white")

    # Beam line
    ax.plot([0, beam_length], [0, 0], color="black", lw=3, solid_capstyle="round")

    # Support
    _draw_support(ax, x0=0.0, y0=0.0, support_type=support_type)

    # Reaction Force
    Fy = reac["Fy"]
    if abs(Fy) > 1e-9:
        _draw_vertical_arrow(
            ax, x=-0.05, y=0.0, magnitude=Fy, label=f"Fy={Fy/1000:.1f} kN"
        )

    # Reaction Moment
    if support_type == "fixed":
        Mz = reac["Mz"]
        if abs(Mz) > 1e-9:
            theta = np.linspace(np.deg2rad(85), np.deg2rad(275), 100)
            r = 0.18
            cx, cy = -0.10, 0.0
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)
            ax.plot(arc_x, arc_y, color="tab:blue", lw=2.4)
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
            text_x = cx - r - 0.12
            text_y = cy + 0.02
            M_value = Mz / 1000
            ax.text(
                text_x,
                text_y,
                f"M = {M_value:.1f} kN·mm",
                color="tab:blue",
                fontsize=10,
                va="center",
                ha="right",
                fontweight="bold",
            )

    # Draw applied loads (red)
    if loads:
        _draw_load(ax, beam_length, loads)

    # Dimension line
    arrow_y = -0.25
    ax.annotate(
        "",
        xy=(0, arrow_y),
        xytext=(beam_length, arrow_y),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="gray"),
    )
    ax.text(
        beam_length / 2,
        arrow_y - 0.05,
        f"L = {beam_length:.2f} {unit}",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )

    # Clean layout
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
        f"📏 Beam length: {beam_length:.2f} {unit}\n"
        f"📍 Support: {support_type}\n"
        f"🔁 Reactions at {reac['node']}: Fx={reac['Fx']:.2f} N, Fy={reac['Fy']:.2f} N, Mz={reac['Mz']:.2f} Nmm\n"
        f"🎯 Loads: {len(loads)} found from {comm_name}"
    )
