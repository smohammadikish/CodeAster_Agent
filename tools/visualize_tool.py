import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_core.tools import tool

from storage import resolve_workspace_file
from units import (
    DEFAULT_UNIT_POLICY,
    convert_force,
    convert_length,
    convert_moment,
)


FIG_DPI = 260
FORCED_DISP_UNIT = "mm"


def _apply_plot_style(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    show_zero: bool = True,
    y_axis_at_x_zero: bool = False,
) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.minorticks_on()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    if y_axis_at_x_zero:
        ax.spines["left"].set_position(("data", 0.0))
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")
    if show_zero:
        ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)


def _ensure_zero_in_ylim(ax, pad_ratio: float = 0.08) -> None:
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-9)
    pad = pad_ratio * span
    new_min = min(y_min, -pad)
    new_max = max(y_max, pad)
    ax.set_ylim(new_min, new_max)


def _annotate_extreme(ax, x: np.ndarray, y: np.ndarray, unit_label: str, color: str) -> float:
    if y.size == 0:
        return 0.0
    idx = int(np.argmax(np.abs(y)))
    extreme = float(y[idx])
    span = max(float(np.max(y) - np.min(y)), 1e-9)
    y_offset = 0.14 * span if extreme < 0 else 0.10 * span
    va = "bottom" if extreme >= 0 else "top"
    text_y = extreme + y_offset if extreme >= 0 else extreme - y_offset
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x[idx] - x_max) < 0.06 * max(x_max - x_min, 1e-9):
        ha = "right"
    elif abs(x[idx] - x_min) < 0.06 * max(x_max - x_min, 1e-9):
        ha = "left"
    else:
        ha = "center"

    ax.scatter([x[idx]], [extreme], color=color, s=40, zorder=5)
    ax.text(
        x[idx],
        text_y,
        f"{extreme:.3f} {unit_label}",
        color=color,
        fontsize=9,
        ha=ha,
        va=va,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
    )
    return extreme


def _safe_node_id(value: str) -> int:
    token = value.strip().upper()
    if not token.startswith("N"):
        raise ValueError("Invalid node token")
    return int(token[1:])


def _node_order_key(node_id: int) -> tuple[int, int]:
    return (0, node_id)


def _parse_disp_state(disp_path: Path) -> dict[int, tuple[float, float]]:
    if not os.path.exists(disp_path):
        return {}

    node_state: dict[int, tuple[float, float]] = {}
    with open(disp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.search(r"\bN\d+\b", line) is None:
                continue
            parts = line.strip().split()
            try:
                node_index = next(i for i, token in enumerate(parts) if re.fullmatch(r"N\d+", token))
            except StopIteration:
                continue
            if len(parts) <= node_index + 1:
                continue
            try:
                node_id = _safe_node_id(parts[node_index])
                x_val = float(parts[node_index + 1])
                dy_val = float(parts[node_index + 4]) if len(parts) > node_index + 4 else 0.0
            except ValueError:
                continue
            node_state[node_id] = (x_val, dy_val)
    return node_state


def _parse_disp_rows(disp_path: Path) -> dict[int, float]:
    state = _parse_disp_state(disp_path)
    return {node_id: values[0] for node_id, values in state.items()}


def _parse_disp_table_rows(disp_path: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    if not os.path.exists(disp_path):
        return rows

    with open(disp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.search(r"\bN\d+\b", line) is None:
                continue
            parts = line.strip().split()
            try:
                node_index = next(i for i, token in enumerate(parts) if re.fullmatch(r"N\d+", token))
            except StopIteration:
                continue
            if len(parts) <= node_index + 8:
                continue
            try:
                node_id = _safe_node_id(parts[node_index])
                x_val = float(parts[node_index + 1])
                dy_val = float(parts[node_index + 4])
                drz_val = float(parts[node_index + 8])
            except ValueError:
                continue
            rows.append((node_id, x_val, dy_val, drz_val))
    return rows


def _build_axis_from_nodes(node_ids: list[int], run_id: str) -> np.ndarray:
    if not node_ids:
        return np.array([], dtype=float)

    disp_path = resolve_workspace_file("tab_disp.txt", "outputs", run_id or None)
    node_x = _parse_disp_rows(disp_path)
    if node_x:
        missing = [nid for nid in node_ids if nid not in node_x]
        if not missing:
            return np.array([node_x[nid] for nid in node_ids], dtype=float)

    geo_path = resolve_workspace_file("model.geo", "inputs", run_id or None)
    beam_length_raw, _ = _read_geo_model(geo_path)
    if beam_length_raw is None or beam_length_raw <= 0:
        beam_length_raw = float(max(len(node_ids) - 1, 1))

    ordered = sorted(node_ids)
    x_line = np.linspace(0.0, beam_length_raw, len(ordered))
    x_map = {nid: x_line[i] for i, nid in enumerate(ordered)}
    return np.array([x_map[nid] for nid in node_ids], dtype=float)


def _parse_sie_rows(stress_path: Path) -> list[dict[str, float]]:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []

    with open(stress_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" not in line:
                continue

            parts = [part.strip() for part in raw_line.split(";")]
            if "NOEUD" in parts and "VY" in parts and "MFZ" in parts:
                header = parts
                continue

            if header is None:
                continue

            if len(parts) < len(header):
                continue

            record = {header[i]: parts[i] for i in range(len(header))}
            node_token = record.get("NOEUD", "")
            if not node_token.startswith("N"):
                continue

            try:
                rows.append(
                    {
                        "node": _safe_node_id(node_token),
                        "x": float(record.get("COOR_X", "nan")),
                        "vy": float(record.get("VY", "nan")),
                        "mfz": float(record.get("MFZ", "nan")),
                    }
                )
            except (ValueError, TypeError):
                continue

    rows = [row for row in rows if not np.isnan(row["vy"]) and not np.isnan(row["mfz"])]
    if rows and any(not np.isnan(r["x"]) for r in rows):
        rows.sort(key=lambda item: (float(item["x"]) if not np.isnan(item["x"]) else float("inf"), int(item["node"])))
    else:
        rows.sort(key=lambda item: int(item["node"]))
    return rows


def _read_reactions(reac_path: Path) -> dict[str, float | str] | None:
    reactions = _read_reactions_all(reac_path)
    return reactions[0] if reactions else None


def _read_reactions_all(reac_path: Path) -> list[dict[str, float | str]]:
    if not os.path.exists(reac_path):
        return []

    reactions: list[dict[str, float | str]] = []

    with open(reac_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.search(r"\bN\d+\b", line) is None:
                continue

            parts = line.strip().split()
            try:
                node_index = next(i for i, token in enumerate(parts) if re.fullmatch(r"N\d+", token))
            except StopIteration:
                continue

            if len(parts) < node_index + 9:
                continue

            try:
                x = float(parts[node_index + 1])
                fx = float(parts[node_index + 3])
                fy = float(parts[node_index + 4])
                mz = float(parts[node_index + 8])
            except ValueError:
                continue

            reactions.append(
                {
                    "node": parts[node_index],
                    "X": x,
                    "Fx": fx,
                    "Fy": fy,
                    "Fz": 0.0,
                    "Mx": 0.0,
                    "My": 0.0,
                    "Mz": mz,
                }
            )

    reactions.sort(key=lambda item: float(item["X"]))
    return reactions


def _read_geo_model(geo_path: Path) -> tuple[float | None, dict[str, float]]:
    if not os.path.exists(geo_path):
        return None, {}

    point_pattern = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\};", flags=re.IGNORECASE)
    physical_point_pattern = re.compile(
        r"Physical\s+Point\(\"([^\"]+)\"\)\s*=\s*\{([^}]+)\};",
        flags=re.IGNORECASE,
    )

    point_x: dict[int, float] = {}
    group_x: dict[str, float] = {}

    with open(geo_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        match = point_pattern.search(line)
        if not match:
            continue
        pid = int(match.group(1))
        xyz = [v.strip() for v in match.group(2).split(",")]
        if len(xyz) < 1:
            continue
        try:
            point_x[pid] = float(xyz[0])
        except ValueError:
            continue

    if point_x:
        xs = list(point_x.values())
        beam_length = max(xs) - min(xs)
    else:
        beam_length = None

    for line in lines:
        match = physical_point_pattern.search(line)
        if not match:
            continue
        name = match.group(1).strip().upper()
        ids = []
        for token in match.group(2).split(","):
            token = token.strip()
            if token.isdigit():
                ids.append(int(token))
        mapped_x = [point_x[idx] for idx in ids if idx in point_x]
        if mapped_x:
            group_x[name] = float(np.mean(mapped_x))

    return beam_length, group_x


def _read_loads(comm_path: Path) -> list[dict[str, float | str]]:
    if not os.path.exists(comm_path):
        return []

    with open(comm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().upper()

    loads: list[dict[str, float | str]] = []

    def extract_blocks(keyword: str) -> list[str]:
        blocks: list[str] = []
        pattern = re.compile(rf"{keyword}\s*=\s*_F\s*\(", flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            open_idx = match.end() - 1  # index of '('
            depth = 1
            i = open_idx + 1
            while i < len(text) and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            if depth == 0:
                blocks.append(text[open_idx + 1 : i - 1])
        return blocks

    for block in extract_blocks("FORCE_NODALE"):
        fy_match = re.search(r"FY\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", block)
        if fy_match is None:
            continue
        group_match = re.search(r"GROUP_NO\s*=\s*\(([^)]*)\)", block)
        group_name = ""
        if group_match:
            candidates = re.findall(r"'([^']+)'", group_match.group(1))
            if candidates:
                group_name = candidates[0].strip().upper()
        loads.append({"type": "point", "Fy": float(fy_match.group(1)), "group": group_name})

    for block in extract_blocks("FORCE_POUTRE"):
        fy_match = re.search(r"FY\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", block)
        if fy_match is None:
            continue
        loads.append({"type": "uniform", "qy": float(fy_match.group(1))})

    return loads


def _read_support_groups(comm_path: Path) -> list[dict[str, str]]:
    if not os.path.exists(comm_path):
        return []

    with open(comm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().upper()

    supports: list[dict[str, str]] = []

    def extract_balanced(inner_text: str, open_idx: int) -> tuple[str, int] | None:
        depth = 1
        i = open_idx + 1
        while i < len(inner_text) and depth > 0:
            if inner_text[i] == "(":
                depth += 1
            elif inner_text[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            return None
        return inner_text[open_idx + 1 : i - 1], i

    def extract_f_blocks(inner_text: str) -> list[str]:
        out: list[str] = []
        pattern_f = re.compile(r"_F\s*\(", flags=re.IGNORECASE)
        for m in pattern_f.finditer(inner_text):
            open_idx = m.end() - 1
            block = extract_balanced(inner_text, open_idx)
            if block:
                out.append(block[0])
        return out

    blocks: list[str] = []
    pattern = re.compile(r"DDL_IMPO\s*=", flags=re.IGNORECASE)
    for match in pattern.finditer(text):
        i = match.end()
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            continue
        if text[i : i + 3] == "_F(":
            block = extract_balanced(text, i + 2)
            if block:
                blocks.append(block[0])
        elif text[i] == "(":
            outer = extract_balanced(text, i)
            if outer:
                blocks.extend(extract_f_blocks(outer[0]))

    for block in blocks:
        group_match = re.search(r"GROUP_NO\s*=\s*\(([^)]*)\)", block)
        if not group_match:
            continue
        names = re.findall(r"'([^']+)'", group_match.group(1))
        if not names:
            continue
        block_clean = block.replace(" ", "")
        has_dx = "DX=" in block_clean
        has_dy = "DY=" in block_clean
        has_drz = "DRZ=" in block_clean
        if has_dx and has_dy and has_drz:
            s_type = "fixed"
        elif has_dy and has_dx:
            s_type = "pinned"
        elif has_dy:
            s_type = "roller"
        else:
            s_type = "pinned"
        supports.append({"group": names[0].strip().upper(), "type": s_type})
    return supports


def _draw_support(ax, x0: float, y0: float, support_type: str, scale: float) -> None:
    h = scale * 0.8
    w = scale * 0.35

    if support_type == "fixed":
        ax.add_patch(plt.Rectangle((x0 - w, y0 - h), w, 2 * h, color="#2C3E50", alpha=0.85))
        n_hatch = 7
        for i in range(n_hatch):
            yy = y0 - h + (2 * h * i / (n_hatch - 1))
            ax.plot([x0 - w, x0 - w - 0.6 * w], [yy, yy - 0.25 * h], color="#2C3E50", lw=1.0)

    elif support_type == "pinned":
        tri_x = [x0, x0 - w, x0 + w]
        tri_y = [y0 - 0.1 * h, y0 - h, y0 - h]
        ax.fill(tri_x, tri_y, color="#2C3E50", alpha=0.88)

    elif support_type == "roller":
        tri_x = [x0, x0 - w, x0 + w]
        tri_y = [y0 - 0.1 * h, y0 - h, y0 - h]
        ax.fill(tri_x, tri_y, color="#2C3E50", alpha=0.88)
        circle_y = y0 - h - 0.22 * h
        for cx in (x0 - 0.5 * w, x0 + 0.5 * w):
            ax.add_patch(plt.Circle((cx, circle_y), 0.18 * w, color="#2C3E50", alpha=0.88))


def _draw_vertical_arrow(ax, x: float, y: float, magnitude: float, label: str, scale: float, color: str) -> None:
    direction = 1 if magnitude >= 0 else -1
    dy = 1.45 * scale * direction
    head_w = 0.24 * scale
    head_l = 0.30 * scale

    ax.arrow(
        x,
        y,
        0,
        dy,
        head_width=head_w,
        head_length=head_l,
        fc=color,
        ec=color,
        lw=2.1,
        length_includes_head=True,
        zorder=5,
    )
    ax.text(
        x + 0.45 * scale,
        y + 0.60 * dy,
        label,
        color=color,
        fontsize=9,
        va="bottom" if direction > 0 else "top",
        ha="left",
        fontweight="bold",
    )


def _draw_loads(
    ax,
    beam_length: float,
    loads: list[dict[str, float | str]],
    scale: float,
    group_pos_visual: dict[str, float],
) -> None:
    for load in loads:
        load_type = str(load.get("type", ""))

        if load_type == "point":
            fy = float(load.get("Fy", 0.0))
            group_name = str(load.get("group", "")).upper()
            if group_name and group_name in group_pos_visual:
                x_loc = group_pos_visual[group_name]
            elif group_name == "MID":
                x_loc = 0.5 * beam_length
            else:
                x_loc = beam_length

            display_force = convert_force(
                fy,
                DEFAULT_UNIT_POLICY.solver_force_unit,
                DEFAULT_UNIT_POLICY.visualization_force_unit,
            )

            _draw_vertical_arrow(
                ax,
                x=x_loc,
                y=0.60 * scale,
                magnitude=-abs(fy),
                label=f"P = {display_force:.2f} {DEFAULT_UNIT_POLICY.visualization_force_unit}",
                scale=0.82 * scale,
                color="#C0392B",
            )

        elif load_type == "uniform":
            qy = float(load.get("qy", 0.0))
            display_force = convert_force(
                qy,
                DEFAULT_UNIT_POLICY.solver_force_unit,
                DEFAULT_UNIT_POLICY.visualization_force_unit,
            )

            n_arrows = 9
            x_positions = np.linspace(0.08 * beam_length, 0.92 * beam_length, n_arrows)
            top_y = 1.35 * scale
            bot_y = 0.05 * scale
            ax.plot([x_positions[0], x_positions[-1]], [top_y, top_y], color="#C0392B", lw=1.8, alpha=0.9)
            for x_i in x_positions:
                ax.arrow(
                    x_i,
                    top_y,
                    0,
                    bot_y - top_y,
                    head_width=0.20 * scale,
                    head_length=0.24 * scale,
                    fc="#C0392B",
                    ec="#C0392B",
                    lw=2.0,
                    length_includes_head=True,
                    zorder=4,
                )
            ax.text(
                0.5 * beam_length,
                top_y + 0.16 * scale,
                f"q = {display_force:.2f} {DEFAULT_UNIT_POLICY.visualization_force_unit}/{DEFAULT_UNIT_POLICY.visualization_length_unit}",
                color="#C0392B",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
            )


@tool("visualize_resu", return_direct=False)
def visualize_resu_tool(resu_name: str = "beam_job.resu", run_id: str = "") -> str:
    """
    Parse Code_Aster .resu and generate:
    1) deflection plot (DY)
    2) rotation plot (DRZ)
    """
    resu_path = resolve_workspace_file(resu_name, "outputs", run_id or None)
    if not os.path.exists(resu_path):
        return f"Error: result file '{resu_path}' not found."

    data: list[tuple[int, float, float, float]] = []

    disp_path = resolve_workspace_file("tab_disp.txt", "outputs", run_id or None)
    disp_rows = _parse_disp_table_rows(disp_path)
    if disp_rows:
        data = disp_rows
    else:
        node_pattern = re.compile(r"^\s*(N\d+)\s+")
        with open(resu_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if node_pattern.match(line) is None:
                    continue
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                try:
                    node_id = _safe_node_id(parts[0])
                    dy = float(parts[2])
                    drz = float(parts[6])
                except ValueError:
                    continue
                data.append((node_id, float("nan"), dy, drz))

    if not data:
        return f"Warning: no nodal displacement data found in '{resu_name}'."

    if any(not np.isnan(d[1]) for d in data):
        data.sort(key=lambda row: (row[1] if not np.isnan(row[1]) else float("inf"), row[0]))
    else:
        data.sort(key=lambda row: row[0])

    node_ids = np.array([d[0] for d in data], dtype=int)
    dy_values = np.array([d[2] for d in data], dtype=float)
    drz_values = np.array([d[3] for d in data], dtype=float)

    if any(not np.isnan(d[1]) for d in data):
        x_raw = np.array([d[1] for d in data], dtype=float)
    else:
        x_raw = _build_axis_from_nodes(node_ids.tolist(), run_id)

    x_positions = np.array(
        [
            convert_length(
                value,
                DEFAULT_UNIT_POLICY.solver_coordinate_unit,
                DEFAULT_UNIT_POLICY.visualization_length_unit,
            )
            for value in x_raw
        ],
        dtype=float,
    )

    dy_display = np.array(
        [
            convert_length(
                value,
                DEFAULT_UNIT_POLICY.solver_displacement_unit,
                FORCED_DISP_UNIT,
            )
            for value in dy_values
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.plot(x_positions, dy_display, color="#1f77b4", lw=2.4, marker="o", ms=4)
    _apply_plot_style(
        ax,
        title="Deflection Diagram (DY)",
        xlabel=f"Beam Axis X [{DEFAULT_UNIT_POLICY.visualization_length_unit}]",
        ylabel=f"Deflection [{FORCED_DISP_UNIT}]",
        show_zero=True,
        y_axis_at_x_zero=True,
    )
    ax.margins(y=0.22)
    _ensure_zero_in_ylim(ax)
    max_disp = _annotate_extreme(
        ax,
        x_positions,
        dy_display,
        FORCED_DISP_UNIT,
        color="#1f77b4",
    )
    span = float(np.max(x_positions) - np.min(x_positions)) if len(x_positions) > 1 else 0.0
    x_at_max = float(x_positions[int(np.nanargmax(np.abs(dy_display)))]) if len(x_positions) > 0 else 0.0
    load_case = "cantilever"
    support_defs = _read_support_groups(resolve_workspace_file("model.comm", "inputs", run_id or None))
    if support_defs:
        unique_groups = {s["group"] for s in support_defs}
        if len(unique_groups) >= 2:
            load_case = "simply_supported"
        elif any(s["type"] == "fixed" for s in support_defs):
            load_case = "cantilever"
    max_loc_note = "tip controls" if load_case == "cantilever" else "midspan/load-point controls"
    # Serviceability range by beam type (common engineering practice defaults).
    if load_case == "cantilever":
        n_min, n_max = 125.0, 180.0
        formula_note = "Cantilever: delta_max = F*L^3/(3*E*I)"
    else:
        n_min, n_max = 200.0, 360.0
        formula_note = "Simply supported: delta_max = F*L^3/(48*E*I)"
    span_disp_unit = convert_length(
        span,
        DEFAULT_UNIT_POLICY.visualization_length_unit,
        FORCED_DISP_UNIT,
    ) if span > 0.0 else 0.0
    delta_lim_lo = span_disp_unit / n_max if span_disp_unit > 0.0 else 0.0
    delta_lim_hi = span_disp_unit / n_min if span_disp_unit > 0.0 else 0.0
    status_note = (
        "OK"
        if delta_lim_lo <= abs(max_disp) <= delta_lim_hi
        else ("Too flexible" if abs(max_disp) > delta_lim_hi else "Very stiff")
    )
    ax.text(
        0.015,
        0.97,
        (
            f"Limit range: L/{int(n_max)} to L/{int(n_min)} -> "
            f"delta in [{delta_lim_lo:.3f}, {delta_lim_hi:.3f}] {FORCED_DISP_UNIT}\n"
            f"Result: max|DY| = {abs(max_disp):.3f} {FORCED_DISP_UNIT} ({status_note})"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9aa0a6", "alpha": 0.9},
    )

    def_path = resolve_workspace_file(f"{Path(resu_name).stem}_def.png", "viz", run_id or None)
    fig.tight_layout()
    fig.savefig(def_path, dpi=FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.plot(x_positions, drz_values, color="#d62728", lw=2.4, marker="o", ms=4)
    _apply_plot_style(
        ax,
        title="Rotation Diagram (DRZ)",
        xlabel=f"Beam Axis X [{DEFAULT_UNIT_POLICY.visualization_length_unit}]",
        ylabel=f"Rotation [{DEFAULT_UNIT_POLICY.visualization_rotation_unit}]",
        show_zero=True,
        y_axis_at_x_zero=True,
    )
    ax.margins(y=0.22)
    _ensure_zero_in_ylim(ax)
    max_rot = _annotate_extreme(
        ax,
        x_positions,
        drz_values,
        DEFAULT_UNIT_POLICY.visualization_rotation_unit,
        color="#d62728",
    )

    rot_path = resolve_workspace_file(f"{Path(resu_name).stem}_rot.png", "viz", run_id or None)
    fig.tight_layout()
    fig.savefig(rot_path, dpi=FIG_DPI)
    plt.close(fig)

    return (
        "Visualizations saved:\n"
        f"- Deflection: {def_path}\n"
        f"- Rotation: {rot_path}\n"
        f"- Nodes: {len(node_ids)} | "
        f"max |DY| = {abs(max_disp):.3f} {FORCED_DISP_UNIT} | "
        f"max |DRZ| = {abs(max_rot):.4f} {DEFAULT_UNIT_POLICY.visualization_rotation_unit}"
    )


@tool("visualize_stress", return_direct=False)
def visualize_stress_tool(stress_name: str = "tab_sie.txt", run_id: str = "") -> str:
    """
    Parse Code_Aster SIEF_ELNO table and plot:
    1) Shear force diagram (VY)
    2) Bending moment diagram (MFZ)
    """
    stress_path = resolve_workspace_file(stress_name, "outputs", run_id or None)
    if not os.path.exists(stress_path):
        return f"Error: file '{stress_path}' not found."

    rows = _parse_sie_rows(stress_path)
    if not rows:
        return f"Warning: no internal force data found in '{stress_name}'."

    x_raw = np.array([row["x"] for row in rows], dtype=float)
    if np.isnan(x_raw).all():
        node_ids = [int(row["node"]) for row in rows]
        x_raw = _build_axis_from_nodes(node_ids, run_id)
        x_label = f"Beam Axis X [{DEFAULT_UNIT_POLICY.visualization_length_unit}]"
    else:
        x_raw = np.nan_to_num(x_raw, nan=float(np.nanmean(x_raw)))
        x_label = f"Beam Axis X [{DEFAULT_UNIT_POLICY.visualization_length_unit}]"

    x_display = np.array(
        [
            convert_length(
                value,
                DEFAULT_UNIT_POLICY.solver_coordinate_unit,
                DEFAULT_UNIT_POLICY.visualization_length_unit,
            )
            for value in x_raw
        ],
        dtype=float,
    )

    vy = np.array([row["vy"] for row in rows], dtype=float)
    mfz = np.array([row["mfz"] for row in rows], dtype=float)

    vy_display = np.array(
        [
            convert_force(
                value,
                DEFAULT_UNIT_POLICY.solver_force_unit,
                DEFAULT_UNIT_POLICY.visualization_force_unit,
            )
            for value in vy
        ],
        dtype=float,
    )
    mfz_display = np.array(
        [
            convert_moment(
                value,
                DEFAULT_UNIT_POLICY.solver_moment_unit,
                DEFAULT_UNIT_POLICY.visualization_moment_unit,
            )
            for value in mfz
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.plot(x_display, vy_display, color="#E67E22", lw=2.3)
    ax.fill_between(x_display, vy_display, 0.0, color="#F5CBA7", alpha=0.75)
    _apply_plot_style(
        ax,
        title="Shear Force Diagram (VY)",
        xlabel=x_label,
        ylabel=f"VY [{DEFAULT_UNIT_POLICY.visualization_force_unit}]",
        show_zero=True,
        y_axis_at_x_zero=True,
    )
    _ensure_zero_in_ylim(ax)

    v_path = resolve_workspace_file("shear_diagram.png", "viz", run_id or None)
    fig.tight_layout()
    fig.savefig(v_path, dpi=FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.plot(x_display, mfz_display, color="#2471A3", lw=2.3)
    ax.fill_between(x_display, mfz_display, 0.0, color="#AED6F1", alpha=0.78)
    _apply_plot_style(
        ax,
        title="Bending Moment Diagram (MFZ)",
        xlabel=x_label,
        ylabel=f"MFZ [{DEFAULT_UNIT_POLICY.visualization_moment_unit}]",
        show_zero=True,
        y_axis_at_x_zero=True,
    )
    _ensure_zero_in_ylim(ax)

    m_path = resolve_workspace_file("moment_diagram.png", "viz", run_id or None)
    fig.tight_layout()
    fig.savefig(m_path, dpi=FIG_DPI)
    plt.close(fig)

    return (
        "Internal force plots generated:\n"
        f"- Shear: {v_path}\n"
        f"- Moment: {m_path}\n"
        f"- Nodes: {len(rows)} | "
        f"VY range: {float(np.min(vy_display)):.3f} to {float(np.max(vy_display)):.3f} {DEFAULT_UNIT_POLICY.visualization_force_unit} | "
        f"MFZ range: {float(np.min(mfz_display)):.3f} to {float(np.max(mfz_display)):.3f} {DEFAULT_UNIT_POLICY.visualization_moment_unit}"
    )


@tool("visualize_mfz_contour", return_direct=False)
def visualize_mfz_contour_tool(stress_name: str = "tab_sie.txt", run_id: str = "") -> str:
    """Visualize bending moment MFZ on a schematic beam with color contour."""
    stress_path = resolve_workspace_file(stress_name, "outputs", run_id or None)
    if not os.path.exists(stress_path):
        return f"Error: file '{stress_path}' not found."

    rows = _parse_sie_rows(stress_path)
    if not rows:
        return f"Warning: no MFZ data found in '{stress_name}'."

    x_raw = np.array([row["x"] for row in rows], dtype=float)
    if np.isnan(x_raw).all():
        node_ids = [int(row["node"]) for row in rows]
        x_raw = _build_axis_from_nodes(node_ids, run_id)
    else:
        x_raw = np.nan_to_num(x_raw, nan=float(np.nanmean(x_raw)))

    x_display = np.array(
        [
            convert_length(
                value,
                DEFAULT_UNIT_POLICY.solver_coordinate_unit,
                DEFAULT_UNIT_POLICY.visualization_length_unit,
            )
            for value in x_raw
        ],
        dtype=float,
    )

    mfz_display = np.array(
        [
            convert_moment(
                row["mfz"],
                DEFAULT_UNIT_POLICY.solver_moment_unit,
                DEFAULT_UNIT_POLICY.visualization_moment_unit,
            )
            for row in rows
        ],
        dtype=float,
    )

    beam_length = float(np.max(x_display) - np.min(x_display))
    if beam_length < 1e-9:
        beam_length = 1.0

    node_ids = [int(row["node"]) for row in rows]
    disp_state = _parse_disp_state(resolve_workspace_file("tab_disp.txt", "outputs", run_id or None))
    if disp_state and all(node_id in disp_state for node_id in node_ids):
        dy_solver = np.array([disp_state[node_id][1] for node_id in node_ids], dtype=float)
    else:
        dy_solver = np.zeros(len(node_ids), dtype=float)
    dy_mm = np.array(
        [
            convert_length(value, DEFAULT_UNIT_POLICY.solver_displacement_unit, FORCED_DISP_UNIT)
            for value in dy_solver
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    cmap = plt.get_cmap("RdYlBu_r")
    v_abs = max(abs(float(np.min(mfz_display))), abs(float(np.max(mfz_display))), 1e-9)
    norm = plt.Normalize(vmin=-v_abs, vmax=v_abs)

    x_min = float(np.min(x_display))
    x_max = float(np.max(x_display))
    beam_h = max(0.08 * beam_length, 0.12)
    y_base = 0.0
    center_ref = y_base + 0.5 * beam_h
    max_abs_dy = float(np.max(np.abs(dy_mm))) if dy_mm.size > 0 else 0.0
    if max_abs_dy > 1e-12:
        defl_scale = 0.65 * beam_h / max_abs_dy
        dy_plot = dy_mm * defl_scale
    else:
        dy_plot = np.zeros_like(dy_mm)
    y_center = center_ref + dy_plot

    for i in range(len(x_display) - 1):
        x0 = float(x_display[i])
        x1 = float(x_display[i + 1])
        y0 = float(y_center[i])
        y1 = float(y_center[i + 1])
        val = 0.5 * (mfz_display[i] + mfz_display[i + 1])
        poly_x = [x0, x1, x1, x0]
        poly_y = [y0 - 0.5 * beam_h, y1 - 0.5 * beam_h, y1 + 0.5 * beam_h, y0 + 0.5 * beam_h]
        ax.fill(poly_x, poly_y, color=cmap(norm(val)), zorder=2, linewidth=0)

    top = y_center + 0.5 * beam_h
    bot = y_center - 0.5 * beam_h
    ax.plot(x_display, top, color="#2E4053", lw=1.2, zorder=3)
    ax.plot(x_display, bot, color="#2E4053", lw=1.2, zorder=3)
    ax.plot(x_display, y_center, color="#0B5345", lw=1.4, ls="--", zorder=4, label=f"Deflection shape (scaled, DY in {FORCED_DISP_UNIT})")

    _, group_pos_solver = _read_geo_model(resolve_workspace_file("model.geo", "inputs", run_id or None))
    group_pos_visual = {
        name: convert_length(
            x_val,
            DEFAULT_UNIT_POLICY.solver_coordinate_unit,
            DEFAULT_UNIT_POLICY.visualization_length_unit,
        )
        for name, x_val in group_pos_solver.items()
    }
    support_defs = _read_support_groups(resolve_workspace_file("model.comm", "inputs", run_id or None))
    support_scale = max(1.05 * beam_h, 0.18)
    for support in support_defs:
        group_name = support["group"]
        if group_name in group_pos_visual:
            sx = group_pos_visual[group_name]
        elif group_name == "BASE":
            sx = x_min
        elif group_name == "TIP":
            sx = x_max
        else:
            continue
        _draw_support(ax, sx, float(np.interp(sx, x_display, y_center)), support["type"], support_scale)

    if not support_defs:
        _draw_support(ax, x_min, float(y_center[0]), "fixed", support_scale)

    # Close the beam tip with an explicit vertical edge.
    ax.plot([x_max, x_max], [float(bot[-1]), float(top[-1])], color="#2E4053", lw=1.5, zorder=4)

    ax.set_xlim(x_min - 0.03 * beam_length, x_max + 0.03 * beam_length)
    y_min = float(np.min(bot)) - 0.9 * beam_h
    y_max = float(np.max(top)) + 1.0 * beam_h
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([])
    ax.set_xlabel(f"Beam Axis X [{DEFAULT_UNIT_POLICY.visualization_length_unit}]", fontsize=10)
    ax.set_title("MFZ Contour on Beam", fontsize=12, fontweight="bold", pad=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.24, shrink=0.48, aspect=30)
    cbar.set_label(f"MFZ [{DEFAULT_UNIT_POLICY.visualization_moment_unit}]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    out_path = resolve_workspace_file("mfz_contour.png", "viz", run_id or None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)

    return (
        f"MFZ contour saved: {out_path}\n"
        f"MFZ range: {float(np.min(mfz_display)):.3f} to {float(np.max(mfz_display)):.3f} {DEFAULT_UNIT_POLICY.visualization_moment_unit}\n"
        f"Beam length: {beam_length:.3f} {DEFAULT_UNIT_POLICY.visualization_length_unit} | Nodes: {len(rows)}"
    )


@tool("visualize_freebody", return_direct=False)
def visualize_freebody_tool(
    reac_name: str = "tab_reac.txt",
    support_type: str = "fixed",
    geo_name: str = "model.geo",
    comm_name: str = "model.comm",
    unit: str = "m",
    run_id: str = "",
) -> str:
    """Draw free-body diagram with support symbols, external loads, and reactions."""
    geo_path = resolve_workspace_file(geo_name, "inputs", run_id or None)
    beam_length_raw, group_pos_solver = _read_geo_model(geo_path)
    if beam_length_raw is None:
        beam_length_raw = 1000.0

    beam_length = convert_length(
        beam_length_raw,
        DEFAULT_UNIT_POLICY.solver_coordinate_unit,
        DEFAULT_UNIT_POLICY.visualization_length_unit,
    )

    group_pos_visual = {
        name: convert_length(
            x_val,
            DEFAULT_UNIT_POLICY.solver_coordinate_unit,
            DEFAULT_UNIT_POLICY.visualization_length_unit,
        )
        for name, x_val in group_pos_solver.items()
    }

    reac_path = resolve_workspace_file(reac_name, "outputs", run_id or None)
    reactions = _read_reactions_all(reac_path)
    if not reactions:
        return f"Error: could not parse reactions from '{reac_path}'."
    reac = reactions[0]

    comm_path = resolve_workspace_file(comm_name, "inputs", run_id or None)
    loads = _read_loads(comm_path)
    support_defs = _read_support_groups(comm_path)

    if support_type == "fixed" and support_defs:
        unique_groups = {s["group"] for s in support_defs}
        if len(unique_groups) >= 2:
            support_type = "simply_supported"
        elif any(s["type"] == "fixed" for s in support_defs):
            support_type = "fixed"
        elif any(s["type"] == "pinned" for s in support_defs):
            support_type = "pinned"
        elif any(s["type"] == "roller" for s in support_defs):
            support_type = "roller"

    scale = max(beam_length * 0.07, 0.18)

    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.set_facecolor("white")

    ax.plot([0.0, beam_length], [0.0, 0.0], color="#1B2631", lw=3.2, solid_capstyle="round", zorder=3)

    if support_type == "simply_supported":
        _draw_support(ax, x0=0.0, y0=0.0, support_type="pinned", scale=scale)
        _draw_support(ax, x0=beam_length, y0=0.0, support_type="roller", scale=scale)
    else:
        _draw_support(ax, x0=0.0, y0=0.0, support_type=support_type, scale=scale)

    reaction_summaries: list[str] = []
    for reac_item in reactions:
        fy = float(reac_item["Fy"])
        fx = float(reac_item["Fx"])
        mz = float(reac_item["Mz"])
        rx = convert_length(
            float(reac_item["X"]),
            DEFAULT_UNIT_POLICY.solver_coordinate_unit,
            DEFAULT_UNIT_POLICY.visualization_length_unit,
        )
        rx_plot = rx if rx > 1e-9 else (-0.55 * scale)

        if abs(fy) > 1e-9:
            fy_display = convert_force(
                fy,
                DEFAULT_UNIT_POLICY.solver_force_unit,
                DEFAULT_UNIT_POLICY.visualization_force_unit,
            )
            _draw_vertical_arrow(
                ax,
                x=rx_plot,
                y=0.0,
                magnitude=fy,
                label=f"Fy = {fy_display:.2f} {DEFAULT_UNIT_POLICY.visualization_force_unit}",
                scale=scale,
                color="#2874A6",
            )

        reaction_summaries.append(
            f"{reac_item['node']}: Fx={convert_force(fx, DEFAULT_UNIT_POLICY.solver_force_unit, DEFAULT_UNIT_POLICY.visualization_force_unit):.3f} {DEFAULT_UNIT_POLICY.visualization_force_unit}, "
            f"Fy={convert_force(fy, DEFAULT_UNIT_POLICY.solver_force_unit, DEFAULT_UNIT_POLICY.visualization_force_unit):.3f} {DEFAULT_UNIT_POLICY.visualization_force_unit}, "
            f"Mz={convert_moment(mz, DEFAULT_UNIT_POLICY.solver_moment_unit, DEFAULT_UNIT_POLICY.visualization_moment_unit):.3f} {DEFAULT_UNIT_POLICY.visualization_moment_unit}"
        )

    if support_type == "fixed":
        mz = float(reac["Mz"])
        if abs(mz) > 1e-9:
            theta = np.linspace(np.deg2rad(70), np.deg2rad(305), 120)
            radius = 0.92 * scale
            cx, cy = -1.12 * scale, 0.0
            ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), color="#2874A6", lw=2.2)
            ax.arrow(
                cx + radius * np.cos(theta[1]),
                cy + radius * np.sin(theta[1]),
                (cx + radius * np.cos(theta[0])) - (cx + radius * np.cos(theta[1])),
                (cy + radius * np.sin(theta[0])) - (cy + radius * np.sin(theta[1])),
                head_width=0.30 * scale,
                head_length=0.34 * scale,
                fc="#2874A6",
                ec="#2874A6",
                lw=0,
                length_includes_head=True,
            )

            m_display = convert_moment(
                mz,
                DEFAULT_UNIT_POLICY.solver_moment_unit,
                DEFAULT_UNIT_POLICY.visualization_moment_unit,
            )
            ax.text(
                cx - 1.0 * scale,
                cy + 0.12 * scale,
                f"Mz = {m_display:.2f} {DEFAULT_UNIT_POLICY.visualization_moment_unit}",
                color="#2874A6",
                fontsize=9,
                va="center",
                ha="right",
                fontweight="bold",
            )

    if loads:
        _draw_loads(ax, beam_length, loads, scale=scale, group_pos_visual=group_pos_visual)

    dim_y = -1.55 * scale
    ax.annotate(
        "",
        xy=(0.0, dim_y),
        xytext=(beam_length, dim_y),
        arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": "#616A6B"},
    )
    ax.text(
        0.5 * beam_length,
        dim_y - 0.20 * scale,
        f"L = {beam_length:.3f} {DEFAULT_UNIT_POLICY.visualization_length_unit}",
        fontsize=9,
        color="#616A6B",
        ha="center",
        va="top",
    )

    x_margin = 2.1 * scale
    ax.set_xlim(-x_margin, beam_length + x_margin)
    ax.set_ylim(-2.0 * scale, 2.2 * scale)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.tight_layout()
    out_path = resolve_workspace_file("freebody.png", "viz", run_id or None)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return (
        "Free-body diagram generated.\n"
        f"Image: {out_path}\n"
        f"Beam length: {beam_length:.3f} {DEFAULT_UNIT_POLICY.visualization_length_unit}\n"
        f"Support: {support_type}\n"
        f"Reactions: {' | '.join(reaction_summaries)}\n"
        f"Loads parsed: {len(loads)} from {comm_name}"
    )
