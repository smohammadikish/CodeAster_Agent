"""Generate comparison reports between analysis methods.

This module compares the canonical result format produced by the integrated
manual solver and the integrated OpenSees solver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_ZERO_TOL = 1e-9
MM_TO_M = 1.0 / 1000.0


def _build_text_table(
    headers: list[str], rows: list[list[str]], align_right: set[int] | None = None
) -> list[str]:
    """Build a compact markdown-style table without padded cells."""
    _ = align_right  # keep backward compatibility with existing call sites
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines

def load_result_json(filepath: Path) -> dict[str, Any]:
    """Load a result JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Result file not found: {filepath}")
    return json.loads(filepath.read_text(encoding="utf-8"))


def load_code_aster_output(outputs_dir: Path) -> dict[str, Any]:
    """Load key result metrics from raw Code_Aster output tables."""
    reactions = _load_code_aster_reactions(outputs_dir / "tab_reac.txt")
    displacements = _load_code_aster_displacements(outputs_dir / "tab_disp.txt")
    internal_forces = _load_code_aster_internal_forces(outputs_dir / "tab_sie.txt")
    return {
        "reactions": reactions,
        "forces": internal_forces,
        "displacements": displacements,
    }


def _load_code_aster_reactions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    reactions: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "REAC_NODA" not in raw_line:
            continue
        tokens = raw_line.split()
        if len(tokens) < 13:
            continue
        reactions.append(
            {
                "name": str(tokens[4]),
                "x": float(tokens[5]) * MM_TO_M,
                "values": (
                    float(tokens[7]),
                    float(tokens[8]),
                    float(tokens[12]) * MM_TO_M,
                ),
            }
        )

    if not reactions:
        return []

    reactions.sort(key=lambda item: item["x"])
    for index, reaction in enumerate(reactions):
        if len(reactions) == 1:
            reaction["name"] = "BASE"
        elif index == 0:
            reaction["name"] = "BASE"
        elif index == len(reactions) - 1:
            reaction["name"] = "TIP"
        else:
            reaction["name"] = f"SUPPORT_{index + 1}"
    return reactions


def _load_code_aster_displacements(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {
            "uy_abs": {"value": 0.0, "x": 0.0},
            "rz_abs": {"value": 0.0, "x": 0.0},
        }

    uy_best = {"value": 0.0, "x": 0.0}
    rz_best = {"value": 0.0, "x": 0.0}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if " DEPL " not in raw_line:
            continue
        tokens = raw_line.split()
        if len(tokens) < 13:
            continue
        x_value = float(tokens[5]) * MM_TO_M
        # Code_Aster model input uses mm and N; convert displacement to SI for comparison.
        uy_value = abs(float(tokens[8])) * MM_TO_M
        rz_value = abs(float(tokens[12]))
        if uy_value >= uy_best["value"]:
            uy_best = {"value": uy_value, "x": x_value}
        if rz_value >= rz_best["value"]:
            rz_best = {"value": rz_value, "x": x_value}

    return {"uy_abs": uy_best, "rz_abs": rz_best}


def _load_code_aster_internal_forces(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {
            "N_abs": {"value": 0.0, "x": 0.0},
            "V_abs": {"value": 0.0, "x": 0.0},
            "M_abs": {"value": 0.0, "x": 0.0},
        }

    results = {
        "N_abs": {"value": 0.0, "x": 0.0},
        "V_abs": {"value": 0.0, "x": 0.0},
        "M_abs": {"value": 0.0, "x": 0.0},
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "BeamForces" not in raw_line:
            continue
        tokens = [token.strip() for token in raw_line.split(";")]
        if len(tokens) < 16:
            continue
        x_value = abs(float(tokens[7])) * MM_TO_M
        n_value = abs(float(tokens[10]))
        v_value = abs(float(tokens[11]))
        m_value = abs(float(tokens[15])) * MM_TO_M
        if n_value >= results["N_abs"]["value"]:
            results["N_abs"] = {"value": n_value, "x": x_value}
        if v_value >= results["V_abs"]["value"]:
            results["V_abs"] = {"value": v_value, "x": x_value}
        if m_value >= results["M_abs"]["value"]:
            results["M_abs"] = {"value": m_value, "x": x_value}
    return results


def _canonical_result(result: dict[str, Any]) -> dict[str, Any]:
    """Unwrap solver-specific payloads into the canonical comparison format."""
    if "derived_summary" in result and isinstance(result["derived_summary"], dict):
        return result["derived_summary"]
    return result


def extract_reactions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract support reactions from canonical solver output."""
    reactions: list[dict[str, Any]] = []

    for index, support in enumerate(_canonical_result(result).get("supports", [])):
        support_reactions = support.get("reactions", {})
        reactions.append(
            {
                "name": str(support.get("name", "Unknown")),
                "x": float(support.get("x", index)),
                "values": (
                    float(support_reactions.get("Rx", 0.0)),
                    float(support_reactions.get("Ry", 0.0)),
                    float(support_reactions.get("Mz", 0.0)),
                ),
            }
        )

    return reactions


def _support_key(support: dict[str, Any]) -> tuple[str, float | str]:
    if "x" in support:
        return ("x", round(float(support["x"]), 9))
    return ("name", str(support["name"]))


def _merge_support_label(
    manual_support: dict[str, Any] | None, opensees_support: dict[str, Any] | None
) -> str:
    manual_name = manual_support["name"] if manual_support else None
    opensees_name = opensees_support["name"] if opensees_support else None
    if manual_name and opensees_name:
        if manual_name == opensees_name:
            return str(manual_name)
        return f"{manual_name} / {opensees_name}"
    return str(manual_name or opensees_name or "Unknown")


def _extract_extrema_metric(
    result: dict[str, Any], field_name: str
) -> dict[str, float]:
    extrema = _canonical_result(result).get("extrema", {})
    field = extrema.get(field_name, {})
    return {
        "value": abs(float(field.get("value", 0.0))),
        "x": float(field.get("x", 0.0)),
    }


def extract_internal_forces(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract envelope-style internal force values from canonical output."""
    return {
        "N_abs": _extract_extrema_metric(result, "axial_force"),
        "V_abs": _extract_extrema_metric(result, "shear_force"),
        "M_abs": _extract_extrema_metric(result, "bending_moment"),
    }


def extract_displacements(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract displacement metrics from canonical output."""
    return {
        "uy_abs": _extract_extrema_metric(result, "displacement_y"),
        "rz_abs": _extract_extrema_metric(result, "rotation_z"),
    }


def compute_difference(value1: float, value2: float) -> tuple[float, float]:
    """Compute absolute and relative difference between two values."""
    if abs(value1) <= REPORT_ZERO_TOL:
        value1 = 0.0
    if abs(value2) <= REPORT_ZERO_TOL:
        value2 = 0.0
    abs_diff = abs(value1 - value2)

    if abs(value2) > 1e-12:
        rel_percent = (abs_diff / abs(value2)) * 100.0
    elif abs(value1) > 1e-12:
        rel_percent = (abs_diff / abs(value1)) * 100.0
    else:
        rel_percent = 0.0

    return abs_diff, rel_percent


def _relative_percent(reference: float, candidate: float) -> float:
    _, rel_percent = compute_difference(candidate, reference)
    return rel_percent


def _agreement_bucket(rel_pct: float) -> str:
    if rel_pct < 1.0:
        return "excellent"
    if rel_pct < 5.0:
        return "acceptable"
    return "poor"


def generate_comparison_report(
    manual_result: dict[str, Any],
    opensees_result: dict[str, Any],
    run_label: str = "",
    code_aster_outputs_dir: Path | None = None,
) -> str:
    """Generate a comparison report between canonical solver outputs."""
    report: list[str] = []
    report.append("=" * 80)
    report.append("COMPARISON REPORT: Code_Aster vs OpenSees vs Manual Analysis")
    report.append("=" * 80)

    if run_label:
        report.append(f"\nAnalysis Run: {run_label}\n")

    manual_reactions = extract_reactions(manual_result)
    opensees_reactions = extract_reactions(opensees_result)
    manual_forces = extract_internal_forces(manual_result)
    opensees_forces = extract_internal_forces(opensees_result)
    manual_displacements = extract_displacements(manual_result)
    opensees_displacements = extract_displacements(opensees_result)
    code_aster_data = (
        load_code_aster_output(code_aster_outputs_dir)
        if code_aster_outputs_dir is not None
        else {
            "reactions": [],
            "forces": {
                "N_abs": {"value": 0.0, "x": 0.0},
                "V_abs": {"value": 0.0, "x": 0.0},
                "M_abs": {"value": 0.0, "x": 0.0},
            },
            "displacements": {
                "uy_abs": {"value": 0.0, "x": 0.0},
                "rz_abs": {"value": 0.0, "x": 0.0},
            },
        }
    )
    code_aster_reactions = code_aster_data["reactions"]
    code_aster_forces = code_aster_data["forces"]
    code_aster_displacements = code_aster_data["displacements"]

    total_comparisons = 0
    excellent = 0
    acceptable = 0
    poor = 0

    report.append("\n" + "=" * 80)
    report.append("1. SUPPORT REACTIONS COMPARISON")
    report.append("=" * 80)

    code_aster_supports = {
        _support_key(support): support for support in code_aster_reactions
    }
    manual_supports = {_support_key(support): support for support in manual_reactions}
    opensees_supports = {
        _support_key(support): support for support in opensees_reactions
    }
    all_supports = (
        set(code_aster_supports) | set(manual_supports) | set(opensees_supports)
    )
    if all_supports:
        report.append(
            f"\n{'Support':<15} {'Comp':<6} {'Code_Aster':<15} {'OpenSees':<15} "
            f"{'Manual':<15} {'M-O %':<10} {'Avg vs CA %':<12}"
        )
        report.append("-" * 96)

        for support in sorted(all_supports):
            code_aster_support = code_aster_supports.get(support)
            manual_support = manual_supports.get(support)
            opensees_support = opensees_supports.get(support)
            support_label = _merge_support_label(
                manual_support or code_aster_support,
                opensees_support,
            )
            code_aster_values = (
                code_aster_support["values"] if code_aster_support else (0.0, 0.0, 0.0)
            )
            manual_values = (
                manual_support["values"] if manual_support else (0.0, 0.0, 0.0)
            )
            opensees_values = (
                opensees_support["values"] if opensees_support else (0.0, 0.0, 0.0)
            )

            for component, c_val, o_val, m_val in (
                ("Rx", code_aster_values[0], opensees_values[0], manual_values[0]),
                ("Ry", code_aster_values[1], opensees_values[1], manual_values[1]),
                ("Mz", code_aster_values[2], opensees_values[2], manual_values[2]),
            ):
                mo_pct = _relative_percent(o_val, m_val)
                avg_val = (m_val + o_val) / 2.0
                avg_vs_ca_pct = _relative_percent(c_val, avg_val)
                report.append(
                    f"{support_label:<15} {component:<6} {c_val:>14.6e} "
                    f"{o_val:>14.6e} {m_val:>14.6e} "
                    f"{mo_pct:>8.2f}% {avg_vs_ca_pct:>10.2f}%"
                )
                total_comparisons += 1
                bucket = _agreement_bucket(max(mo_pct, avg_vs_ca_pct))
                if bucket == "excellent":
                    excellent += 1
                elif bucket == "acceptable":
                    acceptable += 1
                else:
                    poor += 1
    else:
        report.append("\nNo support reactions found in either analysis.")

    report.append("\n" + "=" * 80)
    report.append("2. INTERNAL FORCE ENVELOPE COMPARISON")
    report.append("=" * 80)

    report.append(
        f"\n{'Force Type':<12} {'CA':<13} {'x_ca':<8} {'OpenSees':<13} {'x_o':<8} "
        f"{'Manual':<13} {'x_m':<8} {'M-O %':<10} {'Avg vs CA %':<12}"
    )
    report.append("-" * 120)

    for force_type in ("N_abs", "V_abs", "M_abs"):
        code_aster_force = code_aster_forces.get(force_type, {"value": 0.0, "x": 0.0})
        manual_force = manual_forces.get(force_type, {"value": 0.0, "x": 0.0})
        opensees_force = opensees_forces.get(force_type, {"value": 0.0, "x": 0.0})
        c_val = code_aster_force["value"]
        m_val = manual_force["value"]
        o_val = opensees_force["value"]
        mo_pct = _relative_percent(o_val, m_val)
        avg_val = (m_val + o_val) / 2.0
        avg_vs_ca_pct = _relative_percent(c_val, avg_val)
        report.append(
            f"{force_type:<12} {c_val:>12.6e} {code_aster_force['x']:>7.4f} "
            f"{o_val:>14.6e} {opensees_force['x']:>9.4f} "
            f"{m_val:>14.6e} {manual_force['x']:>9.4f} "
            f"{mo_pct:>8.2f}% {avg_vs_ca_pct:>10.2f}%"
        )
        total_comparisons += 1
        bucket = _agreement_bucket(max(mo_pct, avg_vs_ca_pct))
        if bucket == "excellent":
            excellent += 1
        elif bucket == "acceptable":
            acceptable += 1
        else:
            poor += 1

    report.append("\n" + "=" * 80)
    report.append("3. DISPLACEMENT COMPARISON")
    report.append("=" * 80)

    report.append(
        f"\n{'Metric':<12} {'CA':<13} {'x_ca':<8} {'OpenSees':<13} {'x_o':<8} "
        f"{'Manual':<13} {'x_m':<8} {'M-O %':<10} {'Avg vs CA %':<12}"
    )
    report.append("-" * 120)

    for metric in ("uy_abs", "rz_abs"):
        code_aster_disp = code_aster_displacements.get(metric, {"value": 0.0, "x": 0.0})
        manual_disp = manual_displacements.get(metric, {"value": 0.0, "x": 0.0})
        opensees_disp = opensees_displacements.get(metric, {"value": 0.0, "x": 0.0})
        m_val = manual_disp["value"]
        o_val = opensees_disp["value"]
        mo_pct = _relative_percent(o_val, m_val)
        avg_val = (m_val + o_val) / 2.0
        c_val = code_aster_disp["value"]
        avg_vs_ca_pct = _relative_percent(c_val, avg_val)
        report.append(
            f"{metric:<12} {c_val:>12.6e} {code_aster_disp['x']:>7.4f} "
            f"{o_val:>14.6e} {opensees_disp['x']:>9.4f} "
            f"{m_val:>14.6e} {manual_disp['x']:>9.4f} "
            f"{mo_pct:>8.2f}% {avg_vs_ca_pct:>10.2f}%"
        )
        total_comparisons += 1
        bucket = _agreement_bucket(max(mo_pct, avg_vs_ca_pct))
        if bucket == "excellent":
            excellent += 1
        elif bucket == "acceptable":
            acceptable += 1
        else:
            poor += 1

    report.append("\n" + "=" * 80)
    report.append("4. METHOD AND FORMULATION NOTES")
    report.append("=" * 80)
    report.append(
        "Code_Aster uses POU_D_T beam elements (Timoshenko-type behavior with shear deformation)."
    )
    report.append(
        "OpenSees uses elasticBeamColumn in the current workflow (Euler-Bernoulli formulation)."
    )
    report.append(
        "Manual solver uses closed-form statically determinate beam equations (Euler-Bernoulli assumptions)."
    )
    report.append(
        "Small support moments near machine precision (around 1e-12) are numerical noise and can be treated as zero."
    )

    report.append("\n" + "=" * 80)
    report.append("5. ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("\nComparison Statistics:")

    if total_comparisons > 0:
        report.append(f"  Total comparisons: {total_comparisons}")
        report.append(
            f"  Excellent agreement (< 1%): {excellent} "
            f"({100.0 * excellent / total_comparisons:.1f}%)"
        )
        report.append(
            f"  Acceptable agreement (1-5%): {acceptable} "
            f"({100.0 * acceptable / total_comparisons:.1f}%)"
        )
        report.append(
            f"  Poor agreement (> 5%): {poor} "
            f"({100.0 * poor / total_comparisons:.1f}%)"
        )
        report.append(
            "  Summary buckets use the worst percentage in each row "
            "(max of M-O % and Avg vs CA %)."
        )
    else:
        report.append("  No quantitative data available for statistical summary.")

    report.append("\n" + "=" * 80)
    report.append("END OF COMPARISON REPORT")
    report.append("=" * 80)
    return "\n".join(report)

