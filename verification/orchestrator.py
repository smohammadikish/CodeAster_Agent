"""Unified orchestrator for running both analysis applications."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Automated_manual_calculations"))
sys.path.insert(0, str(ROOT / "opensees" / "src"))

from beam_analysis import analyze_beam_source
from comparison_reporter import generate_comparison_report
from report_renderer import render_report
from shared_input import get_input_source_info, resolve_input_source_path
from ver_opensees.config import load_beam_input
from ver_opensees.config import \
    resolve_input_source_path as resolve_opensees_path
from ver_opensees.runner import run_analysis
from workspace_manager import (create_embedded_verification_workspace,
                               create_run_workspace, save_comparison_report,
                               save_report, save_solver_result)


def run_automated_manual(input_path: Path) -> tuple[dict[str, Any], str | None]:
    """Run the automated manual solver and return canonical results."""
    try:
        result = analyze_beam_source(input_path)

        report_text = None
        if result.get("case_type") in {
            "cantilever_tip_load",
            "simply_supported_midspan_point_load",
        }:
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".txt", delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)

                report_path = render_report(
                    result,
                    tmp_path,
                    root=ROOT / "Automated_manual_calculations",
                )
                report_text = report_path.read_text(encoding="utf-8")
                tmp_path.unlink()
            except Exception as exc:
                print(
                    f"Warning: Could not generate report for automated_manual: {exc}"
                )

        return result, report_text
    except Exception as exc:
        raise RuntimeError(f"Automated manual solver failed: {exc}") from exc


def run_opensees_solver(input_path: Path) -> tuple[dict[str, Any], str | None]:
    """Run the OpenSees solver and return canonical results."""
    try:
        resolved_path = resolve_opensees_path(input_path)
        beam_input, input_source = load_beam_input(resolved_path)

        import shutil
        import uuid

        executable = shutil.which(beam_input.opensees_executable)
        if executable is None:
            raise FileNotFoundError(
                f"OpenSees executable '{beam_input.opensees_executable}' not found in PATH. "
                f"Set OPENSEES_EXECUTABLE environment variable or specify in input JSON."
            )

        temp_root = ROOT / "workspace_temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        tmpdir = temp_root / f"opensees_{uuid.uuid4().hex}"
        try:
            tmpdir.mkdir(parents=True, exist_ok=False)
            temp_workspace = tmpdir / "workspace_temp"
            beam_input.output_root = temp_workspace
            run_dir = run_analysis(beam_input, input_source)

            results_json_path = run_dir / "results" / "results.json"
            if not results_json_path.exists():
                raise FileNotFoundError(f"OpenSees results not found: {results_json_path}")

            raw_result = json.loads(results_json_path.read_text(encoding="utf-8"))
            result = raw_result.get("derived_summary", raw_result)

            report_path = run_dir / "results" / "report.txt"
            report_text = (
                report_path.read_text(encoding="utf-8")
                if report_path.exists()
                else None
            )

            return result, report_text
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as exc:
        raise RuntimeError(f"OpenSees solver failed: {exc}") from exc


def run_both_solvers(
    input_path: Path | str,
    workspace_root: Path | str = "workspace",
    verification_root: Path | str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    """Run both solvers on the same input and organize results."""
    resolved_input = resolve_input_source_path(input_path)
    input_source = get_input_source_info(resolved_input)

    input_label = resolved_input.stem if resolved_input.is_file() else resolved_input.name
    if verification_root is not None:
        run_dir = create_embedded_verification_workspace(
            input_source,
            verification_root,
            input_label,
        )
        code_aster_outputs_dir = Path(verification_root).parent / "outputs"
    else:
        run_dir = create_run_workspace(input_source, input_label, workspace_root)
        code_aster_outputs_dir = None

    print(f"\nRun: {input_label}")

    print("- automated_manual...")
    try:
        manual_result, manual_report = run_automated_manual(resolved_input)
        save_solver_result(run_dir, "manual", manual_result)
        if manual_report:
            save_report(run_dir, "manual", manual_report)
        print("  [OK]")
    except Exception as exc:
        print(f"  [FAILED] {exc}")
        manual_result = {}

    print("- opensees...")
    try:
        opensees_result, opensees_report = run_opensees_solver(resolved_input)
        save_solver_result(run_dir, "opensees", opensees_result, "results.json")
        if opensees_report:
            save_report(run_dir, "opensees", opensees_report)
        print("  [OK]")
    except Exception as exc:
        print(f"  [FAILED] {exc}")
        opensees_result = {}

    print("- comparison...")
    if manual_result and opensees_result:
        try:
            comparison = generate_comparison_report(
                manual_result,
                opensees_result,
                input_label,
                code_aster_outputs_dir=code_aster_outputs_dir,
            )
            save_comparison_report(run_dir, comparison)
            print("  [OK]")
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            comparison = ""
    else:
        if not manual_result:
            print("  [SKIPPED] automated_manual unavailable")
        if not opensees_result:
            print("  [SKIPPED] opensees unavailable")
        comparison = ""
    return run_dir, manual_result, opensees_result, comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run both automated_manual and opensees solvers for beam analysis."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="input",
        help="Path to a beam input JSON file or a folder containing Code_Aster input files. Default: input",
    )
    parser.add_argument(
        "--workspace-root",
        default="workspace",
        help="Root folder for run workspaces. Default: workspace",
    )

    args = parser.parse_args()

    try:
        run_dir, manual_result, opensees_result, comparison = run_both_solvers(
            args.input_file, args.workspace_root
        )
        print(f"Results saved to: {run_dir}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
