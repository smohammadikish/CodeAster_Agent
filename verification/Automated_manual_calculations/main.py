"""Command-line entry point for beam analysis runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from beam_analysis import analyze_beam_source
from report_renderer import render_report

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "input"
DEFAULT_WORKSPACE = REPO_ROOT / "workspace"


def _supports_builtin_report(result: dict) -> bool:
    return result.get("case_type") in {
        "cantilever_tip_load",
        "simply_supported_midspan_point_load",
    }


def create_run_workspace(
    input_path_like: str | Path,
    rendered_result: str,
    root: str | Path = DEFAULT_WORKSPACE,
) -> Path:
    """Create one folder per run and save the source files plus the result JSON."""
    input_path = Path(input_path_like).resolve()
    workspace_root = Path(root)
    run_label = input_path.stem if input_path.is_file() else input_path.name
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{run_label}"
    run_workspace = workspace_root / run_name
    run_workspace.mkdir(parents=True, exist_ok=False)

    if input_path.is_dir():
        for file_path in sorted(path for path in input_path.iterdir() if path.is_file()):
            shutil.copy2(file_path, run_workspace / file_path.name)
    else:
        shutil.copy2(input_path, run_workspace / input_path.name)
    (run_workspace / "result.json").write_text(rendered_result + "\n", encoding="utf-8")
    return run_workspace


def main() -> None:
    """Read the selected source, run the solver, and store the outputs."""
    parser = argparse.ArgumentParser(
        description="Solve statically determinate 2D beam reactions and internal actions from JSON."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=(
            "Path to a beam input JSON file or to a folder containing "
            "model.comm, model.export, and beam.geo. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path for the analysis result JSON file.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE),
        help=f"Folder where a timestamped workspace is created for each run. Default: {DEFAULT_WORKSPACE}",
    )
    parser.add_argument(
        "--report-output",
        help=(
            "Optional path for a rendered report file. When omitted, a report is "
            "saved inside the run workspace when a built-in template is available."
        ),
    )
    parser.add_argument(
        "--report-template",
        help="Optional custom report template. If omitted, the built-in template is selected from the beam type.",
    )
    args = parser.parse_args()

    result = analyze_beam_source(args.input_file)
    rendered = json.dumps(result, indent=2)
    run_workspace = create_run_workspace(args.input_file, rendered, args.workspace_root)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    report_output = args.report_output
    if report_output is None and _supports_builtin_report(result):
        report_stem = {
            "cantilever_tip_load": "cantilever",
            "simply_supported_midspan_point_load": "simply_supported",
        }[result["case_type"]]
        report_output = run_workspace / f"{report_stem}_report.txt"

    if report_output:
        render_report(
            result,
            report_output,
            template_path_like=args.report_template,
            root=APP_ROOT,
        )

    print(rendered)
    print(f"Saved run workspace: {run_workspace}", file=sys.stderr)


if __name__ == "__main__":
    main()
