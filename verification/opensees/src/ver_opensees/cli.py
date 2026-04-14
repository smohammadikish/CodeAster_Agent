from __future__ import annotations

import argparse
from pathlib import Path

from ver_opensees.config import load_beam_input, resolve_input_source_path
from ver_opensees.runner import run_analysis

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "input"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an OpenSees beam analysis from JSON or Code_Aster input."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=(
            "Path to a JSON file, a Code_Aster file, or a directory containing "
            f"the Code_Aster input set. Default: {DEFAULT_INPUT}"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = resolve_input_source_path(Path(args.input_file))
    beam_input, input_source = load_beam_input(input_path)

    run_dir = run_analysis(beam_input, input_source)
    print(f"Analysis complete. Run workspace written to: {run_dir}")


if __name__ == "__main__":
    main()
