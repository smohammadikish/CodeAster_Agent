"""Unified CLI for integrated beam analysis verification."""

import sys

from orchestrator import run_both_solvers


def main() -> None:
    """Run the integrated analysis workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run both automated_manual and opensees beam solvers together.",
        epilog=(
            "Integrates two verification methods for 2D beam analysis. "
            "Results are stored in a unified workspace structure with comparison reports."
        ),
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="input",
        help=(
            "Path to beam input JSON file or folder with Code_Aster files "
            "(model.comm, model.export, beam.geo). Default: input"
        ),
    )
    parser.add_argument(
        "--workspace-root",
        default="workspace",
        help="Root folder where timestamped run directories are created. Default: workspace",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all previous analysis runs in the workspace.",
    )

    args = parser.parse_args()

    if args.list_runs:
        from workspace_manager import list_runs

        runs = list_runs(args.workspace_root)
        if runs:
            print(f"\nAnalysis runs in '{args.workspace_root}':\n")
            for run_path in runs:
                print(f"  {run_path.name}")
        else:
            print(f"\nNo analysis runs found in '{args.workspace_root}'")
        return

    try:
        run_dir, manual_result, opensees_result, comparison = run_both_solvers(
            args.input_file, args.workspace_root
        )

        if manual_result and opensees_result:
            status = "both solvers completed"
            if comparison:
                status += ", comparison generated"
        elif manual_result:
            status = "automated_manual completed, opensees failed"
        elif opensees_result:
            status = "opensees completed, automated_manual failed"
        else:
            print("[FAILED] Both solvers failed")
            sys.exit(1)

        print(f"\nRun directory: {run_dir}")
        print(f"[OK] {status}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FAILED] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
