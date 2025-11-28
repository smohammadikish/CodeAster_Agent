import os
import subprocess
import sys

# Hard-coded paths used inside the container
WORKSPACE_DIR = "/workspace"
ASTER_BIN = "/opt/spack/var/spack/environments/simvia_env/.spack-env/view/bin/run_aster"


def run_gmsh(geo_file: str, med_file: str) -> None:
    """Run Gmsh in CLI mode to generate a MED mesh (v4.0.0 compatible)."""
    # Same options you were using inside the LangChain tool
    gmsh_opts = [
        "-setnumber",
        "Mesh.MedFileMinorVersion",
        "0",  # MED 4.0.0
        "-setnumber",
        "Mesh.SaveWithoutOrphans",
        "1",
        "-v",
        "2",
    ]

    cmd = [
        "gmsh",
        geo_file,
        "-1",
        "-format",
        "med",
        "-o",
        med_file,
    ] + gmsh_opts

    print("Running Gmsh:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_aster(export_file: str) -> None:
    """Run Code_Aster using the export file."""
    cmd = [ASTER_BIN, export_file]
    print("Running Code_Aster:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run_solver.py <geo_name> <export_name>")
        sys.exit(1)

    geo = sys.argv[1]  # e.g. "model.geo"
    export = sys.argv[2]  # e.g. "model.export"

    geo_path = os.path.join(WORKSPACE_DIR, geo)
    med_path = os.path.join(WORKSPACE_DIR, geo.replace(".geo", ".med"))
    export_path = os.path.join(WORKSPACE_DIR, export)

    print(f"Geo: {geo_path}")
    print(f"MED will be: {med_path}")
    print(f"Export: {export_path}")

    # 1) Run Gmsh -> creates .med
    run_gmsh(geo_path, med_path)

    # 2) Run Code_Aster -> consumes .med and .comm via .export
    run_aster(export_path)

    print("All done.")
