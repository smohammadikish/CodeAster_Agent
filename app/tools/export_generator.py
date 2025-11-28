# --------------------------------------------------------
#  tools/export_generator.py
#  Responsible for generating Code_Aster .export file
# --------------------------------------------------------

import os

from langchain_core.tools import tool  # ✅ Updated import for LangChain 1.x

# Workspace directory where files will be stored
WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)


# Template of Code_Aster export file
EXPORT_TEMPLATE = """# --- Code_Aster Export File ---
P actions make_etude
P debug nodebug
P memjob 4194304
P memory_limit 4096.0
P mode interactif
P mpi_nbcpu 1
P mpi_nbnoeud 1
P ncpus 1
P nomjob {job_name}
P origine CodeAster_Agent
P server localhost
P time_limit 1800.0
P tpsjob 32
P version stable

A memjeveux 256.0
A tpmax 1800.0

# --- File Associations ---
F comm {comm_path} D 1
F libr {med_path} D 20

# Optional result tables and logs (created by visualize tools)
F mess {mess_path} R 6
F resu {resu_path} R 80
F libr {tab_reac} R 18
F libr {tab_disp} R 19
F libr {local_axes} R 53
F libr {tab_sie} R 21

# Base directory for temporary Code_Aster data
R base {base_path} R 0
"""


@tool("generate_export", return_direct=False)
def generate_export_tool(
    job_name: str = "beam_job",
    comm_name: str = "model.comm",
    med_name: str = "model.med",
    export_name: str = "model.export",
) -> str:
    """
    Generates a Code_Aster .export file used by as_run.
    Ensures correct file paths for COMM, MED, and result output files.

    Args:
        job_name (str): Name of the Code_Aster job
        comm_name (str): Name of .comm file generated earlier
        med_name (str): Name of .med mesh file from Gmsh
        export_name (str): Output export filename

    Returns:
        str: Success or error message
    """

    # Build absolute paths
    comm_path = os.path.abspath(os.path.join(WORKSPACE, comm_name))
    med_path = os.path.abspath(os.path.join(WORKSPACE, med_name))
    export_path = os.path.abspath(os.path.join(WORKSPACE, export_name))

    # Output files for Code_Aster logs and results
    mess_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}.mess"))
    resu_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}.resu"))
    base_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}_base"))

    # Intermediate tables used by visualization tools
    tab_reac = os.path.abspath(os.path.join(WORKSPACE, "tab_reac.txt"))
    tab_disp = os.path.abspath(os.path.join(WORKSPACE, "tab_disp.txt"))
    local_axes = os.path.abspath(os.path.join(WORKSPACE, "local_axes.txt"))
    tab_sie = os.path.abspath(os.path.join(WORKSPACE, "tab_sie.txt"))

    # Verify existence of required input files
    if not os.path.exists(comm_path):
        return f"❌ Error: {comm_path} not found."

    if not os.path.exists(med_path):
        return f"❌ Error: {med_path} not found."

    # Fill template with correct paths
    content = EXPORT_TEMPLATE.format(
        job_name=job_name,
        comm_path=comm_path,
        med_path=med_path,
        mess_path=mess_path,
        resu_path=resu_path,
        tab_reac=tab_reac,
        tab_disp=tab_disp,
        local_axes=local_axes,
        tab_sie=tab_sie,
        base_path=base_path,
    )

    # Write .export file
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"✅ Export file generated: {export_path}"
