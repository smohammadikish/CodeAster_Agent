# --------------------------------------------------------
#  tools/export_generator.py
# --------------------------------------------------------

import os

from langchain_core.tools import tool

from storage import resolve_workspace_file


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
    run_id: str = "",
) -> str:
    """
    Generates a Code_Aster .export file used by as_run.
    Ensures correct file paths for COMM, MED, and result output files.
    """

    comm_path = resolve_workspace_file(comm_name, "inputs", run_id or None).resolve()
    med_path = resolve_workspace_file(med_name, "inputs", run_id or None).resolve()
    export_path = resolve_workspace_file(export_name, "inputs", run_id or None).resolve()

    mess_path = resolve_workspace_file(f"{job_name}.mess", "outputs", run_id or None).resolve()
    resu_path = resolve_workspace_file(f"{job_name}.resu", "outputs", run_id or None).resolve()
    base_path = resolve_workspace_file(f"{job_name}_base", "outputs", run_id or None).resolve()
    tab_reac = resolve_workspace_file("tab_reac.txt", "outputs", run_id or None).resolve()
    tab_disp = resolve_workspace_file("tab_disp.txt", "outputs", run_id or None).resolve()
    local_axes = resolve_workspace_file("local_axes.txt", "outputs", run_id or None).resolve()
    tab_sie = resolve_workspace_file("tab_sie.txt", "outputs", run_id or None).resolve()

    if not os.path.exists(comm_path):
        return f"âŒ Error: {comm_path} not found."

    if not os.path.exists(med_path):
        return f"âŒ Error: {med_path} not found."

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

    with open(export_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"âœ… Export file generated: {export_path}"
