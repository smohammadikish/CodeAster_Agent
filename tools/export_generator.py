# tools/export_generator.py
import os

from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

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

# Optional result tables and logs
F mess {mess_path} R 6
F resu {resu_path} R 80
F libr {tab_reac} R 18
F libr {tab_disp} R 19
F libr {local_axes} R 53
F libr {tab_sie} R 21

# --- Base directory for temporary data ---
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
    Generate a Code_Aster .export file with absolute paths (cross-platform).
    Compatible with Code_Aster 14 (as_run / as_run.bat).
    """

    # --- Build absolute paths ---
    comm_path = os.path.abspath(os.path.join(WORKSPACE, comm_name))
    med_path = os.path.abspath(os.path.join(WORKSPACE, med_name))
    export_path = os.path.abspath(os.path.join(WORKSPACE, export_name))
    mess_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}.mess"))
    resu_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}.resu"))
    base_path = os.path.abspath(os.path.join(WORKSPACE, f"{job_name}_base"))
    tab_reac = os.path.abspath(os.path.join(WORKSPACE, "tab_reac.txt"))
    tab_disp = os.path.abspath(os.path.join(WORKSPACE, "tab_disp.txt"))
    local_axes = os.path.abspath(os.path.join(WORKSPACE, "local_axes.txt"))
    tab_sie = os.path.abspath(os.path.join(WORKSPACE, "tab_sie.txt"))

    # --- Verify required files exist ---
    if not os.path.exists(comm_path):
        return f"❌ Error: {comm_path} not found."
    if not os.path.exists(med_path):
        return f"❌ Error: {med_path} not found."

    # --- Fill template ---
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

    # --- Write export file ---
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(content)

    return (
        f"✅ Generated {export_name}\n"
        f"📄 Job name: {job_name}\n"
        f"🗂 Export file: {export_path}\n"
        f"Includes:\n"
        f"  • COMM file: {comm_path}\n"
        f"  • MED mesh: {med_path}\n"
        f"  • MESS log: {mess_path}\n"
        f"  • RESU results: {resu_path}\n"
        f"  • TAB_REAC: {tab_reac}\n"
        f"  • TAB_DISP: {tab_disp}\n"
        f"  • LOCAL_AXES: {local_axes}\n"
        f"  • tab_sie: {tab_sie}\n"
        f"  • BASE dir: {base_path}\n"
        "→ Ready to run via `as_run model.export` or `as_run.bat model.export`"
    )
