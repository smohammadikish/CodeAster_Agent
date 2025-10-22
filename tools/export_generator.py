# tools/export_generator.py
import os

from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

EXPORT_TEMPLATE = """P actions make_etude
P debug nodebug
P memjob 4194304
P memory_limit 4096.0
P mode interactif
P mpi_nbcpu 1
P mpi_nbnoeud 1
P ncpus 1
P nomjob {job_name}
P origine beam_agent_project
P server localhost
P time_limit 1800.0
P version stable
A memjeveux 256.0
A tpmax 1800.0

F comm {comm_path} D 1
F mmed {med_path} D 20
F mess {log_path} R 6
F resu {resu_path} R 80

R base {base_path} R 0
"""


@tool("generate_export", return_direct=False)
def generate_export_tool(
    job_name: str = "beam_job",
    comm_name: str = "model.comm",
    med_name: str = "model.med",
    export_name: str = "model.export",
    out_dir: str = WORKSPACE,
) -> str:
    """
    Generate a Code_Aster .export file mapping all input/output files.
    """
    comm_path = os.path.join(WORKSPACE, comm_name)
    med_path = os.path.join(WORKSPACE, med_name)
    export_path = os.path.join(WORKSPACE, export_name)
    log_path = os.path.join(WORKSPACE, f"{job_name}.mess")
    resu_path = os.path.join(WORKSPACE, f"{job_name}.resu")
    base_path = os.path.join(WORKSPACE, f"{job_name}_base")

    # verify inputs
    if not os.path.exists(comm_path):
        return f"❌ Error: {comm_path} not found."
    if not os.path.exists(med_path):
        return f"❌ Error: {med_path} not found."

    # fill template
    content = EXPORT_TEMPLATE.format(
        job_name=job_name,
        comm_path=comm_path,
        med_path=med_path,
        log_path=log_path,
        resu_path=resu_path,
        base_path=base_path,
    )

    with open(export_path, "w") as f:
        f.write(content)

    return (
        f"✅ Generated {export_name} in '{WORKSPACE}'\n"
        f"Job: {job_name}\n"
        f"Inputs: {comm_name}, {med_name}\n"
        f"Outputs: {job_name}.resu, {job_name}.mess\n"
        f"→ Ready for Code_Aster run with 'as_run {export_name}'"
    )
