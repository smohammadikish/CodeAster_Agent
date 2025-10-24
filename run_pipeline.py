# run_pipeline.py 
# JUST FOR testing the pipeline _ there is no LLM
import os
import time

from tools.geo_generator import generate_geo_tool
from tools.mesher_gmsh import mesh_with_gmsh_tool
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.run_aster_tool import run_code_aster_tool
from tools.visualize_tool import visualize_resu_tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

def step(title):
    print("\n" + "=" * 60)
    print(f"🔹 {title}")
    print("=" * 60)

def main():
    start = time.time()

    # 1. Generate geometry (.geo)
    step("Step 1 — Generating beam geometry (.geo)")
    print(generate_geo_tool.invoke({
        "length_m": 3.0,
        "n_elems": 15,
        "geo_name": "model.geo"
    }))

    # 2. Mesh with Gmsh and convert to MED 4.0
    step("Step 2 — Meshing geometry with Gmsh (.msh → .med)")
    print(mesh_with_gmsh_tool.invoke({
        "geo_name": "model.geo",
        "med_name": "model.med"
    }))

    # 3. Generate .comm file for Code_Aster
    step("Step 3 — Creating Code_Aster command file (.comm)")
    print(generate_comm_tool.invoke({
        "med_name": "model.med",
        "comm_name": "model.comm",
        "E": 11e9,
        "NU": 0.3,
        "RHO": 480.0,
        "HY": 0.1,
        "HZ": 0.1,
        "FY": -1000.0
    }))

    # 4. Generate .export file (absolute paths)
    step("Step 4 — Creating .export file")
    print(generate_export_tool.invoke({
        "job_name": "beam_job",
        "comm_name": "model.comm",
        "med_name": "model.med",
        "export_name": "model.export"
    }))

    # 5. Run Code_Aster analysis
    step("Step 5 — Running Code_Aster analysis")
    print(run_code_aster_tool.invoke({
        "export_name": "model.export"
    }))

    # 6. Visualize results
    step("Step 6 — Visualizing results (.resu → .png)")
    print(visualize_resu_tool.invoke({
        "resu_name": "beam_job.resu",
        "n_points": 100
    }))

    end = time.time()
    print("\n✅ Pipeline completed successfully in %.2f seconds." % (end - start))
    print(f"📂 Check the '{WORKSPACE}' folder for all generated files.")

if __name__ == "__main__":
    main()