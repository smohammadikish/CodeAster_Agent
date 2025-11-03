# ----------------------------
#  test/test_mesher.py
# ----------------------------

from tools.mesher_gmsh import mesh_with_gmsh_tool

if __name__ == "__main__":
    result = mesh_with_gmsh_tool.invoke(
        {"geo_name": "model.geo", "med_name": "model.med"}
    )
    print(result)
