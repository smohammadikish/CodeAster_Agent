# test_mesher.py
from tools.mesher_gmsh import mesh_with_gmsh_tool

print(mesh_with_gmsh_tool.invoke({"geo_path": "model.geo", "med_path": "model.med"}))
