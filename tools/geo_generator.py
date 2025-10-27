# tools/geo_generator.py
import os
from langchain.tools import tool

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

GEO_TEMPLATE = """// ---- 1D Beam geometry (OpenCASCADE) ----
SetFactory("OpenCASCADE");

// Mesh settings to preserve points as nodes in MED
Mesh.MedFileMinorVersion = 0;
Mesh.SaveWithoutOrphans = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Parameters
// length = {length} (m)
Point(1) = {{0, 0, 0}};
Point(2) = {{{length}, 0, 0}};
Line(1) = {{1, 2}};

// Transfinite line definition for controlled meshing
// n_nodes = n_elems + 1 = {n_nodes}
Transfinite Line {{1}} = {n_nodes};

// Physical groups for Code_Aster boundary conditions
Physical Point("BASE") = {{1}};
Physical Point("TIP")  = {{2}};
Physical Curve("BEAM") = {{1}};
"""

@tool("generate_geo", return_direct=False)
def generate_geo_tool(
    length_m: float = 3.0,
    n_elems: int | None = None,
    geo_name: str = "model.geo",
) -> str:
    """
    Generate the .geo file for a 1D cantilever beam.
    Compatible with Code_Aster 14 and Gmsh → MED v4.0.
    """

    # --- Input validation ---
    if length_m <= 0:
        return "❌ Error: Beam length must be > 0."

    # Estimate number of elements if not provided
    if n_elems is None:
        n_elems = max(2, int(length_m / 0.2))

    if n_elems < 1:
        return "❌ Error: n_elems must be >= 1."

    n_nodes = n_elems + 1
    geo_path = os.path.join(WORKSPACE, geo_name)
    content = GEO_TEMPLATE.format(length=length_m, n_nodes=n_nodes)

    with open(geo_path, "w", encoding="utf-8") as f:
        f.write(content)

    return (
        f"✅ Generated {geo_name}\n"
        f"Length = {length_m} m, Elements = {n_elems}, Nodes = {n_nodes}\n"
        "Groups defined: BASE, TIP, BEAM\n"
        f"→ Saved to {geo_path}"
    )