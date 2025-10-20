# tools/geo_generator.py
from langchain.tools import tool

GEO_TEMPLATE = """// Simple 1D beam geometry (OpenCASCADE)
SetFactory("OpenCASCADE");

// parameters
// length = {length}
Point(1) = {{0, 0, 0}};
Point(2) = {{{length}, 0, 0}};

Line(1) = {{1, 2}};

// enforce exact number of elements via Transfinite
// n_nodes = n_elems + 1 = {n_nodes}
Transfinite Line {{1}} = {n_nodes};

// physical groups for Code_Aster boundary conditions & properties
Physical Point("BASE") = {{1}};
Physical Point("TIP")  = {{2}};
Physical Curve("BEAM") = {{1}};
"""


@tool("generate_geo", return_direct=False)
def generate_geo_tool(
    length_m: float = 3.0, n_elems: int = 10, geo_path: str = "model.geo"
) -> str:
    """
    Generate a minimal 1D beam .geo file for Gmsh with BASE/TIP/BEAM groups.
    Mesh density is controlled by the number of elements (n_elems).
    - length_m > 0
    - n_elems >= 1
    """
    # --- validation ---
    if length_m <= 0:
        return "Error: length_m must be > 0."
    if n_elems < 1:
        return "Error: n_elems must be >= 1."

    n_nodes = n_elems + 1
    content = GEO_TEMPLATE.format(length=length_m, n_nodes=n_nodes)

    with open(geo_path, "w") as f:
        f.write(content)

    return (
        f"Generated {geo_path} "
        f"(L={length_m} m, n_elems={n_elems}, n_nodes={n_nodes}).\n"
        "Physicals: BASE (node 1), TIP (node 2), BEAM (line 1)."
    )
