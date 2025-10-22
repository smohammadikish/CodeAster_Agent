# tools/geo_generator.py
import os

from langchain.tools import tool

WORKSPACE = "workspace"
try:
    os.makedirs(WORKSPACE, exist_ok=True)
except FileExistsError:
    pass

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
    length_m: float = 3.0,
    n_elems: int | None = None,
    geo_name: str = "model.geo",
) -> str:
    """
    Generate a minimal 1D beam .geo file for Gmsh with BASE/TIP/BEAM groups.
    Mesh density is controlled by the number of elements (n_elems).
    - length_m > 0
    - n_elems >= 1 (if not given, estimated from length)
    """
    # --- validation ---
    if length_m <= 0:
        return "Error: length_m must be > 0."

    # estimate number of elements if not provided
    if n_elems is None:
        n_elems = max(2, int(length_m / 0.2))  # roughly 1 element per 0.2 m

    if n_elems < 1:
        return "Error: n_elems must be >= 1."

    n_nodes = n_elems + 1
    geo_path = os.path.join(WORKSPACE, geo_name)
    content = GEO_TEMPLATE.format(length=length_m, n_nodes=n_nodes)

    with open(geo_path, "w") as f:
        f.write(content)

    return (
        f"✅ Generated {geo_name}\n"
        f"Length = {length_m} m, Elements = {n_elems}, Nodes = {n_nodes}\n"
        "Groups: BASE (1), TIP (2), BEAM (1)"
    )
