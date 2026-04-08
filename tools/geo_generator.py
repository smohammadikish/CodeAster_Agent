# ----------------------------
#  tools/geo_generator.py
# ----------------------------

from langchain_core.tools import tool

from storage import resolve_workspace_file


GEO_TEMPLATE = """// ---- 1D Beam geometry (OpenCASCADE) ----
SetFactory("OpenCASCADE");


// Mesh settings to preserve points as nodes in MED
Mesh.MedFileMinorVersion = 0;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;


// Parameters
// length = {length_mm} (mm)
Point(1) = {{0, 0, 0}};
Point(2) = {{{length_mm}, 0, 0}};
Line(1) = {{1, 2}};


// Transfinite line definition
// n_nodes = n_elems + 1 = {n_nodes}
Transfinite Line {{1}} = {n_nodes};


// Physical groups
Physical Point("BASE") = {{1}};
Physical Point("TIP")  = {{2}};
Physical Curve("BEAM") = {{1}};
"""


@tool("generate_geo", return_direct=False)
def generate_geo_tool(
    length_mm: float = 300,
    n_elems: int | None = None,
    geo_name: str = "model.geo",
    run_id: str = "",
) -> str:
    """
    Generate a .geo file for a 1D beam.
    Input units: millimeters (mm)
    """

    if length_mm <= 0:
        return "âŒ Error: Beam length must be > 0."

    if n_elems is None:
        n_elems = max(2, int(length_mm / 200))

    if n_elems < 1:
        return "âŒ Error: n_elems must be >= 1."

    n_nodes = n_elems + 1
    geo_path = resolve_workspace_file(geo_name, "inputs", run_id or None)

    content = GEO_TEMPLATE.format(length_mm=length_mm, n_nodes=n_nodes)
    with open(geo_path, "w", encoding="utf-8") as f:
        f.write(content)

    return (
        f"âœ… Generated {geo_name}\n"
        f"Length = {length_mm} mm, Elements = {n_elems}, Nodes = {n_nodes}\n"
        "Groups defined: BASE, TIP, BEAM\n"
        f"â†’ Saved to {geo_path.resolve()}"
    )
