# ----------------------------
#  tools/geo_generator.py
# ----------------------------

from langchain_core.tools import tool

from storage import resolve_workspace_file

GEO_TEMPLATE_CANTILEVER = """// ---- 1D Beam geometry (OpenCASCADE) ----
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


GEO_TEMPLATE_SIMPLY_SUPPORTED = """// ---- 1D Beam geometry (OpenCASCADE) ----
SetFactory("OpenCASCADE");


// Mesh settings to preserve points as nodes in MED
Mesh.MedFileMinorVersion = 0;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;


// Parameters
// length = {length_mm} (mm)
Point(1) = {{0, 0, 0}};
Point(2) = {{{midpoint_mm}, 0, 0}};
Point(3) = {{{length_mm}, 0, 0}};
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};


// Transfinite line definition
// n_nodes = n_elems + 1 = {n_nodes}
Transfinite Line {{1}} = {half_nodes};
Transfinite Line {{2}} = {half_nodes};


// Physical groups
Physical Point("BASE") = {{1}};
Physical Point("MID")  = {{2}};
Physical Point("TIP")  = {{3}};
Physical Curve("BEAM") = {{1, 2}};
"""


@tool("generate_geo", return_direct=False)
def generate_geo_tool(
    scenario: str = "cantilever_tip_load",
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
        return "Error: Beam length must be > 0."

    if n_elems is None:
        n_elems = max(2, int(length_mm / 200))

    if n_elems < 1:
        return "Error: n_elems must be >= 1."

    n_nodes = n_elems + 1
    geo_path = resolve_workspace_file(geo_name, "inputs", run_id or None)

    if scenario == "simply_supported_midpoint_load":
        if n_elems % 2 != 0:
            return "Error: simply_supported_midpoint_load requires an even n_elems."
        half_nodes = (n_elems // 2) + 1
        content = GEO_TEMPLATE_SIMPLY_SUPPORTED.format(
            length_mm=length_mm,
            midpoint_mm=length_mm / 2.0,
            n_nodes=n_nodes,
            half_nodes=half_nodes,
        )
        groups_line = "Groups defined: BASE, MID, TIP, BEAM"
    else:
        content = GEO_TEMPLATE_CANTILEVER.format(
            length_mm=length_mm,
            n_nodes=n_nodes,
        )
        groups_line = "Groups defined: BASE, TIP, BEAM"

    with open(geo_path, "w", encoding="utf-8") as f:
        f.write(content)

    return (
        f"Generated {geo_name}\n"
        f"Length = {length_mm} mm, Elements = {n_elems}, Nodes = {n_nodes}\n"
        f"{groups_line}\n"
        f"Saved to {geo_path.resolve()}"
    )
