// ---- 1D Beam geometry (OpenCASCADE) ----
SetFactory("OpenCASCADE");


// Mesh settings to preserve points as nodes in MED
Mesh.MedFileMinorVersion = 0;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;


// Parameters
// length = 6130.0 (mm)
Point(1) = {0, 0, 0};
Point(2) = {6130.0, 0, 0};
Line(1) = {1, 2};


// Transfinite line definition
// n_nodes = n_elems + 1 = 13
Transfinite Line {1} = 13;


// Physical groups
Physical Point("BASE") = {1};
Physical Point("TIP")  = {2};
Physical Curve("BEAM") = {1};
