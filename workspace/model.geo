// ---- 1D Beam geometry (OpenCASCADE) ----
SetFactory("OpenCASCADE");

// Mesh settings to preserve points as nodes in MED
Mesh.MedFileMinorVersion = 0;
Mesh.SaveWithoutOrphans = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Parameters
// length = 3.0 (m)
Point(1) = {0, 0, 0};
Point(2) = {3.0, 0, 0};
Line(1) = {1, 2};

// Transfinite line definition for controlled meshing
// n_nodes = n_elems + 1 = 16
Transfinite Line {1} = 16;

// Physical groups for Code_Aster boundary conditions
Physical Point("BASE") = {1};
Physical Point("TIP")  = {2};
Physical Curve("BEAM") = {1};
