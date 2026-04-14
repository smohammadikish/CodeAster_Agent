// This file defines the simple 1D beam geometry used by Code_Aster.
SetFactory("OpenCASCADE");

// Keep the end points as mesh nodes in the MED file.
Mesh.MedFileMinorVersion = 0;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Beam geometry.
// Length = 4.0 m
Point(1) = {0, 0, 0};
Point(2) = {4.0, 0, 0};
Line(1) = {1, 2};

// Use a controlled number of beam nodes.
// Number of nodes = number of elements + 1 = 11
Transfinite Line {1} = 11;

// Name the physical groups used by Code_Aster.
Physical Point("BASE") = {1};
Physical Point("TIP")  = {2};
Physical Curve("BEAM") = {1};
