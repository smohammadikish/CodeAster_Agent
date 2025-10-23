// Simple 1D beam geometry (OpenCASCADE)
SetFactory("OpenCASCADE");

// parameters
// length = 4.0
Point(1) = {0, 0, 0};
Point(2) = {4.0, 0, 0};

Line(1) = {1, 2};

// enforce exact number of elements via Transfinite
// n_nodes = n_elems + 1 = 21
Transfinite Line {1} = 21;

// physical groups for Code_Aster boundary conditions & properties
Physical Point("BASE") = {1};
Physical Point("TIP")  = {2};
Physical Curve("BEAM") = {1};
