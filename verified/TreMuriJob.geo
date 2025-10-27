//3Muri project created on 27/10/2025 12:04:34
SetFactory("OpenCASCADE");
Mesh.MedFileMinorVersion = 0;
Mesh.SaveWithoutOrphans = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Point(1) = {-1018.328289046, -325.747665948, 0, 100000};
Point(2) = {-1018.328289046, -325.747665948, 300, 100000};
Point(3) = {-518.328289046, -325.747665948, 0, 100000};
Point(4) = {-518.328289046, -325.747665948, 300, 100000};
Line(1) = {1, 2};
Line(2) = {3, 4};
Line(3) = {2, 4};
Physical Point("ALL_P") = {1,2,3,4};
Physical Curve("ALL_L") = {1,2,3};
Physical Surface("ALL_S") = {};
Physical Curve("P_3") = {1};
Physical Curve("P_4") = {2};
Physical Curve("T_1") = {3};
Physical Point("NS_1_VVVVVV") = {1,3};
Field[1] = Constant;
Field[1].IncludeBoundary = 0;
Field[1].CurvesList = {1,2,3};
Field[1].VIn =  50;
Field[1].VOut =  100000;

Field[3] = Min;
Field[3].FieldsList = {1,2};Background Field = 3;
