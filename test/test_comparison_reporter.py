from verification.comparison_reporter import generate_comparison_report


def _solver_result(uy: float, rz: float) -> dict:
    return {
        "supports": [],
        "extrema": {
            "axial_force": {"value": 0.0, "x": 0.0},
            "shear_force": {"value": 0.0, "x": 0.0},
            "bending_moment": {"value": 0.0, "x": 0.0},
            "displacement_y": {"value": uy, "x": 5.0},
            "rotation_z": {"value": rz, "x": 5.0},
        },
    }


def test_report_uses_explicit_code_aster_displacement_unit_conversion(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "tab_reac.txt").write_text("", encoding="utf-8")
    (outputs_dir / "tab_sie.txt").write_text("", encoding="utf-8")
    (outputs_dir / "tab_disp.txt").write_text(
        "0 0 DEPL 0 0 5000 0 0 -3.934400e-04 0 0 0 1.045650e-07\n",
        encoding="utf-8",
    )

    manual_result = _solver_result(0.3485511, 0.1045653)
    opensees_result = _solver_result(0.3485510, 0.1045650)

    report = generate_comparison_report(
        manual_result,
        opensees_result,
        code_aster_outputs_dir=outputs_dir,
    )

    assert "uy_abs       3.934400e-07" in report
    assert "rz_abs       1.045650e-07" in report
    assert "Displacement unit normalization:" not in report


def test_report_reads_all_code_aster_support_reactions(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "tab_reac.txt").write_text(
        "\n".join(
            [
                "res_stat REAC_NODA 0 1 N1 0.00000E+00 0.00000E+00 0.00000E+00 6.00000E+02 0.00000E+00 0.00000E+00 0.00000E+00 9.31323E-10",
                "res_stat REAC_NODA 0 1 N3 5.00000E+03 0.00000E+00 0.00000E+00 6.00000E+02 0.00000E+00 0.00000E+00 0.00000E+00 9.31323E-10",
            ]
        ),
        encoding="utf-8",
    )
    (outputs_dir / "tab_sie.txt").write_text("", encoding="utf-8")
    (outputs_dir / "tab_disp.txt").write_text("", encoding="utf-8")

    manual_result = {
        "supports": [
            {"name": "BASE", "x": 0.0, "reactions": {"Rx": 0.0, "Ry": 600.0, "Mz": 0.0}},
            {"name": "TIP", "x": 5.0, "reactions": {"Rx": 0.0, "Ry": 600.0, "Mz": 0.0}},
        ],
        "extrema": {
            "axial_force": {"value": 0.0, "x": 0.0},
            "shear_force": {"value": 0.0, "x": 0.0},
            "bending_moment": {"value": 0.0, "x": 0.0},
            "displacement_y": {"value": 0.0, "x": 0.0},
            "rotation_z": {"value": 0.0, "x": 0.0},
        },
    }
    opensees_result = {
        "supports": [
            {"name": "A", "x": 0.0, "reactions": {"Rx": 0.0, "Ry": 600.0, "Mz": 3.58e-12}},
            {"name": "B", "x": 5.0, "reactions": {"Rx": 0.0, "Ry": 600.0, "Mz": 1.19e-12}},
        ],
        "extrema": manual_result["extrema"],
    }

    report = generate_comparison_report(
        manual_result,
        opensees_result,
        code_aster_outputs_dir=outputs_dir,
    )

    assert "TIP / B         Ry       6.000000e+02   6.000000e+02   6.000000e+02" in report
    assert "TIP / B         Mz       0.000000e+00" in report
