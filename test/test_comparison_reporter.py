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


def test_report_normalizes_code_aster_displacement_units(tmp_path):
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

    assert "uy_abs       3.934400e-01" in report
    assert "rz_abs       1.045650e-01" in report
    assert "scaled Code_Aster value by 1000" in report
    assert "scaled Code_Aster value by 1e+06" in report
