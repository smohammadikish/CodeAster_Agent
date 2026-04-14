"""Solver tests for both JSON inputs and parsed Code_Aster inputs."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from beam_analysis import analyze_beam_file, load_beam_source


class BeamAnalysisTests(unittest.TestCase):
    def test_code_aster_input_folder_is_parsed(self) -> None:
        model = load_beam_source(REPO_ROOT / "input")

        self.assertTrue(math.isclose(model.length, 4.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertEqual(len(model.supports), 1)
        self.assertEqual(model.supports[0].type, "fixed")
        self.assertTrue(
            math.isclose(model.supports[0].x, 0.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertEqual(len(model.point_forces), 1)
        self.assertTrue(
            math.isclose(model.point_forces[0].x, 4.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(model.point_forces[0].fy, -1500.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertIn("E", model.material)
        self.assertIn("Iz", model.section)

    def test_cantilever_tip_load(self) -> None:
        result = analyze_beam_file(APP_ROOT / "examples" / "cantilever_tip_load.json")
        support = result["supports"][0]["reactions"]

        self.assertTrue(math.isclose(support["Ry"], 10.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(math.isclose(support["Mz"], 40.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertEqual(result["case_type"], "cantilever_tip_load")
        self.assertEqual(result["load"]["x"], 4.0)

        shear = result["extrema"]["shear_force"]
        moment = result["extrema"]["bending_moment"]
        displacement = result["extrema"]["displacement_y"]
        rotation = result["extrema"]["rotation_z"]
        response_points = {
            point["label"]: point for point in result["response_points"]
        }
        tip = response_points["TIP"]

        self.assertTrue(
            math.isclose(abs(shear["value"]), 10.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(abs(moment["value"]), 40.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(
                abs(displacement["value"]),
                abs(tip["uy"]),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(math.isclose(displacement["x"], 4.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(math.isclose(rotation["x"], 4.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(
            math.isclose(
                tip["uy"],
                -(10.0 * 4.0**3) / (3.0 * 210000000000.0 * 4.1e-6),
                rel_tol=1e-5,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                tip["rz"],
                -(10.0 * 4.0**2) / (2.0 * 210000000000.0 * 4.1e-6),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )

    def test_simple_beam_midpoint_load(self) -> None:
        result = analyze_beam_file(
            APP_ROOT / "examples" / "simple_beam_midpoint_load.json"
        )
        support_a = result["supports"][0]["reactions"]
        support_b = result["supports"][1]["reactions"]

        self.assertTrue(math.isclose(support_a["Ry"], 6.0, rel_tol=1e-9, abs_tol=1e-9))
        self.assertTrue(math.isclose(support_b["Ry"], 6.0, rel_tol=1e-9, abs_tol=1e-9))

        shear = result["extrema"]["shear_force"]
        moment = result["extrema"]["bending_moment"]
        axial = result["extrema"]["axial_force"]
        displacement = result["extrema"]["displacement_y"]
        response_points = {
            point["label"]: point for point in result["response_points"]
        }
        midspan = response_points["MIDSPAN"]

        self.assertEqual(result["case_type"], "simply_supported_midspan_point_load")
        self.assertTrue(
            math.isclose(abs(shear["value"]), 6.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(moment["value"], 18.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(abs(axial["value"]), 0.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(
                abs(displacement["value"]),
                abs(midspan["uy"]),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(displacement["x"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(
                midspan["uy"],
                -(12.0 * 6.0**3) / (48.0 * 210000000000.0 * 6.0e-6),
                rel_tol=1e-5,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(midspan["x"], 3.0)


if __name__ == "__main__":
    unittest.main()
