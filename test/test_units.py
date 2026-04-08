import shutil
import unittest
import uuid

from storage import ensure_run_dirs, resolve_workspace_file
from tools.visualize_tool import (visualize_freebody_tool,
                                  visualize_mfz_contour_tool,
                                  visualize_resu_tool, visualize_stress_tool)
from units import (DEFAULT_UNIT_POLICY, convert_force, convert_length,
                   convert_moment, normalize_user_params,
                   normalize_user_params_mixed_units)


class UnitPolicyTests(unittest.TestCase):
    def test_base_conversions(self) -> None:
        self.assertAlmostEqual(convert_length(3000.0, "mm", "m"), 3.0)
        self.assertAlmostEqual(convert_length(0.12, "m", "mm"), 120.0)
        self.assertAlmostEqual(convert_force(1500.0, "N", "kN"), 1.5)
        self.assertAlmostEqual(convert_moment(3_000_000.0, "Nmm", "kNm"), 3.0)

    def test_normalize_user_params_to_solver_units(self) -> None:
        params = {
            "length_m": 300.0,
            "HY": 12.0,
            "HZ": 10.0,
            "FY": -1.5,
            "n_elems": 15,
            "E": 2.1e11,
            "NU": 0.3,
            "RHO": 7850.0,
        }
        normalized = normalize_user_params(
            params,
            input_length_unit="cm",
            input_force_unit="kN",
        )
        self.assertEqual(normalized["length_m"], 3000.0)
        self.assertEqual(normalized["HY"], 120.0)
        self.assertEqual(normalized["HZ"], 100.0)
        self.assertEqual(normalized["FY"], -1500.0)

    def test_normalize_user_params_with_mixed_units(self) -> None:
        params = {
            "length_m": 5.0,
            "HY": 20.0,
            "HZ": 12.0,
            "FY": -1.2,
            "n_elems": 10,
            "E": 3.0e10,
            "NU": 0.2,
            "RHO": 2500.0,
        }
        normalized = normalize_user_params_mixed_units(
            params,
            length_unit="m",
            hy_unit="cm",
            hz_unit="cm",
            force_unit="kN",
        )
        self.assertEqual(normalized["length_m"], 5000.0)
        self.assertEqual(normalized["HY"], 200.0)
        self.assertEqual(normalized["HZ"], 120.0)
        self.assertEqual(normalized["FY"], -1200.0)


class VisualizationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"unitviz_{uuid.uuid4().hex[:8]}"
        self.paths = ensure_run_dirs(self.run_id)

        resolve_workspace_file("beam_job.resu", "outputs", self.run_id).write_text(
            "\n".join(
                [
                    " NOEUD                 DX                    DY                    DZ                    DRX                   DRY                   DRZ",
                    " N1        0.0 0.0 0.0 0.0 0.0 0.0",
                    " N2        0.0 -5.0E-06 0.0 0.0 0.0 -2.7E-09",
                ]
            ),
            encoding="utf-8",
        )
        resolve_workspace_file("tab_sie.txt", "outputs", self.run_id).write_text(
            "\n".join(
                [
                    "INTITULE;NOEUD;RESU;NOM_CHAM;NUME_ORDRE;INST;ABSC_CURV;COOR_X;COOR_Y;COOR_Z;N;VY;VZ;MT;MFY;MFZ",
                    "BeamForces;N1;res_stat;SIEF_ELNO;1;0.0;0.0;0.0;0.0;0.0;0.0;-1000.0;0.0;0.0;0.0;-3000000.0",
                    "BeamForces;N2;res_stat;SIEF_ELNO;1;0.0;3000.0;3000.0;0.0;0.0;0.0;-1000.0;0.0;0.0;0.0;0.0",
                ]
            ),
            encoding="utf-8",
        )
        resolve_workspace_file("tab_reac.txt", "outputs", self.run_id).write_text(
            "\n".join(
                [
                    "RESULTAT NOM_CHAM INST NUME_ORDRE NOEUD COOR_X COOR_Y DX DY DZ DRX DRY DRZ",
                    "res_stat REAC_NODA 0.0 1 N1 0.0 0.0 0.0 1000.0 0.0 0.0 0.0 3000000.0",
                ]
            ),
            encoding="utf-8",
        )
        resolve_workspace_file("model.geo", "inputs", self.run_id).write_text(
            "\n".join(
                [
                    'Point(1) = {0, 0, 0};',
                    'Point(2) = {3000, 0, 0};',
                ]
            ),
            encoding="utf-8",
        )
        resolve_workspace_file("model.comm", "inputs", self.run_id).write_text(
            "FORCE_NODALE=_F(GROUP_NO=('TIP',), FY=-1500.0);",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        if self.paths.root.exists():
            shutil.rmtree(self.paths.root)

    def test_visualizations_use_centralized_units(self) -> None:
        resu_result = visualize_resu_tool.invoke(
            {"resu_name": "beam_job.resu", "run_id": self.run_id}
        )
        stress_result = visualize_stress_tool.invoke(
            {"stress_name": "tab_sie.txt", "run_id": self.run_id}
        )
        contour_result = visualize_mfz_contour_tool.invoke(
            {"stress_name": "tab_sie.txt", "run_id": self.run_id}
        )
        freebody_result = visualize_freebody_tool.invoke(
            {
                "reac_name": "tab_reac.txt",
                "geo_name": "model.geo",
                "comm_name": "model.comm",
                "run_id": self.run_id,
            }
        )

        self.assertIn(DEFAULT_UNIT_POLICY.visualization_displacement_unit, resu_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_force_unit, stress_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_moment_unit, stress_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_moment_unit, contour_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_length_unit, contour_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_force_unit, freebody_result)
        self.assertIn(DEFAULT_UNIT_POLICY.visualization_moment_unit, freebody_result)
        self.assertTrue((self.paths.viz / "beam_job_def.png").exists())
        self.assertTrue((self.paths.viz / "beam_job_rot.png").exists())
        self.assertTrue((self.paths.viz / "shear_diagram.png").exists())
        self.assertTrue((self.paths.viz / "moment_diagram.png").exists())
        self.assertTrue((self.paths.viz / "mfz_contour.png").exists())
        self.assertTrue((self.paths.viz / "freebody.png").exists())


if __name__ == "__main__":
    unittest.main()
