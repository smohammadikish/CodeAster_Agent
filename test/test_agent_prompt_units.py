import unittest
from unittest.mock import Mock, patch

from chat.resolution import resolve_analysis_inputs
from input_parser import ParameterSheet, build_state_aware_input
from main_agent import (build_agent, build_system_prompt,
                        extract_scenario_from_text, is_exit_input,
                        run_agent_one_shot, run_full_pipeline)


class AgentPromptUnitTests(unittest.TestCase):
    def test_system_prompt_direct_run_and_units(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("cantilever_tip_load", prompt)
        self.assertIn("simply_supported_midpoint_load", prompt)
        self.assertIn("Never request confirmation before running.", prompt)
        self.assertIn("length_unit", prompt)
        self.assertIn("force_unit", prompt)

    def test_build_agent_exposes_catalog_and_pipeline_tools(self) -> None:
        agent = build_agent()
        tool_names = {tool.name for tool in agent.tools}
        self.assertIn("list_material_catalog", tool_names)
        self.assertIn("get_material_catalog", tool_names)
        self.assertIn("list_section_catalog", tool_names)
        self.assertIn("get_section_catalog", tool_names)
        self.assertIn("build_parametric_section_catalog", tool_names)
        self.assertIn("run_full_pipeline", tool_names)

    @patch("main_agent.run_hybrid_pipeline_with_run_storage")
    def test_run_full_pipeline_accepts_per_parameter_units(self, mocked_run) -> None:
        mocked_run.return_value = "Simulation finished successfully."
        result = run_full_pipeline.invoke(
            {
                "scenario": "cantilever_tip_load",
                "length_m": 5.0,
                "n_elems": 10,
                "E": 3.0e10,
                "NU": 0.2,
                "RHO": 2500.0,
                "HY": 20.0,
                "HZ": 12.0,
                "FY": 1.2,
                "length_unit": "m",
                "hy_unit": "cm",
                "hz_unit": "cm",
                "force_unit": "kN",
            }
        )
        self.assertIn("Simulation finished successfully.", result)
        self.assertNotIn("Missing required parameters", result)

    @patch("main_agent.run_hybrid_pipeline_with_run_storage")
    def test_cantilever_with_catalog_section_does_not_require_hy_hz(self, mocked_run) -> None:
        mocked_run.return_value = "Simulation finished successfully."
        result = run_full_pipeline.invoke(
            {
                "scenario": "cantilever_tip_load",
                "length_m": 5.0,
                "n_elems": 12,
                "E": 210e9,
                "NU": 0.3,
                "RHO": 7850.0,
                "HY": None,
                "HZ": None,
                "A_mm2": 2010.0,
                "IY_mm4": 8693000.0,
                "IZ_mm4": 683100.0,
                "JX_mm4": 50000.0,
                "AY_mm2": 1000.0,
                "AZ_mm2": 1000.0,
                "FY": 1.2,
                "length_unit": "m",
                "force_unit": "kN",
            }
        )
        self.assertIn("Simulation finished successfully.", result)
        self.assertNotIn("Invalid value for HY", result)
        self.assertNotIn("Invalid value for HZ", result)

    def test_one_shot_flow_passes_parser_hint_to_agent(self) -> None:
        agent_executor = Mock()
        agent_executor.invoke.return_value = {"output": "ok"}
        extractor_llm = Mock()
        extractor_llm.invoke.return_value = Mock(
            content='{"updates":{"n_elems":10,"HY":{"value":20,"unit":"cm"},"HZ":{"value":12,"unit":"cm"},"FY":{"value":1.2,"unit":"kN"}},"user_intent":"provide_parameters"}'
        )
        run_agent_one_shot(
            agent_executor,
            "10 elements, 20 cm height, 12 cm width, 1.2 kN load",
            extractor_llm=extractor_llm,
        )
        called_input = agent_executor.invoke.call_args.args[0]["input"]
        self.assertIn("[Deterministic parser hint]", called_input)
        self.assertIn("[Deterministic parameter sheet]", called_input)
        self.assertIn('"n_elems": 10', called_input)

    def test_exit_detection_handles_farewell_phrases(self) -> None:
        self.assertTrue(is_exit_input("ok bye!"))
        self.assertTrue(is_exit_input("thanks"))
        self.assertTrue(is_exit_input("goodbye"))
        self.assertFalse(is_exit_input("width 12 cm"))

    def test_resolution_maps_c25_shorthand_to_catalog_grade(self) -> None:
        resolved = resolve_analysis_inputs({"material_hint": "C25", "length_m": 5.0, "n_elems": 11, "FY": 1.2})
        material = resolved["resolved"]["resolved_material"]
        self.assertEqual(material["name"], "C25/30")
        self.assertFalse(material["derived"])
        self.assertTrue(
            any("Concrete shorthand 'C25' was resolved to catalog grade 'C25/30'." in note for note in resolved["notes"])
        )

    def test_extract_scenario_from_plain_language(self) -> None:
        self.assertEqual(
            extract_scenario_from_text("simple beam 5 m beam, S355, IPE160, 12 elements, 1.2 kN load"),
            "simply_supported_midpoint_load",
        )
        self.assertEqual(
            extract_scenario_from_text("cantilever beam 3 m, S355, IPE160, 10 elements, 2 kN tip load"),
            "cantilever_tip_load",
        )

    def test_sheet_state_hint_builds(self) -> None:
        sheet = ParameterSheet()
        combined = build_state_aware_input(
            "cantilever beam 4 m, 10 elements, HY 20 cm, HZ 10 cm, C30 concrete, FY 1.2 kN",
            sheet,
        )
        self.assertIn("[Deterministic parameter sheet]", combined)


if __name__ == "__main__":
    unittest.main()
