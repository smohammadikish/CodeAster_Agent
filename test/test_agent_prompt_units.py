import unittest
from unittest.mock import Mock

from main_agent import (build_agent, build_confirmation_summary, build_pipeline_kwargs_from_sheet,
                        build_post_run_idle_message,
                        build_post_run_message, build_post_success_message,
                        build_system_prompt, get_sheet_signature,
                        is_exit_input,
                        run_agent_one_shot, run_full_pipeline)
from input_parser import ParameterSheet, build_state_aware_input


class AgentPromptUnitTests(unittest.TestCase):
    def test_system_prompt_accepts_mixed_user_units(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("You must accept explicit user units and convert them internally.", prompt)
        self.assertIn("Do not ask the user to manually convert", prompt)
        self.assertIn("length_unit", prompt)
        self.assertIn("hy_unit", prompt)
        self.assertIn("hz_unit", prompt)
        self.assertIn("force_unit", prompt)
        self.assertIn("Use list_material_catalog for material lists.", prompt)
        self.assertIn("Catalog sections return Code_Aster-ready values A, IY, IZ, JX, AY, and AZ.", prompt)

    def test_build_agent_exposes_catalog_and_pipeline_tools(self) -> None:
        agent = build_agent()
        tool_names = {tool.name for tool in agent.tools}
        self.assertIn("list_material_catalog", tool_names)
        self.assertIn("get_material_catalog", tool_names)
        self.assertIn("list_section_catalog", tool_names)
        self.assertIn("get_section_catalog", tool_names)
        self.assertIn("build_parametric_section_catalog", tool_names)
        self.assertIn("run_full_pipeline", tool_names)

    def test_run_full_pipeline_accepts_per_parameter_units(self) -> None:
        result = run_full_pipeline.invoke(
            {
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
                "confirm": False,
            }
        )
        self.assertIn("simulation has NOT been confirmed", result)
        self.assertNotIn("Missing required parameters", result)

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

    def test_pipeline_kwargs_and_confirmation_summary_from_sheet(self) -> None:
        sheet = ParameterSheet()
        build_state_aware_input("analyse a 4 m c30 concrete beam", sheet)
        build_state_aware_input("12 elemens/ hight 20 cm, wedth 12 cm / fy 1.3 kn", sheet)
        kwargs = build_pipeline_kwargs_from_sheet(sheet)
        summary = build_confirmation_summary(sheet)
        self.assertEqual(kwargs["length_m"], 4.0)
        self.assertEqual(kwargs["length_unit"], "m")
        self.assertEqual(kwargs["hy_unit"], "cm")
        self.assertEqual(kwargs["hz_unit"], "cm")
        self.assertEqual(kwargs["force_unit"], "kN")
        self.assertEqual(kwargs["E"], 33e9)
        self.assertTrue(kwargs["confirm"])
        self.assertIn("AY_mm2", kwargs)
        self.assertIn("AZ_mm2", kwargs)
        self.assertIn("All required parameters are available.", summary)
        self.assertIn("- AY:", summary)
        self.assertIn("- AZ:", summary)
        self.assertIn("Reply with 'yes', 'confirm', or 'run it'", summary)

    def test_exit_detection_handles_farewell_phrases(self) -> None:
        self.assertTrue(is_exit_input("ok bye!"))
        self.assertTrue(is_exit_input("thanks"))
        self.assertTrue(is_exit_input("goodbye"))
        self.assertFalse(is_exit_input("width 12 cm"))

    def test_post_run_message_is_not_a_confirmation_prompt(self) -> None:
        message = build_post_run_message()
        self.assertIn("already been completed", message)
        self.assertIn("change one or more parameters", message)

    def test_post_success_message_removes_parameter_dump(self) -> None:
        run_output = (
            "Simulation finished successfully.\n"
            "Run ID: 20260408_185800_beam_6e5acb74\n"
            "Run folder: C:\\test\\run\n\n"
            "Input units: length=m, HY=m, HZ=cm, force=N\n"
            "Used parameters:\n"
            "{...}"
        )
        message = build_post_success_message(run_output)
        self.assertIn("Simulation finished successfully.", message)
        self.assertIn("Run ID: 20260408_185800_beam_6e5acb74", message)
        self.assertIn("Would you like to do another analysis or finish the process?", message)
        self.assertNotIn("Input units:", message)
        self.assertNotIn("Used parameters:", message)

    def test_rerun_signature_changes_when_sheet_changes(self) -> None:
        sheet = ParameterSheet()
        build_state_aware_input("2.35 m beam, 11 elements, hy 0.125 m, hz 10.34 cm, fy 1230 n, c30 concrete", sheet)
        first_signature = get_sheet_signature(sheet)
        build_state_aware_input("change fy to 1.234 kn", sheet)
        second_signature = get_sheet_signature(sheet)
        self.assertNotEqual(first_signature, second_signature)

    def test_post_run_idle_message_is_short(self) -> None:
        self.assertEqual(
            build_post_run_idle_message(),
            "Would you like to do another analysis or finish the process?",
        )


if __name__ == "__main__":
    unittest.main()
