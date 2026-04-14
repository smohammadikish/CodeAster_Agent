import unittest
from unittest.mock import Mock

from input_parser import (ParameterSheet, build_parser_hint,
                          build_state_aware_input, parse_beam_request,
                          resolve_analysis_inputs, validate_state_update)


class InputParserTests(unittest.TestCase):
    def test_parse_initial_beam_request(self) -> None:
        parsed = parse_beam_request("analyse a 5m c30 concrete beam")
        self.assertEqual(parsed["length_m"], 5.0)
        self.assertEqual(parsed["length_unit"], "m")
        self.assertEqual(parsed["material_hint"], "c30 concrete")

    def test_parse_mixed_unit_followup(self) -> None:
        parsed = parse_beam_request("10 elements/ 20cm hight and 12cm width / 1.2 kn")
        self.assertEqual(parsed["n_elems"], 10)
        self.assertEqual(parsed["HY"], 20.0)
        self.assertEqual(parsed["hy_unit"], "cm")
        self.assertEqual(parsed["HZ"], 12.0)
        self.assertEqual(parsed["hz_unit"], "cm")
        self.assertEqual(parsed["FY"], 1.2)
        self.assertEqual(parsed["force_unit"], "kN")

    def test_parse_common_typos_and_label_before_value(self) -> None:
        parsed = parse_beam_request("12 elemens/ hight 20 cm, wedth 12 cm / fy 1.3 kn")
        self.assertEqual(parsed["n_elems"], 12)
        self.assertEqual(parsed["HY"], 20.0)
        self.assertEqual(parsed["HZ"], 12.0)
        self.assertEqual(parsed["FY"], 1.3)
        later = parse_beam_request("width .13 m")
        self.assertEqual(later["HZ"], 0.13)
        self.assertEqual(later["hz_unit"], "m")
        compact_h = parse_beam_request("analyse a c30 concrete 2.35 m beam. 1230 n load, .125 m h and 10.34 cm wedth.")
        self.assertEqual(compact_h["HY"], 0.125)
        self.assertEqual(compact_h["hy_unit"], "m")

    def test_parse_simple_beam_infers_supported_scenario(self) -> None:
        parsed = parse_beam_request("simple beam 5 m beam, S355, IPE160, 12 elements, 1.2 kN load")
        self.assertEqual(parsed["scenario"], "simply_supported_midpoint_load")

    def test_parse_cantilever_infers_cantilever_scenario(self) -> None:
        parsed = parse_beam_request("cantilever beam 3 m, S355, IPE160, 10 elements, 2 kN tip load")
        self.assertEqual(parsed["scenario"], "cantilever_tip_load")

    def test_build_parser_hint_appends_structured_extraction(self) -> None:
        text = "10 elements, 20 cm height, 12 cm width, 1.2 kN load"
        hinted = build_parser_hint(text)
        self.assertIn("[Deterministic parser hint]", hinted)
        self.assertIn('"n_elems": 10', hinted)
        self.assertIn('"HY": 20.0', hinted)
        self.assertIn('"force_unit": "kN"', hinted)

    def test_parameter_sheet_accumulates_across_turns(self) -> None:
        sheet = ParameterSheet()
        first = build_state_aware_input("analyse a 5m c30 concrete beam", sheet)
        second = build_state_aware_input(
            "10 elements/ 20cm hight and 12cm width / 1.2 kn",
            sheet,
        )

        self.assertIn('"length_m": 5.0', first)
        self.assertIn('"material_hint": "c30 concrete"', first)
        self.assertIn('"length_m": 5.0', second)
        self.assertIn('"n_elems": 10', second)
        self.assertIn('"HY": 20.0', second)
        self.assertIn('"HZ": 12.0', second)
        self.assertIn('"FY": 1.2', second)
        self.assertIn("Missing required fields:", second)

    def test_newer_value_overwrites_previous_value(self) -> None:
        sheet = ParameterSheet()
        build_state_aware_input("5 m beam", sheet)
        build_state_aware_input("6 m beam", sheet)
        self.assertEqual(sheet.to_dict()["length_m"], 6.0)

    def test_cli_summary_shows_known_and_missing_fields(self) -> None:
        sheet = ParameterSheet()
        build_state_aware_input("5 m beam with 10 elements", sheet)
        summary = sheet.build_cli_summary()
        self.assertIn("Current parameter sheet:", summary)
        self.assertIn("- Beam length: 5.0", summary)
        self.assertIn("- Length unit: m", summary)
        self.assertIn("- Elements: 10", summary)
        self.assertIn("Missing required fields: FY, E, NU, RHO, section_definition", summary)

    def test_validate_and_merge_llm_state_update(self) -> None:
        payload = validate_state_update(
            {
                "updates": {
                    "HY": {"value": 0.225, "unit": "m"},
                    "FY": {"value": 1245, "unit": "N"},
                    "material_hint": "c30 concrete",
                },
                "user_intent": "provide_parameters",
            }
        )
        sheet = ParameterSheet()
        sheet.merge_update_payload(payload)
        self.assertEqual(sheet.to_dict()["HY"], 0.225)
        self.assertEqual(sheet.to_dict()["hy_unit"], "m")
        self.assertEqual(sheet.to_dict()["FY"], 1245.0)
        self.assertEqual(sheet.to_dict()["force_unit"], "N")
        self.assertEqual(sheet.last_user_intent, "provide_parameters")

    def test_state_aware_input_uses_llm_extraction_to_fill_hy(self) -> None:
        llm = Mock()
        llm.invoke.return_value = Mock(
            content='{"updates":{"HY":{"value":0.225,"unit":"m"}},"user_intent":"provide_parameters"}'
        )
        sheet = ParameterSheet()
        build_state_aware_input(".225 m is hight", sheet, llm=llm)
        self.assertEqual(sheet.to_dict()["HY"], 0.225)
        self.assertEqual(sheet.to_dict()["hy_unit"], "m")

    def test_llm_is_primary_for_scenario_interpretation(self) -> None:
        llm = Mock()
        llm.invoke.return_value = Mock(
            content='{"updates":{"scenario":"simply_supported_midpoint_load","length":{"value":4,"unit":"m"},"n_elems":10,"HY":{"value":0.12,"unit":"m"},"HZ":{"value":0.10,"unit":"m"},"material_hint":"S235","FY":{"value":1500,"unit":"N"}},"user_intent":"provide_parameters"}'
        )
        sheet = ParameterSheet()
        build_state_aware_input(
            "simpl beam, 4 m beam, 10 elements, HY=0.12 m, HZ=0.10 m, steel S235, 1500 N load",
            sheet,
            llm=llm,
        )
        self.assertEqual(sheet.to_dict()["scenario"], "simply_supported_midpoint_load")
        self.assertEqual(sheet.to_dict()["material_hint"], "S235")

    def test_llm_can_resolve_second_as_second_scenario(self) -> None:
        llm = Mock()
        llm.invoke.return_value = Mock(
            content='{"updates":{"scenario":"simply_supported_midpoint_load"},"user_intent":"provide_parameters"}'
        )
        sheet = ParameterSheet()
        build_state_aware_input("second", sheet, llm=llm)
        self.assertEqual(sheet.to_dict()["scenario"], "simply_supported_midpoint_load")

    def test_same_as_before_updates_only_changed_field(self) -> None:
        llm = Mock()
        llm.invoke.side_effect = [
            Mock(
                content='{"updates":{"scenario":"cantilever_tip_load","length":{"value":5,"unit":"m"},"n_elems":10,"material_hint":"S355","section_hint":"IPE160","FY":{"value":1.2,"unit":"kN"}},"user_intent":"provide_parameters"}'
            ),
            Mock(
                content='{"updates":{"n_elems":12},"user_intent":"provide_parameters"}'
            ),
        ]
        sheet = ParameterSheet()
        build_state_aware_input(
            "cantilever beam 5 m, S355, IPE160, 10 elements, 1.2 kN tip load",
            sheet,
            llm=llm,
        )
        build_state_aware_input("same as before but 12 elements", sheet, llm=llm)
        self.assertEqual(sheet.to_dict()["scenario"], "cantilever_tip_load")
        self.assertEqual(sheet.to_dict()["n_elems"], 12)
        self.assertEqual(sheet.to_dict()["section_hint"], "IPE160")

    def test_resolution_returns_shear_areas_for_named_section(self) -> None:
        resolved = resolve_analysis_inputs(
            {
                "length_m": 5.0,
                "n_elems": 10,
                "FY": -1000.0,
                "material_hint": "s355",
                "section_hint": "ipe 200",
            }
        )["resolved"]
        self.assertIsNotNone(resolved["AY_mm2"])
        self.assertIsNotNone(resolved["AZ_mm2"])

    def test_state_model_does_not_mix_material_and_section_hints(self) -> None:
        sheet = ParameterSheet({"material_hint": "6082-T6"})
        build_state_aware_input("aluminum", sheet)
        self.assertEqual(sheet.to_dict()["material_hint"], "aluminum")
        self.assertNotIn("section_hint", sheet.to_dict())


if __name__ == "__main__":
    unittest.main()
