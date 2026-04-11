import json
import unittest

from tools.catalog_tools import (build_parametric_section_catalog_tool,
                                 get_material_catalog_tool,
                                 get_section_catalog_tool,
                                 list_material_catalog_tool)


class CatalogToolTests(unittest.TestCase):
    def test_list_material_catalog_tool_returns_families(self) -> None:
        payload = json.loads(list_material_catalog_tool.invoke({"family": ""}))
        self.assertIn("steel", payload["families"])

    def test_get_material_catalog_tool_resolves_flexible_name(self) -> None:
        payload = json.loads(
            get_material_catalog_tool.invoke(
                {"family": "steel", "material_name": "s 355 steel"}
            )
        )
        self.assertEqual(payload["name"], "S355")
        self.assertEqual(payload["E"], 210e9)

    def test_get_section_catalog_tool_returns_code_aster_mapping(self) -> None:
        payload = json.loads(
            get_section_catalog_tool.invoke(
                {"family": "steel", "section_name": "ipe 200"}
            )
        )
        self.assertEqual(payload["code_aster"]["CARA"], ["A", "IY", "IZ", "JX", "AY", "AZ"])
        self.assertEqual(payload["code_aster"]["VALE"][0], 2848)
        self.assertEqual(payload["code_aster"]["VALE"][4], payload["Avy"])
        self.assertEqual(payload["code_aster"]["VALE"][5], payload["Avz"])

    def test_build_parametric_section_catalog_tool_returns_code_aster_mapping(self) -> None:
        payload = json.loads(
            build_parametric_section_catalog_tool.invoke(
                {"shape": "rectangle", "width_mm": 100.0, "height_mm": 200.0}
            )
        )
        self.assertEqual(payload["code_aster"]["CARA"], ["A", "IY", "IZ", "JX", "AY", "AZ"])
        self.assertEqual(payload["code_aster"]["VALE"][0], 20000.0)
        self.assertEqual(payload["code_aster"]["VALE"][4], payload["Avy"])
        self.assertEqual(payload["code_aster"]["VALE"][5], payload["Avz"])


if __name__ == "__main__":
    unittest.main()
