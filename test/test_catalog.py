import math
import unittest

from library.catalog import (build_parametric_section, get_material,
                             get_section, list_material_families,
                             list_sections, search_material)


class CatalogTests(unittest.TestCase):
    def test_material_families_include_uploaded_catalogs(self) -> None:
        self.assertEqual(
            list_material_families(),
            ["aluminum", "concrete", "steel", "wood"],
        )

    def test_steel_name_matching_is_not_rigid(self) -> None:
        material = get_material("steel", "s 355 steel")
        self.assertEqual(material["name"], "S355")
        self.assertEqual(material["FY"], 355e6)
        self.assertEqual(material["E"], 210e9)

    def test_aluminium_alias_resolves_to_aluminum_family(self) -> None:
        material = get_material("aluminium", "6082 t6")
        self.assertEqual(material["name"], "6082-T6")
        self.assertEqual(material["family"], "aluminum")

    def test_concrete_formula_fallback_when_grade_not_in_table(self) -> None:
        material = get_material("concrete", "c32/40 concrete")
        expected_e = 22.0 * (((32.0 + 8.0) / 10.0) ** 0.3) * 1e9
        self.assertTrue(material["derived"])
        self.assertEqual(material["name"], "C32/40")
        self.assertAlmostEqual(material["E"], expected_e)
        self.assertEqual(material["NU"], 0.2)
        self.assertEqual(material["RHO"], 2400.0)
        self.assertEqual(material["FCK"], 32e6)
        self.assertEqual(material["FCM"], 40e6)

    def test_section_family_aliases_cover_timber_catalog(self) -> None:
        section = get_section("wood", "100x200")
        self.assertEqual(section["Profile"], "100x200")
        available = list_sections("timber")
        self.assertIn("100x200", available)

    def test_search_material_finds_glulam_entry(self) -> None:
        material = search_material("gl24h")
        self.assertEqual(material["name"], "GL24h (Glulam)")

    def test_rectangular_section_uses_requested_torsion_formula(self) -> None:
        section = build_parametric_section("rectangular", width_mm=100, height_mm=200)
        self.assertAlmostEqual(section["A"], 20000.0)
        self.assertAlmostEqual(section["Iy"], 100 * 200**3 / 12.0)
        self.assertAlmostEqual(section["Iz"], 200 * 100**3 / 12.0)
        a = 200.0
        b = 100.0
        expected_it = (a * b**3 / 3.0) * (1.0 - 0.21 * (b / a) * (1.0 - (b**4) / (12.0 * a**4)))
        self.assertAlmostEqual(section["IT"], expected_it)
        self.assertAlmostEqual(section["Avy"], (5.0 / 6.0) * 20000.0)
        self.assertAlmostEqual(section["Avz"], (5.0 / 6.0) * 20000.0)

    def test_circular_section_uses_documented_shear_area_approximation(self) -> None:
        section = build_parametric_section("circle", d_mm=100)
        area = math.pi * 100**2 / 4.0
        self.assertAlmostEqual(section["A"], area)
        self.assertAlmostEqual(section["Iy"], math.pi * 100**4 / 64.0)
        self.assertAlmostEqual(section["Iz"], math.pi * 100**4 / 64.0)
        self.assertAlmostEqual(section["IT"], math.pi * 100**4 / 32.0)
        self.assertAlmostEqual(section["Avy"], 0.9 * area)
        self.assertAlmostEqual(section["Avz"], 0.9 * area)


if __name__ == "__main__":
    unittest.main()
