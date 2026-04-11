from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .section_calculators import circular_section, rectangular_section


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "catalog_manifest.json"


class CatalogError(KeyError):
    pass


class UnknownFamilyError(CatalogError):
    pass


class UnknownMaterialError(CatalogError):
    pass


class UnknownSectionError(CatalogError):
    pass


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _ascii_fold(value).upper())


def _parse_number(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.startswith("~"):
        text = text[1:]
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() else number


HEADER_ALIASES = {
    "A (MM2)": "A",
    "A": "A",
    "IY (MM4)": "Iy",
    "IY": "Iy",
    "IZ (MM4)": "Iz",
    "IZ": "Iz",
    "IT (MM4)": "IT",
    "IT": "IT",
    "AVY (MM2)": "Avy",
    "AVY": "Avy",
    "AVZ (MM2)": "Avz",
    "AVZ": "Avz",
    "E (GPA)": "E_GPa",
    "NU": "NU",
    "RHO (KG/M3)": "RHO_kg_m3",
    "FY (MPA)": "FY_MPa",
    "F_CK (MPA)": "fck_MPa",
    "PROFILE": "Profile",
    "SECTION": "Section",
    "CLASS": "class",
    "STEEL TYPE": "steel type",
    "ALLOY": "Alloy",
}

MATERIAL_NAME_FIELDS = {
    "steel type": "name",
    "Alloy": "name",
    "class": "name",
}


def _normalize_header(header: str) -> str:
    folded = _ascii_fold(header).strip().upper()
    return HEADER_ALIASES.get(folded, header.strip())


def _strip_family_tokens(value: str, family: str, aliases: list[str]) -> str:
    result = value
    for token in [family, *aliases]:
        if token:
            result = re.sub(re.escape(token), " ", result, flags=re.IGNORECASE)
    return result


def _normalize_lookup_name(value: str, family: str, aliases: list[str]) -> str:
    return _normalize_token(_strip_family_tokens(value, family, aliases))


def _convert_material_row(family: str, row: dict[str, Any], reference: str) -> dict[str, Any]:
    item = dict(row)
    item["family"] = family
    item["reference"] = reference
    item["derived"] = False

    if "E_GPa" in item and item["E_GPa"] is not None:
        item["E"] = float(item.pop("E_GPa")) * 1e9
    if "FY_MPa" in item and item["FY_MPa"] is not None:
        item["FY"] = float(item.pop("FY_MPa")) * 1e6
    if "RHO_kg_m3" in item and item["RHO_kg_m3"] is not None:
        item["RHO"] = float(item.pop("RHO_kg_m3"))
    if "fck_MPa" in item and item["fck_MPa"] is not None:
        item["FCK"] = float(item.pop("fck_MPa")) * 1e6
    return item


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw_row in reader:
            row: dict[str, Any] = {}
            for header, value in raw_row.items():
                if header is None:
                    continue
                row[_normalize_header(header)] = _parse_number(value or "")
            rows.append(row)
        return rows


def _extract_concrete_grade_tokens(value: str) -> tuple[int, int] | None:
    match = re.search(r"C\s*(\d+)\s*/\s*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _build_concrete_formula_material(material_name: str, family: str, config: dict[str, Any]) -> dict[str, Any]:
    grade_tokens = _extract_concrete_grade_tokens(material_name)
    if grade_tokens is None:
        raise UnknownMaterialError(f"Material '{material_name}' not found in family '{family}'")
    fck_mpa, upper_strength = grade_tokens
    fcm_mpa = fck_mpa + 8.0
    e_pa = 22.0 * ((fcm_mpa / 10.0) ** 0.3) * 1e9
    fixed = config["fixed_properties"]
    return {
        "family": family,
        "name": f"C{fck_mpa}/{upper_strength}",
        "class": f"C{fck_mpa}/{upper_strength}",
        "E": e_pa,
        "NU": float(fixed["NU"]),
        "RHO": float(fixed["RHO"]),
        "FCK": fck_mpa * 1e6,
        "FCM": fcm_mpa * 1e6,
        "reference": f"{config['reference']} formula fallback",
        "derived": True,
    }


@dataclass(frozen=True)
class FamilyIndex:
    canonical_name: str
    aliases: list[str]
    rows: dict[str, dict[str, Any]]


class Catalog:
    def __init__(self) -> None:
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self._material_families = self._build_material_index()
        self._section_families = self._build_section_index()

    def _resolve_family(self, family: str, families: dict[str, FamilyIndex]) -> str:
        normalized = _normalize_token(family)
        for canonical_name, index in families.items():
            if normalized == _normalize_token(canonical_name):
                return canonical_name
            if normalized in {_normalize_token(alias) for alias in index.aliases}:
                return canonical_name
        raise UnknownFamilyError(f"Unknown family '{family}'")

    def _build_material_index(self) -> dict[str, FamilyIndex]:
        families: dict[str, FamilyIndex] = {}
        for family, config in self.manifest["materials"].items():
            csv_path = BASE_DIR / config["source_file"]
            rows_by_name: dict[str, dict[str, Any]] = {}
            for row in _read_csv_rows(csv_path):
                name_field = config["key_field"]
                raw_name = row.get(name_field)
                if not raw_name:
                    continue
                normalized_name = _normalize_lookup_name(str(raw_name), family, config.get("aliases", []))
                converted = _convert_material_row(family, row, config["reference"])
                converted[MATERIAL_NAME_FIELDS.get(name_field, name_field)] = raw_name
                converted["name"] = str(raw_name)
                rows_by_name[normalized_name] = converted
            families[family] = FamilyIndex(family, config.get("aliases", []), rows_by_name)
        return families

    def _build_section_index(self) -> dict[str, FamilyIndex]:
        families: dict[str, FamilyIndex] = {}
        for family, config in self.manifest["sections"].items():
            if config["source_type"] == "generated":
                families[family] = FamilyIndex(family, config.get("aliases", []), {})
                continue
            csv_path = BASE_DIR / config["source_file"]
            key_field = config["key_field"]
            rows_by_name: dict[str, dict[str, Any]] = {}
            for row in _read_csv_rows(csv_path):
                raw_name = row.get(key_field)
                if not raw_name:
                    continue
                normalized_name = _normalize_lookup_name(str(raw_name), family, config.get("aliases", []))
                row["family"] = family
                row["reference"] = config["reference"]
                row[key_field] = raw_name
                rows_by_name[normalized_name] = row
            families[family] = FamilyIndex(family, config.get("aliases", []), rows_by_name)
        return families

    def list_material_families(self) -> list[str]:
        return sorted(self._material_families)

    def list_section_families(self) -> list[str]:
        return sorted(self._section_families)

    def list_materials(self, family: str) -> list[str]:
        canonical = self._resolve_family(family, self._material_families)
        return sorted(item["name"] for item in self._material_families[canonical].rows.values())

    def list_sections(self, family: str) -> list[str]:
        canonical = self._resolve_family(family, self._section_families)
        return sorted(
            next((row.get(field) for field in ("Profile", "Section") if row.get(field)), "")
            for row in self._section_families[canonical].rows.values()
        )

    def get_material(self, family: str, material_name: str) -> dict[str, Any]:
        canonical = self._resolve_family(family, self._material_families)
        config = self.manifest["materials"][canonical]
        aliases = self._material_families[canonical].aliases
        normalized_name = _normalize_lookup_name(material_name, canonical, aliases)
        found = self._material_families[canonical].rows.get(normalized_name)
        if found is not None:
            return dict(found)
        if canonical == "concrete":
            return _build_concrete_formula_material(material_name, canonical, config)
        raise UnknownMaterialError(f"Material '{material_name}' not found in family '{canonical}'")

    def search_material(self, material_name: str) -> dict[str, Any]:
        normalized = _normalize_token(material_name)
        for family, index in self._material_families.items():
            maybe = index.rows.get(_normalize_lookup_name(material_name, family, index.aliases))
            if maybe is not None:
                return dict(maybe)
            if family == "concrete" and _extract_concrete_grade_tokens(material_name):
                return _build_concrete_formula_material(material_name, family, self.manifest["materials"][family])
            if normalized == _normalize_token(family):
                continue
        raise UnknownMaterialError(f"Material '{material_name}' not found in the catalog")

    def get_section(self, family: str, section_name: str) -> dict[str, Any]:
        canonical = self._resolve_family(family, self._section_families)
        if canonical == "parametric":
            raise UnknownSectionError("Use build_parametric_section for generated shapes")
        aliases = self._section_families[canonical].aliases
        normalized_name = _normalize_lookup_name(section_name, canonical, aliases)
        found = self._section_families[canonical].rows.get(normalized_name)
        if found is None:
            raise UnknownSectionError(f"Section '{section_name}' not found in family '{canonical}'")
        return dict(found)

    def build_parametric_section(self, shape: str, **dimensions_mm: float) -> dict[str, Any]:
        normalized_shape = _normalize_token(shape)
        if normalized_shape in {"RECTANGLE", "RECTANGULAR"}:
            width = dimensions_mm.get("width_mm") or dimensions_mm.get("b_mm")
            height = dimensions_mm.get("height_mm") or dimensions_mm.get("h_mm")
            if width is None or height is None:
                raise ValueError("Rectangle requires width_mm/b_mm and height_mm/h_mm")
            result = rectangular_section(width, height)
        elif normalized_shape in {"CIRCLE", "CIRCULAR"}:
            diameter = dimensions_mm.get("diameter_mm") or dimensions_mm.get("d_mm")
            if diameter is None:
                raise ValueError("Circle requires diameter_mm/d_mm")
            result = circular_section(diameter)
        else:
            raise UnknownSectionError(f"Unsupported parametric shape '{shape}'")
        result["family"] = "parametric"
        return result


_CATALOG = Catalog()

MATERIALS: dict[str, dict[str, dict[str, Any]]] = {
    family: {item["name"]: dict(item) for item in index.rows.values()}
    for family, index in _CATALOG._material_families.items()
}

SECTIONS: dict[str, dict[str, dict[str, Any]]] = {
    family: {
        str(row.get("Profile") or row.get("Section")): dict(row)
        for row in index.rows.values()
    }
    for family, index in _CATALOG._section_families.items()
    if family != "parametric"
}


def list_material_families() -> list[str]:
    return _CATALOG.list_material_families()


def list_section_families() -> list[str]:
    return _CATALOG.list_section_families()


def list_materials(family: str) -> list[str]:
    return _CATALOG.list_materials(family)


def list_sections(family: str) -> list[str]:
    return _CATALOG.list_sections(family)


def get_material(family: str, material_name: str) -> dict[str, Any]:
    return _CATALOG.get_material(family, material_name)


def search_material(material_name: str) -> dict[str, Any]:
    return _CATALOG.search_material(material_name)


def get_section(family: str, section_name: str) -> dict[str, Any]:
    return _CATALOG.get_section(family, section_name)


def build_parametric_section(shape: str, **dimensions_mm: float) -> dict[str, Any]:
    return _CATALOG.build_parametric_section(shape, **dimensions_mm)


def search_section(section_name: str) -> dict[str, Any]:
    normalized = _normalize_token(section_name)
    for family, index in _CATALOG._section_families.items():
        if family == "parametric":
            continue
        maybe = index.rows.get(_normalize_lookup_name(section_name, family, index.aliases))
        if maybe is not None:
            return dict(maybe)
        if normalized == _normalize_token(family):
            continue
    raise UnknownSectionError(f"Section '{section_name}' not found in the catalog")
