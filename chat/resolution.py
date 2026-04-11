from __future__ import annotations

import re
from typing import Any

from library import build_parametric_section, search_material, search_section
from units import convert_length


# This module turns user-facing hints into final analysis inputs.
# It resolves materials, sections, and parametric shapes.

TOP_LEVEL_REQUIRED_FIELDS = ("length_m", "n_elems", "FY")

# Short concrete names are expanded to Eurocode table names here.
CONCRETE_GRADE_MAP = {
    "C20": "C20/25",
    "C25": "C25/30",
    "C30": "C30/37",
    "C35": "C35/45",
    "C40": "C40/50",
    "C50": "C50/60",
}


# Small conversion helper used by the section resolver.
def _convert_optional_length(value: float | None, from_unit: str | None) -> float | None:
    if value is None or from_unit is None:
        return None
    return convert_length(float(value), from_unit, "mm")


# Concrete helper.
# It maps short names like C30 to a table/fallback grade like C30/37.
def _expand_concrete_hint(material_hint: str) -> str:
    cleaned = material_hint.strip().upper()
    concrete_match = re.fullmatch(r"C\s*(\d+)(?:\s+CONCRETE)?", cleaned)
    if not concrete_match:
        return material_hint
    shorthand = f"C{int(concrete_match.group(1))}"
    return CONCRETE_GRADE_MAP.get(shorthand, material_hint)


# Resolution block.
# It turns user-facing hints into final analysis inputs:
# material -> E, NU, RHO
# section -> A, IY, IZ, JX, AY, AZ
def resolve_analysis_inputs(values: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(values)
    missing: list[str] = []
    notes: list[str] = []

    for required in TOP_LEVEL_REQUIRED_FIELDS:
        if resolved.get(required) is None:
            missing.append(required)

    if not all(resolved.get(key) is not None for key in ("E", "NU", "RHO")):
        material_hint = resolved.get("material_hint")
        if material_hint:
            try:
                material = search_material(_expand_concrete_hint(material_hint))
                resolved["E"] = material["E"]
                resolved["NU"] = material["NU"]
                resolved["RHO"] = material["RHO"]
                resolved["resolved_material"] = material
            except Exception:
                notes.append(f"Material '{material_hint}' was not found in the catalog.")
        else:
            missing.extend(
                key for key in ("E", "NU", "RHO") if resolved.get(key) is None
            )

    if not all(resolved.get(key) is not None for key in ("E", "NU", "RHO")):
        missing.extend(
            key
            for key in ("E", "NU", "RHO")
            if resolved.get(key) is None and key not in missing
        )

    if not all(
        resolved.get(key) is not None
        for key in ("A_mm2", "IY_mm4", "IZ_mm4", "JX_mm4", "AY_mm2", "AZ_mm2")
    ):
        section_hint = resolved.get("section_hint")
        if section_hint:
            try:
                section = search_section(section_hint)
                resolved["A_mm2"] = float(section["A"])
                resolved["IY_mm4"] = float(section["Iy"])
                resolved["IZ_mm4"] = float(section["Iz"])
                resolved["JX_mm4"] = float(section["IT"])
                resolved["AY_mm2"] = float(section["Avy"])
                resolved["AZ_mm2"] = float(section["Avz"])
                resolved["resolved_section"] = section
            except Exception:
                notes.append(f"Section '{section_hint}' was not found in the catalog.")

    if not all(
        resolved.get(key) is not None
        for key in ("A_mm2", "IY_mm4", "IZ_mm4", "JX_mm4", "AY_mm2", "AZ_mm2")
    ):
        section_shape = resolved.get("section_shape")
        if section_shape == "circle":
            diameter_mm = _convert_optional_length(
                resolved.get("diameter"),
                resolved.get("diameter_unit"),
            )
            if diameter_mm is not None:
                section = build_parametric_section("circle", diameter_mm=diameter_mm)
                resolved["A_mm2"] = float(section["A"])
                resolved["IY_mm4"] = float(section["Iy"])
                resolved["IZ_mm4"] = float(section["Iz"])
                resolved["JX_mm4"] = float(section["IT"])
                resolved["AY_mm2"] = float(section["Avy"])
                resolved["AZ_mm2"] = float(section["Avz"])
                resolved["resolved_section"] = section
        else:
            hy_mm = _convert_optional_length(resolved.get("HY"), resolved.get("hy_unit"))
            hz_mm = _convert_optional_length(resolved.get("HZ"), resolved.get("hz_unit"))
            if hy_mm is not None and hz_mm is not None:
                section = build_parametric_section(
                    "rectangle",
                    width_mm=hz_mm,
                    height_mm=hy_mm,
                )
                resolved["A_mm2"] = float(section["A"])
                resolved["IY_mm4"] = float(section["Iy"])
                resolved["IZ_mm4"] = float(section["Iz"])
                resolved["JX_mm4"] = float(section["IT"])
                resolved["AY_mm2"] = float(section["Avy"])
                resolved["AZ_mm2"] = float(section["Avz"])
                resolved["resolved_section"] = section

    if not all(
        resolved.get(key) is not None
        for key in ("A_mm2", "IY_mm4", "IZ_mm4", "JX_mm4", "AY_mm2", "AZ_mm2")
    ):
        missing.extend(key for key in ("section_definition",) if key not in missing)

    deduped_missing: list[str] = []
    for item in missing:
        if item not in deduped_missing:
            deduped_missing.append(item)

    return {
        "resolved": resolved,
        "missing": deduped_missing,
        "notes": notes,
    }
