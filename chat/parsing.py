from __future__ import annotations

import re


# This module reads direct values from the latest user message.
# It handles simple regex-based extraction for geometry, loads, and hints.

NUMBER = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
LENGTH_UNIT = r"mm|cm|m"
FORCE_UNIT = r"kN|N|kn|n"

# Pattern block for common material names typed by the user.
MATERIAL_PATTERN = re.compile(
    r"\b("
    r"S\s*[- ]?\s*\d{3}|"
    r"C\s*\d+\s*/\s*\d+|"
    r"GL\s*\d+\s*[A-Z]?|"
    r"C\s*\d+\b(?:\s+concrete)?|"
    r"\d{4}\s*-\s*T\d+"
    r")\b",
    flags=re.IGNORECASE,
)

# Pattern block for common section names typed by the user.
SECTION_PATTERN = re.compile(
    r"\b("
    r"IPE\s*\d+|"
    r"RHS\s*\d+\s*x\s*\d+\s*x\s*\d+|"
    r"SHS\s*\d+\s*x\s*\d+\s*x\s*\d+|"
    r"CHS\s*(?:Ø|O|D)?\s*\d+\s*x\s*\d+|"
    r"RCB-[A-Z0-9x+\-]+|"
    r"\d+\s*x\s*\d+"
    r")\b",
    flags=re.IGNORECASE,
)


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _normalize_force_unit(unit: str) -> str:
    upper = unit.upper()
    return "kN" if upper == "KN" else "N"


def _normalize_hint(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _find_material_hint(text: str) -> str | None:
    match = MATERIAL_PATTERN.search(text)
    return _normalize_hint(match.group(1)) if match else None


def _find_section_hint(text: str) -> str | None:
    match = SECTION_PATTERN.search(text)
    return _normalize_hint(match.group(1)) if match else None


def _find_section_shape(text: str) -> str | None:
    if re.search(r"\b(rectangle|rectangular)\b", text, flags=re.I):
        return "rectangle"
    if re.search(r"\b(circle|circular|round)\b", text, flags=re.I):
        return "circle"
    return None


def parse_beam_request(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}

    elem_match = re.search(
        r"\b(?P<n>\d+)\s*(?:elements?|elems?|elemens?)\b", text, flags=re.I
    )
    if elem_match:
        parsed["n_elems"] = int(elem_match.group("n"))

    length_patterns = [
        rf"\b(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})(?:\s+\w+){{0,4}}\s+beam\b",
        rf"\b(?:length|beam length|span)\s*[:=]?\s*(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\b",
    ]
    for pattern in length_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed["length_m"] = _to_float(match.group("value"))
            parsed["length_unit"] = match.group("unit").lower()
            break

    hy_patterns = [
        rf"\b(?:hy|height|hight|depth|high)\s*[:=]?\s*(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\b",
        rf"(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\s*(?:h|hy|height|hight|high|deep|depth)\b",
    ]
    for pattern in hy_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed["HY"] = _to_float(match.group("value"))
            parsed["hy_unit"] = match.group("unit").lower()
            break

    hz_patterns = [
        rf"\b(?:hz|width|wedth|breadth|wide)\s*[:=]?\s*(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\b",
        rf"(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\s*(?:width|wedth|wide|breadth)\b",
    ]
    for pattern in hz_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed["HZ"] = _to_float(match.group("value"))
            parsed["hz_unit"] = match.group("unit").lower()
            break

    diameter_patterns = [
        rf"\b(?:d|dia|diameter)\s*[:=]?\s*(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\b",
        rf"\b(?P<value>{NUMBER})\s*(?P<unit>{LENGTH_UNIT})\s*(?:diameter|dia)\b",
    ]
    for pattern in diameter_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed["diameter"] = _to_float(match.group("value"))
            parsed["diameter_unit"] = match.group("unit").lower()
            break

    force_patterns = [
        rf"\b(?:fy|load)\s*[:=]?\s*(?P<value>{NUMBER})\s*(?P<unit>{FORCE_UNIT})\b",
        rf"\b(?P<value>{NUMBER})\s*(?P<unit>{FORCE_UNIT})\b(?:\s*(?:load|force))?\b",
    ]
    for pattern in force_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed["FY"] = _to_float(match.group("value"))
            parsed["force_unit"] = _normalize_force_unit(match.group("unit"))
            break

    material_hint = _find_material_hint(text)
    if material_hint:
        parsed["material_hint"] = material_hint

    section_hint = _find_section_hint(text)
    if section_hint:
        parsed["section_hint"] = section_hint

    section_shape = _find_section_shape(text)
    if section_shape:
        parsed["section_shape"] = section_shape

    return parsed
