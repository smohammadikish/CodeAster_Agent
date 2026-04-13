from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from library import (resolve_material_family_name, resolve_section_family_name,
                     search_material, search_section)

from .extraction import extract_state_update
from .parsing import parse_beam_request
from .resolution import resolve_analysis_inputs


# This module keeps the running conversation state and builds helper text
# that is attached to the LLM input.


def _is_material_family_only(value: str) -> bool:
    try:
        resolve_material_family_name(value)
    except Exception:
        return False
    try:
        search_material(value)
        return False
    except Exception:
        return True


def _is_section_family_only(value: str) -> bool:
    try:
        resolve_section_family_name(value)
    except Exception:
        return False
    try:
        search_section(value)
        return False
    except Exception:
        return True


@dataclass
class ParameterSheet:
    # This class stores the running conversation state.
    # It merges new user input and reports what is still missing.
    values: dict[str, Any] = field(default_factory=dict)
    last_user_intent: str = "unknown"

    def _normalize_contextual_hints(
        self, updates: dict[str, Any], previous_values: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(updates)
        previous_resolution = resolve_analysis_inputs(previous_values)
        previous_resolved = previous_resolution["resolved"]
        material_fixed = previous_resolved.get("resolved_material") is not None
        section_fixed = previous_resolved.get("resolved_section") is not None
        section_missing = "section_definition" in previous_resolution["missing"]
        material_missing = any(
            key in previous_resolution["missing"] for key in ("E", "NU", "RHO")
        )

        material_hint = normalized.get("material_hint")
        if (
            isinstance(material_hint, str)
            and material_fixed
            and not section_fixed
            and section_missing
            and "section_hint" not in normalized
            and _is_section_family_only(material_hint)
        ):
            normalized["section_hint"] = material_hint
            normalized.pop("material_hint", None)

        section_hint = normalized.get("section_hint")
        if (
            isinstance(section_hint, str)
            and section_fixed
            and not material_fixed
            and material_missing
            and "material_hint" not in normalized
            and _is_material_family_only(section_hint)
        ):
            normalized["material_hint"] = section_hint
            normalized.pop("section_hint", None)

        return normalized

    def merge_update_payload(self, payload: dict[str, Any]) -> None:
        updates = payload.get("updates", {})
        if "length" in updates:
            self.values["length_m"] = updates["length"]["value"]
            self.values["length_unit"] = updates["length"]["unit"]
        if "n_elems" in updates:
            self.values["n_elems"] = updates["n_elems"]
        if "HY" in updates:
            self.values["HY"] = updates["HY"]["value"]
            self.values["hy_unit"] = updates["HY"]["unit"]
        if "HZ" in updates:
            self.values["HZ"] = updates["HZ"]["value"]
            self.values["hz_unit"] = updates["HZ"]["unit"]
        if "diameter" in updates:
            self.values["diameter"] = updates["diameter"]["value"]
            self.values["diameter_unit"] = updates["diameter"]["unit"]
        if "FY" in updates:
            self.values["FY"] = updates["FY"]["value"]
            self.values["force_unit"] = updates["FY"]["unit"]
        for scalar in (
            "E",
            "NU",
            "RHO",
            "A_mm2",
            "IY_mm4",
            "IZ_mm4",
            "JX_mm4",
            "AY_mm2",
            "AZ_mm2",
            "material_hint",
            "section_hint",
            "section_shape",
        ):
            if scalar in updates:
                self.values[scalar] = updates[scalar]
        self.last_user_intent = payload.get("user_intent", self.last_user_intent)

    def update_from_text(self, text: str, llm: Any | None = None) -> dict[str, Any]:
        previous_values = self.to_dict()
        parsed = self._normalize_contextual_hints(parse_beam_request(text), previous_values)
        self.values.update(parsed)
        if llm is not None:
            try:
                payload = extract_state_update(llm, text, self.to_dict())
                payload["updates"] = self._normalize_contextual_hints(
                    payload.get("updates", {}),
                    previous_values,
                )
                self.merge_update_payload(payload)
            except Exception:
                pass
        return parsed

    def missing_required_fields(self) -> list[str]:
        return resolve_analysis_inputs(self.values)["missing"]

    def resolved_state(self) -> dict[str, Any]:
        return resolve_analysis_inputs(self.values)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def build_hint_block(self) -> str:
        parsed = self.to_dict()
        if not parsed:
            return ""

        resolution = self.resolved_state()
        resolved = resolution["resolved"]
        parts = [
            "[Deterministic parameter sheet]",
            "Use this accumulated structured state from the conversation if relevant:",
            json.dumps(parsed, ensure_ascii=True),
            f"Missing required fields: {', '.join(resolution['missing']) if resolution['missing'] else 'none'}",
        ]
        material = resolved.get("resolved_material")
        if material:
            material_name = material.get("name")
            material_ref = material.get("reference")
            material_mode = "formula" if material.get("derived") else "catalog"
            parts.append(
                f"Resolved material: {material_name} [{material_mode}] ({material_ref})"
            )
        section = resolved.get("resolved_section")
        if section:
            section_name = section.get("Profile") or section.get("Section") or section.get("shape")
            section_ref = section.get("reference", "generated section")
            parts.append(f"Resolved section: {section_name} ({section_ref})")
        if resolution["notes"]:
            parts.append("Resolution notes: " + " | ".join(resolution["notes"]))
        return "\n".join(parts)

    def build_cli_summary(self) -> str:
        if not self.values:
            return "Parameter sheet is empty."

        ordered_fields = [
            ("length_m", "Beam length"),
            ("length_unit", "Length unit"),
            ("n_elems", "Elements"),
            ("HY", "Section height"),
            ("hy_unit", "HY unit"),
            ("HZ", "Section width"),
            ("hz_unit", "HZ unit"),
            ("diameter", "Section diameter"),
            ("diameter_unit", "Diameter unit"),
            ("FY", "Load"),
            ("force_unit", "Force unit"),
            ("E", "Young's modulus"),
            ("NU", "Poisson ratio"),
            ("RHO", "Density"),
            ("A_mm2", "Area"),
            ("IY_mm4", "Second moment IY"),
            ("IZ_mm4", "Second moment IZ"),
            ("JX_mm4", "Torsion constant JX"),
            ("AY_mm2", "Shear area AY"),
            ("AZ_mm2", "Shear area AZ"),
            ("material_hint", "Material hint"),
            ("section_hint", "Section hint"),
            ("section_shape", "Section shape"),
            ("last_user_intent", "Last user intent"),
        ]
        lines = ["Current parameter sheet:"]
        for key, label in ordered_fields:
            if key in self.values:
                lines.append(f"- {label}: {self.values[key]}")
        resolution = self.resolved_state()
        if resolution["notes"]:
            lines.append("Resolution notes: " + " | ".join(resolution["notes"]))
        lines.append(
            "Missing required fields: "
            + (", ".join(resolution["missing"]) if resolution["missing"] else "none")
        )
        return "\n".join(lines)


# Prompt helper block.
# These functions append deterministic state to the LLM input.
def build_parser_hint(text: str, parsed: dict[str, Any] | None = None) -> str:
    parsed = parsed if parsed is not None else parse_beam_request(text)
    if not parsed:
        return text

    hint = (
        "\n\n[Deterministic parser hint]\n"
        "Use these extracted values from the latest user message if relevant:\n"
        f"{json.dumps(parsed, ensure_ascii=True)}"
    )
    return text + hint


def build_state_aware_input(text: str, sheet: ParameterSheet, llm: Any | None = None) -> str:
    parsed = sheet.update_from_text(text, llm=llm)
    state_block = sheet.build_hint_block()
    parser_hint = build_parser_hint(text, parsed)
    if not state_block:
        return parser_hint
    return parser_hint + "\n\n" + state_block
