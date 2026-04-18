from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .extraction import extract_state_update
from .parsing import MATERIAL_PATTERN, SECTION_PATTERN, parse_beam_request
from .resolution import resolve_analysis_inputs

# This module keeps the running conversation state and builds helper text
# that is attached to the LLM input.

@dataclass
class ParameterSheet:
    # This class stores the running conversation state.
    # It merges new user input and reports what is still missing.
    values: dict[str, Any] = field(default_factory=dict)
    last_user_intent: str = "unknown"

    @staticmethod
    def _normalize_hint(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _looks_like_material(value: str) -> bool:
        return bool(MATERIAL_PATTERN.search(value))

    @staticmethod
    def _looks_like_section(value: str) -> bool:
        return bool(SECTION_PATTERN.search(value))

    def _apply_hint_safety(self, updates: dict[str, Any]) -> None:
        material_hint_raw = updates.get("material_hint")
        section_hint_raw = updates.get("section_hint")

        material_hint = self._normalize_hint(material_hint_raw) if material_hint_raw is not None else None
        section_hint = self._normalize_hint(section_hint_raw) if section_hint_raw is not None else None

        if material_hint:
            if self._looks_like_section(material_hint) and not self._looks_like_material(material_hint):
                if not section_hint:
                    self.values["section_hint"] = material_hint
            else:
                self.values["material_hint"] = material_hint

        if section_hint:
            if self._looks_like_material(section_hint) and not self._looks_like_section(section_hint):
                if not material_hint:
                    self.values["material_hint"] = section_hint
            else:
                self.values["section_hint"] = section_hint

    def sanitize_hints(self) -> bool:
        changed = False
        material_hint = self._normalize_hint(self.values.get("material_hint"))
        section_hint = self._normalize_hint(self.values.get("section_hint"))

        if material_hint and self._looks_like_section(material_hint) and not self._looks_like_material(material_hint):
            if not section_hint:
                self.values["section_hint"] = material_hint
            self.values.pop("material_hint", None)
            changed = True

        if section_hint and self._looks_like_material(section_hint) and not self._looks_like_section(section_hint):
            if not self.values.get("material_hint"):
                self.values["material_hint"] = section_hint
            self.values.pop("section_hint", None)
            changed = True

        return changed

    def merge_update_payload(self, payload: dict[str, Any]) -> None:
        updates = payload.get("updates", {})
        if "scenario" in updates:
            self.values["scenario"] = updates["scenario"]
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
            "section_shape",
        ):
            if scalar in updates:
                self.values[scalar] = updates[scalar]
        self._apply_hint_safety(updates)
        self.last_user_intent = payload.get("user_intent", self.last_user_intent)

    def _apply_fallback_updates(self, fallback_updates: dict[str, Any]) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for key, value in fallback_updates.items():
            if self.values.get(key) != value:
                self.values[key] = value
                applied[key] = value
        return applied

    def update_from_text(self, text: str, llm: Any | None = None) -> dict[str, Any]:
        self.sanitize_hints()
        llm_payload: dict[str, Any] | None = None
        llm_updates: dict[str, Any] = {}
        if llm is not None:
            try:
                llm_payload = extract_state_update(llm, text, self.to_dict())
                llm_updates = dict(llm_payload.get("updates", {}))
                self.merge_update_payload(llm_payload)
            except Exception:
                llm_payload = None
                llm_updates = {}

        fallback_updates = parse_beam_request(text)
        fallback_applied = self._apply_fallback_updates(
            {
                key: value
                for key, value in fallback_updates.items()
                if key not in llm_updates
            }
        )

        return {
            "llm_payload": llm_payload,
            "llm_updates": llm_updates,
            "fallback_updates": fallback_updates,
            "applied_updates": {**llm_updates, **fallback_applied},
            "user_intent": (
                llm_payload.get("user_intent", self.last_user_intent)
                if llm_payload
                else self.last_user_intent
            ),
        }

    def missing_required_fields(self) -> list[str]:
        return resolve_analysis_inputs(self.values)["missing"]

    def resolved_state(self) -> dict[str, Any]:
        self.sanitize_hints()
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


def build_state_aware_input(
    text: str,
    sheet: ParameterSheet,
    llm: Any | None = None,
    turn_update: dict[str, Any] | None = None,
) -> str:
    update = turn_update if turn_update is not None else sheet.update_from_text(text, llm=llm)
    state_block = sheet.build_hint_block()
    parser_hint = build_parser_hint(text, update.get("fallback_updates"))
    if not state_block:
        return parser_hint
    return parser_hint + "\n\n" + state_block
