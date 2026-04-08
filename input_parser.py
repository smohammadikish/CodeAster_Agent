from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


NUMBER = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
LENGTH_UNIT = r"mm|cm|m"
FORCE_UNIT = r"kN|N|kn|n"
REQUIRED_FIELDS = ("length_m", "n_elems", "E", "NU", "RHO", "HY", "HZ", "FY")
USER_INTENTS = {"provide_parameters", "confirm_run", "cancel", "question", "unknown"}
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "updates": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["m", "cm", "mm"]},
                    },
                    "required": ["value", "unit"],
                    "additionalProperties": False,
                },
                "n_elems": {"type": "integer"},
                "HY": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["m", "cm", "mm"]},
                    },
                    "required": ["value", "unit"],
                    "additionalProperties": False,
                },
                "HZ": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["m", "cm", "mm"]},
                    },
                    "required": ["value", "unit"],
                    "additionalProperties": False,
                },
                "FY": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["N", "kN"]},
                    },
                    "required": ["value", "unit"],
                    "additionalProperties": False,
                },
                "E": {"type": "number"},
                "NU": {"type": "number"},
                "RHO": {"type": "number"},
                "material_hint": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "user_intent": {"type": "string", "enum": sorted(USER_INTENTS)},
        "notes": {"type": "string"},
    },
    "required": ["updates", "user_intent"],
    "additionalProperties": False,
}
CONCRETE_C30_DEFAULTS = {
    "E": 30e9,
    "NU": 0.2,
    "RHO": 2400.0,
}


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _normalize_force_unit(unit: str) -> str:
    upper = unit.upper()
    return "kN" if upper == "KN" else "N"


def _extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in extractor response")
    return json.loads(text[start : end + 1])


def validate_state_update(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("State update payload must be an object")

    extra_keys = set(payload) - {"updates", "user_intent", "notes"}
    if extra_keys:
        raise ValueError(f"Unexpected top-level keys: {sorted(extra_keys)}")

    if "updates" not in payload or not isinstance(payload["updates"], dict):
        raise ValueError("Payload must include an 'updates' object")

    user_intent = payload.get("user_intent", "unknown")
    if user_intent not in USER_INTENTS:
        raise ValueError(f"Unsupported user_intent: {user_intent}")

    validated_updates: dict[str, Any] = {}
    for key, value in payload["updates"].items():
        if key not in {"length", "n_elems", "HY", "HZ", "FY", "E", "NU", "RHO", "material_hint"}:
            raise ValueError(f"Unexpected update key: {key}")
        if key in {"length", "HY", "HZ"}:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object with value/unit")
            if set(value) != {"value", "unit"}:
                raise ValueError(f"{key} must contain only value/unit")
            if value["unit"] not in {"m", "cm", "mm"}:
                raise ValueError(f"Invalid length unit for {key}: {value['unit']}")
            validated_updates[key] = {
                "value": float(value["value"]),
                "unit": value["unit"],
            }
        elif key == "FY":
            if not isinstance(value, dict):
                raise ValueError("FY must be an object with value/unit")
            if set(value) != {"value", "unit"}:
                raise ValueError("FY must contain only value/unit")
            if value["unit"] not in {"N", "kN"}:
                raise ValueError(f"Invalid force unit for FY: {value['unit']}")
            validated_updates[key] = {
                "value": float(value["value"]),
                "unit": value["unit"],
            }
        elif key == "n_elems":
            validated_updates[key] = int(value)
        elif key in {"E", "NU", "RHO"}:
            validated_updates[key] = float(value)
        elif key == "material_hint":
            validated_updates[key] = str(value)

    validated_payload = {
        "updates": validated_updates,
        "user_intent": user_intent,
    }
    if "notes" in payload:
        validated_payload["notes"] = str(payload["notes"])
    return validated_payload


def extract_state_update(llm: Any, text: str, current_state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Extract beam-analysis state updates from the latest user message.\n"
        "Return only one JSON object that matches this schema exactly.\n"
        f"Schema: {json.dumps(EXTRACTION_SCHEMA, ensure_ascii=True)}\n"
        f"Current state: {json.dumps(current_state, ensure_ascii=True)}\n"
        f"Latest user message: {text}\n"
        "Rules:\n"
        "- Only include fields explicitly supported by the schema.\n"
        "- If a value is not stated in the latest message, omit it from updates.\n"
        "- Preserve user units.\n"
        "- For unclear text, leave the field out instead of guessing.\n"
        "- Output JSON only."
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return validate_state_update(_extract_first_json_object(content))


def parse_beam_request(text: str) -> dict[str, Any]:
    """
    Extract common beam-analysis quantities from one user message.
    This is intentionally narrow and only handles explicit, common phrasings.
    """
    parsed: dict[str, Any] = {}

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

    if re.search(r"\bc30\b", text, flags=re.I):
        parsed["material_hint"] = "c30 concrete"

    return parsed


@dataclass
class ParameterSheet:
    values: dict[str, Any] = field(default_factory=dict)
    last_user_intent: str = "unknown"

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
        if "FY" in updates:
            self.values["FY"] = updates["FY"]["value"]
            self.values["force_unit"] = updates["FY"]["unit"]
        for scalar in ("E", "NU", "RHO", "material_hint"):
            if scalar in updates:
                self.values[scalar] = updates[scalar]
        self.last_user_intent = payload.get("user_intent", self.last_user_intent)
        self._apply_material_defaults()

    def update_from_text(self, text: str, llm: Any | None = None) -> dict[str, Any]:
        parsed = parse_beam_request(text)
        self.values.update(parsed)
        self._apply_material_defaults()
        if llm is not None:
            try:
                payload = extract_state_update(llm, text, self.to_dict())
                self.merge_update_payload(payload)
            except Exception:
                pass
        return parsed

    def _apply_material_defaults(self) -> None:
        if self.values.get("material_hint") == "c30 concrete":
            for key, value in CONCRETE_C30_DEFAULTS.items():
                self.values.setdefault(key, value)
            self.values.setdefault("material_defaults_source", "c30 concrete defaults")

    def missing_required_fields(self) -> list[str]:
        return [field for field in REQUIRED_FIELDS if field not in self.values]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def build_hint_block(self) -> str:
        parsed = self.to_dict()
        if not parsed:
            return ""

        missing = self.missing_required_fields()
        parts = [
            "[Deterministic parameter sheet]",
            "Use this accumulated structured state from the conversation if relevant:",
            json.dumps(parsed, ensure_ascii=True),
            f"Missing required fields: {', '.join(missing) if missing else 'none'}",
        ]
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
            ("FY", "Load"),
            ("force_unit", "Force unit"),
            ("E", "Young's modulus"),
            ("NU", "Poisson ratio"),
            ("RHO", "Density"),
            ("material_hint", "Material hint"),
            ("material_defaults_source", "Material defaults"),
            ("last_user_intent", "Last user intent"),
        ]
        lines = ["Current parameter sheet:"]
        for key, label in ordered_fields:
            if key in self.values:
                lines.append(f"- {label}: {self.values[key]}")
        missing = self.missing_required_fields()
        lines.append(
            "Missing required fields: " + (", ".join(missing) if missing else "none")
        )
        return "\n".join(lines)


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
