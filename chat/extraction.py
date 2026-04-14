from __future__ import annotations

import json
from typing import Any

# This module defines the strict JSON contract used for optional LLM extraction.

USER_INTENTS = {
    "provide_parameters",
    "confirm_run",
    "cancel",
    "question",
    "unknown",
}
SCENARIOS = {
    "cantilever_tip_load",
    "simply_supported_midpoint_load",
}

# LLM extraction schema.
# It limits what the model can write back into the state.
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
                "diameter": {
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
                "scenario": {"type": "string", "enum": sorted(SCENARIOS)},
                "E": {"type": "number"},
                "NU": {"type": "number"},
                "RHO": {"type": "number"},
                "A_mm2": {"type": "number"},
                "IY_mm4": {"type": "number"},
                "IZ_mm4": {"type": "number"},
                "JX_mm4": {"type": "number"},
                "AY_mm2": {"type": "number"},
                "AZ_mm2": {"type": "number"},
                "material_hint": {"type": "string"},
                "section_hint": {"type": "string"},
                "section_shape": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "user_intent": {"type": "string", "enum": sorted(USER_INTENTS)},
        "notes": {"type": "string"},
    },
    "required": ["updates", "user_intent"],
    "additionalProperties": False,
}


def _extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in extractor response")
    return json.loads(text[start : end + 1])


# Validation block for structured LLM updates.
# It keeps the state format predictable and safe.
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
    scalar_keys = {
        "scenario",
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
    }
    dimension_keys = {"length", "HY", "HZ", "diameter"}

    for key, value in payload["updates"].items():
        if key not in dimension_keys | {"n_elems", "FY"} | scalar_keys:
            raise ValueError(f"Unexpected update key: {key}")
        if key in dimension_keys:
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
        elif key in {"scenario", "material_hint", "section_hint", "section_shape"}:
            validated_updates[key] = str(value)
        else:
            validated_updates[key] = float(value)

    validated_payload = {
        "updates": validated_updates,
        "user_intent": user_intent,
    }
    if "notes" in payload:
        validated_payload["notes"] = str(payload["notes"])
    return validated_payload


# LLM extraction block.
# It asks the model for a strict JSON update only.
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
        "- Use the latest message together with the current state to resolve short follow-ups.\n"
        "- If the user says 'first' or 'second' while choosing between supported scenarios, map them to:\n"
        "  first -> cantilever_tip_load\n"
        "  second -> simply_supported_midpoint_load\n"
        "- If the user says 'simple beam', 'simply supported', or equivalent, set scenario to simply_supported_midpoint_load.\n"
        "- If the user says 'cantilever', set scenario to cantilever_tip_load.\n"
        "- If the user says things like 'same as before but 12 elements', only update the changed field.\n"
        "- If the user replies with a catalog item like 'S235' or 'IPE160', put it in the matching hint field.\n"
        "- Do not move a scenario name into material_hint or section_hint.\n"
        "- Do not move material or section hints into each other unless the latest message clearly names that category.\n"
        "- For unclear text, leave the field out instead of guessing.\n"
        "- Output JSON only."
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return validate_state_update(_extract_first_json_object(content))
