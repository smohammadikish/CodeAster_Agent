from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from library import (build_parametric_section, get_material, get_section,
                     list_material_families, list_materials,
                     list_section_families, list_sections)


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


@tool("list_material_catalog", return_direct=False)
def list_material_catalog_tool(family: str = "") -> str:
    """List material families or available materials in one family."""
    try:
        if family.strip():
            return _json({"family": family, "materials": list_materials(family)})
        return _json({"families": list_material_families()})
    except Exception as exc:
        return _json({"error": "catalog_lookup_failed", "message": str(exc)})


@tool("get_material_catalog", return_direct=False)
def get_material_catalog_tool(family: str, material_name: str) -> str:
    """
    Resolve one material from the catalog.

    Returned units:
    - E in Pa
    - NU dimensionless
    - RHO in kg/m3
    - FY/FCK/FCM in Pa when available
    """
    try:
        return _json(get_material(family, material_name))
    except Exception as exc:
        return _json(
            {
                "error": "material_not_found",
                "family": family,
                "query": material_name,
                "message": str(exc),
            }
        )


@tool("list_section_catalog", return_direct=False)
def list_section_catalog_tool(family: str = "") -> str:
    """List section families or available sections in one family."""
    try:
        if family.strip():
            return _json({"family": family, "sections": list_sections(family)})
        return _json({"families": list_section_families()})
    except Exception as exc:
        return _json({"error": "catalog_lookup_failed", "message": str(exc)})


@tool("get_section_catalog", return_direct=False)
def get_section_catalog_tool(family: str, section_name: str) -> str:
    """
    Resolve one section and provide the Code_Aster GENERALE mapping.

    Required Code_Aster values:
    - A
    - IY
    - IZ
    - JX
    - AY
    - AZ
    """
    try:
        section = get_section(family, section_name)
        return _json(
            {
                **section,
                "code_aster": {
                    "SECTION": "GENERALE",
                    "CARA": ["A", "IY", "IZ", "JX", "AY", "AZ"],
                    "VALE": [
                        section.get("A"),
                        section.get("Iy"),
                        section.get("Iz"),
                        section.get("IT"),
                        section.get("Avy"),
                        section.get("Avz"),
                    ],
                },
            }
        )
    except Exception as exc:
        return _json(
            {
                "error": "section_not_found",
                "family": family,
                "query": section_name,
                "message": str(exc),
            }
        )


@tool("build_parametric_section_catalog", return_direct=False)
def build_parametric_section_catalog_tool(
    shape: str,
    width_mm: float | None = None,
    height_mm: float | None = None,
    diameter_mm: float | None = None,
) -> str:
    """Build a generated rectangle/circle section and provide Code_Aster mapping."""
    try:
        kwargs: dict[str, float] = {}
        if width_mm is not None:
            kwargs["width_mm"] = width_mm
        if height_mm is not None:
            kwargs["height_mm"] = height_mm
        if diameter_mm is not None:
            kwargs["diameter_mm"] = diameter_mm
        section = build_parametric_section(shape, **kwargs)
        return _json(
            {
                **section,
                "code_aster": {
                    "SECTION": "GENERALE",
                    "CARA": ["A", "IY", "IZ", "JX", "AY", "AZ"],
                    "VALE": [
                        section.get("A"),
                        section.get("Iy"),
                        section.get("Iz"),
                        section.get("IT"),
                        section.get("Avy"),
                        section.get("Avz"),
                    ],
                },
            }
        )
    except Exception as exc:
        return _json(
            {
                "error": "parametric_section_failed",
                "shape": shape,
                "message": str(exc),
            }
        )
