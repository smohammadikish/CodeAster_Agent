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
            return _json(
                {
                    "family": family,
                    "materials": list_materials(family),
                    "source_note": (
                        "These are material entries currently available in the local catalog. "
                        "Catalog materials are database suggestions, not the only valid choices."
                    ),
                    "guidance": (
                        "Where applicable, catalog entries are sourced from Eurocode references. "
                        "If the user wants a material outside the catalog, ask for explicit E, NU, and RHO."
                    ),
                }
            )
        return _json(
            {
                "families": list_material_families(),
                "source_note": (
                    "These families come from the local material catalog used by the application."
                ),
            }
        )
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
        material = get_material(family, material_name)
        return _json(
            {
                **material,
                "catalog_note": (
                    "This material comes from the local catalog used by the application."
                ),
                "guidance": (
                    "Use this as a catalog suggestion/reference. "
                    "If the user prefers another material, ask for explicit properties."
                ),
            }
        )
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
            return _json(
                {
                    "family": family,
                    "sections": list_sections(family),
                    "source_note": (
                        "These are section entries currently available in the local catalog. "
                        "Catalog sections are database suggestions, not the only valid choices."
                    ),
                    "guidance": (
                        "If the user wants a section outside the catalog, ask for explicit section properties "
                        "or dimensions for a parametric section."
                    ),
                }
            )
        return _json(
            {
                "families": list_section_families(),
                "source_note": (
                    "These families come from the local section catalog used by the application."
                ),
            }
        )
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
                "catalog_note": (
                    "This section comes from the local catalog used by the application."
                ),
                "guidance": (
                    "Use this as a catalog suggestion/reference. "
                    "If the user prefers another section, ask for explicit section properties or dimensions."
                ),
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
