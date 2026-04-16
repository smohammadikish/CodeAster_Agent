"""Render beam analysis reports from Word templates with placeholders."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile


def _load_docx_template():
    """Import docxtpl only when report rendering is requested."""
    try:
        docxtpl_module = importlib.import_module("docxtpl")
    except ImportError as exc:
        raise RuntimeError(
            "Report rendering requires the 'docxtpl' package. "
            "Install project dependencies first."
        ) from exc
    return docxtpl_module.DocxTemplate


def _supports_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expose supports by name so templates can access them directly."""
    return {
        support["name"]: support
        for support in result.get("supports", [])
        if "name" in support
    }


def _build_template_context(result: dict[str, Any]) -> dict[str, Any]:
    """Build the context object passed into the Word template."""
    # Ensure material and section have default values for optional fields
    material = result.get("material", {})
    if "name" not in material:
        material = {**material, "name": "Unspecified"}
    
    section = result.get("section", {})
    for key in ("width", "height"):
        if key not in section:
            section = {**section, key: None}
    
    context = {
        **result,
        "material": material,
        "section": section,
        "supports_by_name": _supports_by_name(result),
    }
    case_type = result.get("case_type")
    response_points = {
        point.get("label"): point for point in result.get("response_points", [])
    }
    context["report"] = {
        "case": {
            "cantilever_tip_load": "cantilever",
            "simply_supported_midspan_point_load": "simply_supported",
        }.get(case_type, "generic")
    }
    context["displacements"] = {
        "free_end": (
            {
                **response_points["TIP"],
                "theta": response_points["TIP"].get("rz"),
            }
            if "TIP" in response_points
            else None
        ),
        "midspan": (
            {
                **response_points["MIDSPAN"],
                "theta": response_points["MIDSPAN"].get("rz"),
            }
            if "MIDSPAN" in response_points
            else None
        ),
    }
    context["extrema"] = {
        **result.get("extrema", {}),
        "bending_moment": {
            **result.get("extrema", {}).get("bending_moment", {}),
            "max": result.get("extrema", {}).get("bending_moment", {}).get("value"),
            "max_abs": abs(
                result.get("extrema", {}).get("bending_moment", {}).get("value", 0.0)
            ),
        },
    }
    return context


def _resolve_placeholder_value(context: dict[str, Any], expression: str) -> Any:
    """Resolve dotted placeholders with optional list indexes.
    
    Returns the value, or a placeholder string if the path doesn't exist.
    """
    token_pattern = re.compile(r"([a-zA-Z_]\w*)|\[(\d+)\]")
    current: Any = context
    try:
        for match in token_pattern.finditer(expression.strip()):
            key = match.group(1)
            index = match.group(2)
            if key is not None:
                if not isinstance(current, dict):
                    return f"[{expression}]"
                current = current.get(key)
                if current is None:
                    return f"[{expression}]"
                continue
            if not isinstance(current, (list, tuple)):
                return f"[{expression}]"
            current = current[int(index)]
        return current
    except (KeyError, IndexError, TypeError, ValueError):
        return f"[{expression}]"


def _render_text_template(template_text: str, context: dict[str, Any]) -> str:
    """Replace {{ ... }} placeholders inside a plain-text template."""

    def replace(match: re.Match[str]) -> str:
        # Plain-text templates only need simple value substitution.
        return str(_resolve_placeholder_value(context, match.group(1)))

    return re.sub(r"{{\s*(.*?)\s*}}", replace, template_text)


def _template_contains_jinja_markers(template_path: Path) -> bool:
    """Check whether the Word template contains Jinja markers anywhere in its XML."""
    with ZipFile(template_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            xml_text = archive.read(name).decode("utf-8", errors="ignore")
            if "{{" in xml_text or "{%" in xml_text:
                return True
    return False


def choose_default_template(result: dict[str, Any], root: str | Path = ".") -> Path:
    """Pick the built-in template that matches the solved beam type."""
    template_relative_path = {
        "cantilever_tip_load": "template_report/cantilever_beam_report.txt",
        "simply_supported_midspan_point_load": "template_report/simply_supported_beam_report.txt",
    }.get(result.get("case_type"))
    if template_relative_path is None:
        raise ValueError(
            "No built-in report template matches this support layout. "
            "Pass an explicit template path instead."
        )

    template_path = (Path(root) / template_relative_path).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Report template not found: {template_path}")
    return template_path


def render_report(
    result: dict[str, Any],
    output_path_like: str | Path,
    template_path_like: str | Path | None = None,
    root: str | Path = ".",
) -> Path:
    """Render a text or Word report using either the chosen or built-in template."""
    template_path = (
        Path(template_path_like).resolve()
        if template_path_like is not None
        else choose_default_template(result, root=root)
    )
    output_path = Path(output_path_like)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = _build_template_context(result)

    if template_path.suffix.lower() == ".txt":
        # Text templates are rendered without extra third-party packages.
        rendered_text = _render_text_template(
            template_path.read_text(encoding="utf-8"),
            context,
        )
        output_path.write_text(rendered_text, encoding="utf-8")
        return output_path.resolve()

    DocxTemplate = _load_docx_template()
    if not _template_contains_jinja_markers(template_path):
        raise ValueError(
            "The selected report template does not contain any Jinja placeholders. "
            "Add markers such as {{ beam.length }} inside the Word file first."
        )

    # Word templates are rendered through docxtpl after a basic placeholder check.
    template = DocxTemplate(str(template_path))
    template.render(context)
    template.save(str(output_path))
    return output_path.resolve()
