from __future__ import annotations

import math
from typing import Any


def _validate_positive(name: str, value: float) -> float:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def _rectangle_torsion_constant(a_mm: float, b_mm: float) -> float:
    """
    Saint-Venant torsion constant approximation for a solid rectangle.

    The larger side is ``a`` and the smaller side is ``b``:

        IT = ab^3 / 3 * [1 - 0.21 * (b/a) * (1 - b^4 / (12 a^4))]

    This follows the engineering formula provided by the user and is commonly
    used for prismatic rectangular sections when a full warping analysis is
    not required.
    """
    return (a_mm * b_mm**3 / 3.0) * (
        1.0 - 0.21 * (b_mm / a_mm) * (1.0 - (b_mm**4) / (12.0 * a_mm**4))
    )


def rectangular_section(width_mm: float, height_mm: float) -> dict[str, Any]:
    """
    Build a solid rectangular section in solver-facing units.

    Output units:
    - A, Avy, Avz: mm^2
    - Iy, Iz, IT: mm^4

    Engineering approximations:
    - ``IT`` uses the Saint-Venant approximation for a solid rectangle.
    - ``Avy`` and ``Avz`` use ``5/6 * A``. This is the standard Timoshenko
      shear-correction approximation for a solid rectangular section and is
      appropriate for preliminary beam analysis when a more detailed shear
      distribution model is not available.
    """
    width = _validate_positive("width_mm", width_mm)
    height = _validate_positive("height_mm", height_mm)
    area = width * height
    iy = width * height**3 / 12.0
    iz = height * width**3 / 12.0
    a_side = max(width, height)
    b_side = min(width, height)
    it = _rectangle_torsion_constant(a_side, b_side)
    shear_area = (5.0 / 6.0) * area
    return {
        "shape": "rectangle",
        "width_mm": width,
        "height_mm": height,
        "A": area,
        "Iy": iy,
        "Iz": iz,
        "IT": it,
        "Avy": shear_area,
        "Avz": shear_area,
        "units": {
            "A": "mm2",
            "Iy": "mm4",
            "Iz": "mm4",
            "IT": "mm4",
            "Avy": "mm2",
            "Avz": "mm2",
        },
        "reference": "Parametric section formulas",
        "approximation_notes": [
            "IT uses the Saint-Venant solid-rectangle approximation supplied in the project requirements.",
            "Avy and Avz use 5/6 of the gross area as a Timoshenko beam shear-area approximation.",
        ],
    }


def circular_section(diameter_mm: float) -> dict[str, Any]:
    """
    Build a solid circular section in solver-facing units.

    Engineering approximations:
    - ``IT`` is exact for a solid circular section: pi*d^4/32.
    - ``Avy`` and ``Avz`` use ``0.9 * A``. This is a common engineering
      approximation for the effective shear area of solid circular sections
      in beam models.
    """
    diameter = _validate_positive("diameter_mm", diameter_mm)
    area = math.pi * diameter**2 / 4.0
    inertia = math.pi * diameter**4 / 64.0
    torsion = math.pi * diameter**4 / 32.0
    shear_area = 0.9 * area
    return {
        "shape": "circle",
        "diameter_mm": diameter,
        "A": area,
        "Iy": inertia,
        "Iz": inertia,
        "IT": torsion,
        "Avy": shear_area,
        "Avz": shear_area,
        "units": {
            "A": "mm2",
            "Iy": "mm4",
            "Iz": "mm4",
            "IT": "mm4",
            "Avy": "mm2",
            "Avz": "mm2",
        },
        "reference": "Parametric section formulas",
        "approximation_notes": [
            "IT is exact for a solid circular section.",
            "Avy and Avz use 0.9 of the gross area as an engineering shear-area approximation.",
        ],
    }
