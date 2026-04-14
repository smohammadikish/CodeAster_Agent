from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Store the beam material properties used by OpenSees.
@dataclass(slots=True)
class BeamMaterial:
    elastic_modulus: float
    poisson_ratio: float | None = None
    density: float | None = None


# Store the beam section properties used by OpenSees.
@dataclass(slots=True)
class BeamSection:
    area: float
    inertia_z: float
    inertia_y: float | None = None


# Store the load and the node where the load is applied.
@dataclass(slots=True)
class PointLoad:
    load_type: str
    direction: str
    magnitude: float
    x: float
    node: int

    @property
    def fx(self) -> float:
        if self.direction == "x_positive":
            return self.magnitude
        if self.direction == "x_negative":
            return -self.magnitude
        return 0.0

    @property
    def fy(self) -> float:
        if self.direction == "y_positive":
            return self.magnitude
        if self.direction == "y_negative":
            return -self.magnitude
        return 0.0

    @property
    def mz(self) -> float:
        return 0.0


# Group all input data for one beam analysis case.
@dataclass(slots=True)
class BeamInput:
    analysis_name: str
    case_type: str
    opensees_executable: str
    output_root: Path
    model_id: str | None
    units: dict[str, str]
    length: float
    num_elements: int
    material: BeamMaterial
    section: BeamSection
    load: PointLoad
    stations: int
    response_positions: tuple[float, ...]
    meta: dict[str, object]

    # Return the total number of nodes from the number of elements.
    @property
    def num_nodes(self) -> int:
        return self.num_elements + 1

    @property
    def support_type(self) -> str:
        if self.case_type == "cantilever_tip_load":
            return "cantilever"
        return "simply_supported"


# Keep track of the original source used to build one analysis case.
@dataclass(slots=True)
class InputSource:
    source_type: str
    root_path: Path
    snapshot_files: tuple[Path, ...]
