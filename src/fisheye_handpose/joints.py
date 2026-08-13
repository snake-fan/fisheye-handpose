"""Versioned canonical hand-landmark definitions and explicit mapping records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class LandmarkKind(StrEnum):
    """Geometric meaning of a canonical landmark."""

    REFERENCE_POINT = "REFERENCE_POINT"
    ARTICULATION_CENTER = "ARTICULATION_CENTER"
    SURFACE_POINT = "SURFACE_POINT"


class MappingQuality(StrEnum):
    """How a target landmark was obtained from a backend-native representation.

    ``EXACT`` requires the same operational construction and version as the canonical
    target. A shared name or broad anatomical interpretation is not sufficient.
    """

    EXACT = "EXACT"
    REGRESSED = "REGRESSED"
    DERIVED = "DERIVED"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class LandmarkDefinition:
    """Normative semantic definition for one landmark.

    ``definition`` is part of the schema contract. Canonical references are operational
    targets, not assertions that a unique anatomical point is directly observable from an
    image. Matching a name or array index is not sufficient for an ``EXACT`` mapping: the
    source must use the same construction and version.
    """

    name: str
    kind: LandmarkKind
    definition: str
    construction_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.definition.strip() or not self.construction_id.strip():
            raise ValueError("landmark name, definition, and construction ID must be non-empty")


@dataclass(frozen=True, slots=True)
class LandmarkSchema:
    version: str
    definitions: tuple[LandmarkDefinition, ...]
    parents: tuple[int, ...]
    tip_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("schema version must be non-empty")
        if len(self.definitions) != 21 or len(self.parents) != 21:
            raise ValueError("a canonical hand schema must contain exactly 21 landmarks")
        if len(set(self.names)) != len(self.names):
            raise ValueError("landmark names must be unique")
        for index, parent in enumerate(self.parents):
            if parent >= index or parent < -1:
                raise ValueError(f"invalid parent {parent} for landmark {index}")
        if self.parents[0] != -1 or any(self.parents[i] == -1 for i in range(1, 21)):
            raise ValueError("fhp21 requires one root at landmark 0")
        if tuple(sorted(set(self.tip_indices))) != self.tip_indices:
            raise ValueError("tip indices must be sorted and unique")
        if any(index < 0 or index >= len(self.definitions) for index in self.tip_indices):
            raise ValueError("tip index is outside the landmark schema")
        if any(
            self.definitions[index].kind is not LandmarkKind.SURFACE_POINT
            for index in self.tip_indices
        ):
            raise ValueError("every tip index must identify a surface landmark")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((parent, child) for child, parent in enumerate(self.parents) if parent >= 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "names": list(self.names),
            "parents": list(self.parents),
            "tip_indices": list(self.tip_indices),
            "edges": [list(edge) for edge in self.edges],
            "landmark_count": len(self.names),
            "definitions": [
                {
                    "index": index,
                    "name": definition.name,
                    "kind": definition.kind,
                    "definition": definition.definition,
                    "construction_id": definition.construction_id,
                }
                for index, definition in enumerate(self.definitions)
            ],
        }


@dataclass(frozen=True, slots=True)
class LandmarkMappingEntry:
    """Mapping of one canonical target landmark from backend-native landmarks."""

    target_index: int
    quality: MappingQuality
    source_indices: tuple[int, ...]
    source_construction_id: str | None
    method: str
    source_definition: str

    def __post_init__(self) -> None:
        if self.target_index < 0:
            raise ValueError("target_index must be non-negative")
        if any(index < 0 for index in self.source_indices):
            raise ValueError("source landmark indices must be non-negative")
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ValueError("source landmark indices must be unique")
        if not self.method.strip() or not self.source_definition.strip():
            raise ValueError("mapping method and source definition must be non-empty")
        if self.quality is MappingQuality.MISSING and self.source_indices:
            raise ValueError("a MISSING mapping cannot name source indices")
        if self.quality is not MappingQuality.MISSING and not self.source_indices:
            raise ValueError("a non-MISSING mapping must name at least one source index")
        if self.quality is MappingQuality.MISSING and self.source_construction_id is not None:
            raise ValueError("a MISSING mapping cannot name a source construction")
        if self.quality is not MappingQuality.MISSING and not self.source_construction_id:
            raise ValueError("a non-MISSING mapping must name its source construction/version")


@dataclass(frozen=True, slots=True)
class LandmarkMappingRecord:
    """Auditable mapping from one native joint set to a canonical schema.

    Entries are ordered by target index and cover every target exactly once. ``mapping_id``
    is intended to be persisted in evidence/output provenance and should change whenever
    definitions, regressors, vertices, or derivation logic change.
    """

    mapping_id: str
    source_joint_set_id: str
    target_schema_version: str
    source_landmark_names: tuple[str, ...]
    entries: tuple[LandmarkMappingEntry, ...]
    provenance: str

    def __post_init__(self) -> None:
        if not self.mapping_id or not self.source_joint_set_id or not self.target_schema_version:
            raise ValueError("mapping and joint-set identifiers must be non-empty")
        if self.target_schema_version != FHP21.version:
            raise ValueError(f"canonical mapping target must be {FHP21.version}")
        if len(self.entries) != len(FHP21.definitions):
            raise ValueError("a canonical mapping record must cover all 21 targets")
        if not self.provenance.strip():
            raise ValueError("mapping provenance must be non-empty")
        if len(set(self.source_landmark_names)) != len(self.source_landmark_names):
            raise ValueError("source landmark names must be unique")
        expected_targets = tuple(range(len(self.entries)))
        actual_targets = tuple(entry.target_index for entry in self.entries)
        if actual_targets != expected_targets:
            raise ValueError("mapping entries must cover target indices once and in order")
        source_count = len(self.source_landmark_names)
        for entry in self.entries:
            if any(index >= source_count for index in entry.source_indices):
                raise ValueError(
                    f"mapping target {entry.target_index} references an absent source landmark"
                )
            if (
                entry.quality is MappingQuality.EXACT
                and entry.source_construction_id
                != FHP21.definitions[entry.target_index].construction_id
            ):
                raise ValueError(
                    "EXACT mapping requires the canonical operational construction/version"
                )

    @property
    def qualities(self) -> tuple[MappingQuality, ...]:
        return tuple(entry.quality for entry in self.entries)


_JOINT_CENTER = (
    "The estimated three-dimensional centre of rotation of the named anatomical "
    "articulation; it is not a mesh vertex or a point on the visible skin surface."
)
_TIP_SURFACE = (
    "The fhp21/v1 canonical distal fingertip surface reference. Its operational realization "
    "(for example an annotation rule, selected mesh vertex, or regressor) is declared by "
    "the mapping record; it does not claim a unique, directly observable distal-most "
    "soft-tissue point or distal phalanx joint centre."
)


def _joint(name: str) -> LandmarkDefinition:
    return LandmarkDefinition(
        name,
        LandmarkKind.ARTICULATION_CENTER,
        _JOINT_CENTER,
        f"fhp21/v1:{name}",
    )


def _tip(name: str) -> LandmarkDefinition:
    return LandmarkDefinition(
        name,
        LandmarkKind.SURFACE_POINT,
        _TIP_SURFACE,
        f"fhp21/v1:{name}",
    )


FHP21 = LandmarkSchema(
    version="fhp21/v1",
    definitions=(
        LandmarkDefinition(
            "wrist_center",
            LandmarkKind.REFERENCE_POINT,
            "The fhp21/v1 canonical kinematic wrist reference at the proximal hand/wrist "
            "base. Its operational realization (for example a native keypoint, annotation "
            "rule, mesh regressor, or derived point) is declared by the mapping record; it "
            "does not claim a unique, directly observable radiocarpal centre.",
            "fhp21/v1:wrist_center",
        ),
        _joint("thumb_cmc"),
        _joint("thumb_mcp"),
        _joint("thumb_ip"),
        _tip("thumb_tip"),
        _joint("index_mcp"),
        _joint("index_pip"),
        _joint("index_dip"),
        _tip("index_tip"),
        _joint("middle_mcp"),
        _joint("middle_pip"),
        _joint("middle_dip"),
        _tip("middle_tip"),
        _joint("ring_mcp"),
        _joint("ring_pip"),
        _joint("ring_dip"),
        _tip("ring_tip"),
        _joint("little_mcp"),
        _joint("little_pip"),
        _joint("little_dip"),
        _tip("little_tip"),
    ),
    parents=(-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19),
    tip_indices=(4, 8, 12, 16, 20),
)


FHP21_IDENTITY_MAPPING = LandmarkMappingRecord(
    mapping_id="fhp21/v1:identity",
    source_joint_set_id=FHP21.version,
    target_schema_version=FHP21.version,
    source_landmark_names=FHP21.names,
    entries=tuple(
        LandmarkMappingEntry(
            target_index=index,
            quality=MappingQuality.EXACT,
            source_indices=(index,),
            source_construction_id=definition.construction_id,
            method="identity",
            source_definition=definition.definition,
        )
        for index, definition in enumerate(FHP21.definitions)
    ),
    provenance="Normative identity mapping defined by fisheye-handpose fhp21/v1.",
)
