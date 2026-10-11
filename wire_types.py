"""Wire detection shared types and constants."""

from dataclasses import dataclass, field

import numpy as np

SORT_X = "x"
SORT_Y = "y"
SORT_Z = "z"
SORT_CUSTOM = "custom"
SORT_CHOICES = (SORT_X, SORT_Y, SORT_Z, SORT_CUSTOM)

WIRE_MODE_BIN_TOP = "bin_top"
WIRE_MODE_PLANE = "plane"
WIRE_MODE_EDGE = "edge"
WIRE_MODE_CHOICES = (WIRE_MODE_BIN_TOP, WIRE_MODE_PLANE, WIRE_MODE_EDGE)

REP_CENTER = "center"
REP_TOP = "top"
REP_BOTTOM = "bottom"
REP_LEFT = "left"
REP_RIGHT = "right"
REP_FIRST_SORT = "first_sort"
REP_CHOICES = (
    REP_CENTER,
    REP_TOP,
    REP_BOTTOM,
    REP_LEFT,
    REP_RIGHT,
    REP_FIRST_SORT,
)


@dataclass
class WireDetectionConfig:
    min_length_px: int = 80
    min_area_px: int = 120
    min_thickness_px: float = 2.0
    max_thickness_px: float = 80.0
    foreground_mode: str = WIRE_MODE_BIN_TOP
    top_band_mm: float = 18.0
    top_percentile: float = 1.0
    depth_min_mm: float = 70.0
    depth_max_mm: float = 0.0
    plane_threshold_mm: float = 6.0
    merge_distance_px: float = 18.0
    merge_angle_deg: float = 35.0
    sort_axis: str = SORT_X
    sort_desc: bool = False
    sort_vector_xyz: tuple = (1.0, 0.0, 0.0)
    sort_vector_uv: tuple = (1.0, 0.0)
    representative_mode: str = REP_CENTER
    tracking_enabled: bool = True
    max_tracking_distance_px: float = 80.0
    confidence_minimum: float = 0.45
    use_depth_foreground: bool = True
    use_edge_fallback: bool = True
    apply_morph_open: bool = True
    apply_morph_close: bool = True
    filter_components: bool = True
    skeleton_enabled: bool = True
    save_debug_images: bool = False
    max_wires: int = 50
    debug_enabled: bool = True


@dataclass
class WireCandidate:
    id: int
    points: np.ndarray
    representative_uv: tuple
    representative_xyz: tuple = None
    depth_mm: float = None
    layer_index: int = None
    is_pick_candidate: bool = True
    pick_score: float = 0.0
    length_px: float = 0.0
    thickness_px: float = 0.0
    confidence: float = 0.0
    confidence_parts: dict = field(default_factory=dict)
    sort_key: float = 0.0
    segment_count: int = 1
    tracking_id: int = None


@dataclass
class WireDetectionResult:
    wires: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    binary: np.ndarray = None
    debug_frames: list = field(default_factory=list)
    status: str = "ok"
    note: str = ""
    debug_path: str = None


@dataclass
class WireSegment:
    id: int
    points: np.ndarray
    endpoints: tuple
    directions: tuple
    length_px: float
    thickness_px: float
    depth_mm: float = None
