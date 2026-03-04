"""Central configuration for Space Detection pipeline."""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Default paths (relative to the lang-segment-anything root)
_LANG_ROOT = PROJECT_ROOT / "lang-segment-anything"
_DEFAULT_INPUT  = _LANG_ROOT / "assets" / "space_data"
_DEFAULT_DEPTH  = _DEFAULT_INPUT / "depth_map"
_DEFAULT_OUTPUT = _LANG_ROOT / "output" / "final"

# All three can be overridden via environment variables (used by Docker pipeline)
INPUT_IMAGE_DIR = Path(os.environ.get("INPUT_IMAGE_DIR", str(_DEFAULT_INPUT)))
DEPTH_MAP_DIR   = Path(os.environ.get("DEPTH_MAP_DIR",   str(_DEFAULT_DEPTH)))
OUTPUT_DIR      = Path(os.environ.get("OUTPUT_DIR",      str(_DEFAULT_OUTPUT)))

# Number of test images to process (test1.png … testN.png)
SPACE_IMAGE_COUNT = int(os.environ.get("SPACE_IMAGE_COUNT", "3"))

# ─── Detection Thresholds ─────────────────────────────────────────────────────
MASK_SIZE_THRESHOLD = 5000       # Minimum mask area in pixels
MIN_COMPONENT_SIZE = 150         # Minimum connected component size
VARIANCE_THRESHOLD = 95000       # Max spatial variance for valid masks
DISTANCE_THRESHOLD = 30          # Min pixel distance between masks

# ─── Clustering ───────────────────────────────────────────────────────────────
OVERLAP_CLUSTER_THRESHOLD = 0.9  # IoU threshold for merging masks

# ─── LangSAM Model ────────────────────────────────────────────────────────────
SAM_TYPE = "sam2.1_hiera_small"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# ─── DepthPro ─────────────────────────────────────────────────────────────────
DEPTH_PRO_CHECKPOINT = PROJECT_ROOT / "depth_pro" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
