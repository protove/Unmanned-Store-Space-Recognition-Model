"""Central configuration for Space Detection pipeline."""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Override these via environment variables for portability
INPUT_IMAGE_DIR = BASE_DIR / "assets" / "images"
DEPTH_MAP_DIR = BASE_DIR / "assets" / "depth_maps"
OUTPUT_DIR = BASE_DIR / "output"

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
