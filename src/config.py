# ============================================================
# config.py - project-wide configuration
# ============================================================
import os
from pathlib import Path

# Path configuration. Set CVAD_BASE_DIR to override the project root.
BASE_DIR = Path(os.environ.get("CVAD_BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Flickr8k-CN paths.
IMG_DIR = DATA_DIR / "Flickr8k" / "train"

# Caption files.
ENC_CAPTION = DATA_DIR / "flickr8kenc.caption.txt"
ZHB_CAPTION = DATA_DIR / "flickr8kzhb.caption.txt"
ZHG_CAPTION = DATA_DIR / "flickr8kzhg.caption.txt"
ZHM_TEST_CAPTION = DATA_DIR / "flickr8kzhmtest.captions.txt"

# Dataset split files. Each row contains one image id.
TRAIN_SPLIT = DATA_DIR / "flickr8ktrain.txt"
TEST_SPLIT = DATA_DIR / "flickr8ktest.txt"
VAL_SPLIT = DATA_DIR / "flickr8kval.txt"

# Flickr30k-CN paths.
F30K_DIR = DATA_DIR / "Flickr30k"
F30K_IMG_DIR = F30K_DIR / "flickr30k-images" / "flickr30k-images"
F30K_ENC_CSV = F30K_DIR / "flickr_annotations_30k.csv"
_F30K_CN_TEST = F30K_DIR / "flickr30k-cn" / "flickr30kzhmbosontest"
F30K_ZHM_TEST_CAPTION = _F30K_CN_TEST / "TextData" / "seg.flickr30kzhmbosontest.caption.txt"
F30K_TEST_SPLIT = _F30K_CN_TEST / "ImageSets" / "flickr30kzhmbosontest.txt"

# Output directories.
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Model checkpoints.
PRIMARY_MODEL = "OFA-Sys/chinese-clip-vit-large-patch14"
SECONDARY_MODEL = "BAAI/AltCLIP"

# Experiment defaults.
DEVICE = "cuda"
IMG_SIZE = 224
N_PATCHES = 16
CAPTIONS_PER_IMAGE = 5
ACTIVATION_THRESHOLD = 0.5

# Visualization defaults.
CHINESE_FONT = "Microsoft YaHei"
FIGURE_DPI = 150
HEATMAP_ALPHA = 0.55
