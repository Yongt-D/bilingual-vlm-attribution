# ============================================================
# data_loader.py - dataset loading and preprocessing utilities
# ============================================================
"""
Utilities for reading Flickr8k-CN and Flickr30k-CN annotations,
normalizing line endings, aligning English and Chinese captions, and
constructing image-caption samples for the experiments.
"""

import ast
import csv
import logging
import re
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def parse_captions(filepath, lang_tag: str = None) -> dict:
    """
    Parse a Flickr8k-CN caption file.

    Each valid row has the form:
        image_id#language_tag#index caption_text

    Parameters
    ----------
    filepath:
        Caption file path.
    lang_tag:
        Optional language tag filter, such as "enc", "zhm", "zhb", or
        "zhg". If None, all language tags are retained.

    Returns
    -------
    dict
        Mapping from image id to a list of captions.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Caption file not found: {filepath}")

    raw = filepath.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.strip().split("\n")

    captions = {}
    skipped = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        sep_idx = line.find(" ")
        if sep_idx == -1:
            skipped += 1
            continue

        key_part = line[:sep_idx]
        caption_text = line[sep_idx + 1:].strip()
        parts = key_part.split("#")
        if len(parts) < 3:
            skipped += 1
            continue

        img_id = parts[0]
        file_lang = parts[1]
        if lang_tag is not None and file_lang != lang_tag:
            continue

        captions.setdefault(img_id, []).append(caption_text)

    if skipped:
        logger.warning("[parse_captions] skipped %d malformed rows in %s", skipped, filepath.name)

    logger.info(
        "[parse_captions] %s | lang=%s | images=%d | captions=%d",
        filepath.name,
        lang_tag,
        len(captions),
        sum(len(v) for v in captions.values()),
    )
    return captions


def load_split(filepath) -> list:
    """
    Read a train/test/validation split file and return image ids.

    Some split files contain suffixes such as "#enc#0"; these suffixes
    are removed. Missing ".jpg" suffixes are added.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Split file not found: {filepath}")

    raw = filepath.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

    img_ids = []
    for line in lines:
        img_id = line.split("#")[0].strip()
        if not img_id:
            continue
        if not img_id.lower().endswith(".jpg"):
            img_id = img_id + ".jpg"
        img_ids.append(img_id)

    seen = set()
    unique_ids = []
    for img_id in img_ids:
        if img_id not in seen:
            seen.add(img_id)
            unique_ids.append(img_id)

    logger.info("[load_split] %s -> %d images", filepath.name, len(unique_ids))
    return unique_ids


def build_aligned_samples(
    img_ids: list,
    en_captions: dict,
    zh_captions: dict,
    img_dir: Path,
    captions_per_image: int = 5,
) -> list:
    """
    Align image ids with English and Chinese captions.

    Samples with missing images or missing captions in either language are
    excluded. Each retained sample contains at most `captions_per_image`
    captions per language.
    """
    samples = []
    missing_img = 0
    missing_en = 0
    missing_zh = 0

    for img_id in img_ids:
        img_path = Path(img_dir) / img_id
        if not img_path.exists():
            missing_img += 1
            continue
        if img_id not in en_captions:
            missing_en += 1
            continue
        if img_id not in zh_captions:
            missing_zh += 1
            continue

        samples.append({
            "img_id": img_id,
            "img_path": img_path,
            "en_caps": en_captions[img_id][:captions_per_image],
            "zh_caps": zh_captions[img_id][:captions_per_image],
        })

    logger.info(
        "[build_aligned_samples] valid=%d | missing_img=%d | missing_en=%d | missing_zh=%d",
        len(samples),
        missing_img,
        missing_en,
        missing_zh,
    )
    return samples


def parse_flickr30k_en_captions(csv_path, split: str = None) -> dict:
    """
    Parse Flickr30k English captions from the annotation CSV.

    The `raw` column contains a Python-list-like string with five English
    captions. If `split` is provided, only rows from that split are used.
    """
    csv_path = Path(csv_path)
    captions = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            fname = row["filename"].strip()
            try:
                caps = ast.literal_eval(row["raw"])
            except Exception:
                continue
            captions[fname] = [str(c) for c in caps]

    logger.info(
        "[parse_flickr30k_en] %s | split=%s | images=%d | captions=%d",
        csv_path.name,
        split or "all",
        len(captions),
        sum(len(v) for v in captions.values()),
    )
    return captions


def parse_flickr30k_zh_captions(txt_path) -> dict:
    """
    Parse Flickr30k-CN Chinese captions from the segmented POS-tagged file.

    Each row contains tokens with POS tags. The POS suffix is removed and
    whitespace is normalized before model input.
    """
    txt_path = Path(txt_path)
    raw = txt_path.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.strip().split("\n")

    captions = {}
    pos_re = re.compile(r":[a-zA-Z]+\b")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        sep = line.find(" ")
        if sep == -1:
            continue

        key_part = line[:sep]
        tagged = line[sep + 1:]
        parts = key_part.split("#")
        if len(parts) < 2:
            continue

        img_id_raw = parts[0].strip()
        img_id = img_id_raw if img_id_raw.lower().endswith(".jpg") else img_id_raw + ".jpg"
        clean = pos_re.sub("", tagged)
        clean = re.sub(r"\s+", "", clean).strip()
        captions.setdefault(img_id, []).append(clean)

    logger.info(
        "[parse_flickr30k_zh] %s | images=%d | captions=%d",
        txt_path.name,
        len(captions),
        sum(len(v) for v in captions.values()),
    )
    return captions


def prepare_flickr30k_test_data(cfg) -> list:
    """Prepare aligned Flickr30k-CN test samples from a config module."""
    test_ids = load_split(cfg.F30K_TEST_SPLIT)
    en_caps = parse_flickr30k_en_captions(cfg.F30K_ENC_CSV, split=None)
    zh_caps = parse_flickr30k_zh_captions(cfg.F30K_ZHM_TEST_CAPTION)

    return build_aligned_samples(
        test_ids,
        en_caps,
        zh_caps,
        img_dir=cfg.F30K_IMG_DIR,
        captions_per_image=cfg.CAPTIONS_PER_IMAGE,
    )


def load_image(img_path) -> Image.Image:
    """Load an image as an RGB PIL image."""
    return Image.open(img_path).convert("RGB")


def prepare_test_data(cfg) -> list:
    """Prepare aligned Flickr8k-CN test samples from a config module."""
    test_ids = load_split(cfg.TEST_SPLIT)
    en_caps = parse_captions(cfg.ENC_CAPTION, lang_tag="enc")

    try:
        zh_caps = parse_captions(cfg.ZHM_TEST_CAPTION, lang_tag="zhm")
        if len(zh_caps) < 500:
            zh_caps_b = parse_captions(cfg.ZHB_CAPTION, lang_tag="zhb")
            for img_id, caps in zh_caps_b.items():
                if img_id not in zh_caps:
                    zh_caps[img_id] = caps
            logger.info("[prepare_test_data] supplemented zhmtest captions with zhb captions")
    except FileNotFoundError:
        logger.warning("[prepare_test_data] zhmtest file missing; falling back to zhb captions")
        zh_caps = parse_captions(cfg.ZHB_CAPTION, lang_tag="zhb")

    return build_aligned_samples(
        test_ids,
        en_caps,
        zh_caps,
        img_dir=cfg.IMG_DIR,
        captions_per_image=cfg.CAPTIONS_PER_IMAGE,
    )
