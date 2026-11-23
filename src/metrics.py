# ============================================================
# metrics.py - quantitative attention-divergence metrics
# ============================================================
"""
Metric functions for comparing English-conditioned and Chinese-conditioned
attention heatmaps.
"""

import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy as scipy_entropy


def _to_prob(heatmap: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Convert a heatmap to a flattened probability distribution."""
    h = heatmap.astype(np.float64).flatten()
    h = h - h.min()
    h = h + eps
    return h / h.sum()


def _binarize(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a heatmap after min-max normalization."""
    h = heatmap.astype(np.float64)
    mn, mx = h.min(), h.max()
    if mx > mn:
        h = (h - mn) / (mx - mn)
    return (h >= threshold).astype(np.float32)


def compute_entropy(heatmap: np.ndarray) -> float:
    """Compute Shannon entropy in nats."""
    return float(scipy_entropy(_to_prob(heatmap)))


def compute_area_ratio(heatmap: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the fraction of patches above the activation threshold."""
    mask = _binarize(heatmap, threshold)
    return float(mask.sum() / mask.size)


def compute_iou(
    heatmap_en: np.ndarray,
    heatmap_zh: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection over Union between thresholded heatmaps."""
    mask_en = _binarize(heatmap_en, threshold)
    mask_zh = _binarize(heatmap_zh, threshold)
    intersection = float((mask_en * mask_zh).sum())
    union = float(np.clip(mask_en + mask_zh, 0, 1).sum())
    if union == 0:
        return 0.0
    return intersection / union


def compute_kl_divergence(
    heatmap_en: np.ndarray,
    heatmap_zh: np.ndarray,
) -> float:
    """Compute KL(P_en || P_zh) between heatmap distributions."""
    p_en = _to_prob(heatmap_en)
    p_zh = _to_prob(heatmap_zh)
    return float(np.sum(rel_entr(p_en, p_zh)))


def compute_symmetric_kl(
    heatmap_en: np.ndarray,
    heatmap_zh: np.ndarray,
) -> float:
    """Compute symmetric KL divergence."""
    return compute_kl_divergence(heatmap_en, heatmap_zh) + compute_kl_divergence(heatmap_zh, heatmap_en)


def compute_all_metrics(
    heatmap_en: np.ndarray,
    heatmap_zh: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all component metrics for one English/Chinese heatmap pair."""
    ent_en = compute_entropy(heatmap_en)
    ent_zh = compute_entropy(heatmap_zh)
    area_en = compute_area_ratio(heatmap_en, threshold)
    area_zh = compute_area_ratio(heatmap_zh, threshold)

    return {
        "entropy_en": round(ent_en, 6),
        "entropy_zh": round(ent_zh, 6),
        "delta_entropy": round(ent_zh - ent_en, 6),
        "area_en": round(area_en, 6),
        "area_zh": round(area_zh, 6),
        "delta_area": round(area_zh - area_en, 6),
        "iou": round(compute_iou(heatmap_en, heatmap_zh, threshold), 6),
        "kl_en_to_zh": round(compute_kl_divergence(heatmap_en, heatmap_zh), 6),
        "kl_zh_to_en": round(compute_kl_divergence(heatmap_zh, heatmap_en), 6),
        "kl_symmetric": round(compute_symmetric_kl(heatmap_en, heatmap_zh), 6),
    }


def summarize_results(df) -> str:
    """Create a compact text summary for a metrics DataFrame."""
    cols_of_interest = [
        "entropy_en",
        "entropy_zh",
        "delta_entropy",
        "area_en",
        "area_zh",
        "delta_area",
        "iou",
        "kl_symmetric",
    ]
    cols = [c for c in cols_of_interest if c in df.columns]

    lines = ["=" * 60, "  Experiment summary", "=" * 60]
    lines.append(f"  Valid samples: {len(df)}")
    lines.append("-" * 60)

    stat = df[cols].agg(["mean", "std", "median"])
    for col in cols:
        mean = stat.loc["mean", col]
        std = stat.loc["std", col]
        median = stat.loc["median", col]
        lines.append(f"  {col:<22s}  mean={mean:+.4f}  std={std:.4f}  median={median:+.4f}")

    if "delta_entropy" in df.columns:
        from scipy import stats as sp_stats

        t_stat, p_val = sp_stats.ttest_1samp(df["delta_entropy"].dropna(), popmean=0)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        lines.append("-" * 60)
        lines.append(f"  Delta-entropy one-sample test: t={t_stat:.3f}  p={p_val:.4f}  {sig}")
        mean_delta = df["delta_entropy"].mean()
        direction = "ZH heatmaps are more dispersed" if mean_delta > 0 else "EN heatmaps are more dispersed"
        lines.append(f"  Direction: delta_entropy={mean_delta:+.4f}; {direction}")

    lines.append("=" * 60)
    return "\n".join(lines)
