# ============================================================
# visualize.py - plotting utilities
# ============================================================
"""
Visualization helpers for heatmap overlays and aggregate metric figures.
Figures that reproduce third-party dataset photographs should not be
included in the manuscript unless publication permission is available.
"""

import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _setup_cjk_font():
    """Set a CJK-capable font for optional Chinese caption rendering."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "STSong",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            logger.debug("[visualize] using CJK-capable font: %s", font)
            return font

    logger.warning("[visualize] no CJK-capable font found; Chinese captions may not render correctly")
    return None


_FONT = _setup_cjk_font()


def resize_heatmap(heatmap: np.ndarray, size: tuple) -> np.ndarray:
    """Resize a heatmap to (height, width) using bicubic interpolation."""
    return cv2.resize(
        heatmap.astype(np.float32),
        (size[1], size[0]),
        interpolation=cv2.INTER_CUBIC,
    )


def overlay_heatmap(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
    target_size: int = 336,
) -> np.ndarray:
    """Overlay a normalized heatmap on an RGB image."""
    img = np.array(image_pil.resize((target_size, target_size))).astype(np.float32)
    hm_resized = resize_heatmap(heatmap, (target_size, target_size))
    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, colormap)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (1.0 - alpha) * img + alpha * hm_colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def diff_map(heatmap_en: np.ndarray, heatmap_zh: np.ndarray) -> np.ndarray:
    """Compute an English-minus-Chinese difference map."""
    h, w = heatmap_en.shape
    zh_resized = resize_heatmap(heatmap_zh, (h, w))
    return heatmap_en.astype(np.float64) - zh_resized.astype(np.float64)


def plot_single_comparison(
    image_pil: Image.Image,
    heatmap_en: np.ndarray,
    heatmap_zh: np.ndarray,
    caption_en: str,
    caption_zh: str,
    metrics: dict,
    save_path=None,
    dpi: int = 150,
    method_label: str = "GradCAM",
) -> plt.Figure:
    """
    Create a four-panel diagnostic figure for local inspection.

    This function can reproduce third-party dataset photographs. Such
    outputs should be used for private quality control unless the dataset
    license and subject permissions allow publication.
    """
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        hspace=0.35,
        wspace=0.28,
        height_ratios=[3, 1.2],
    )

    target_size = 336
    img_np = np.array(image_pil.resize((target_size, target_size)))
    overlay_en = overlay_heatmap(image_pil, heatmap_en)
    overlay_zh = overlay_heatmap(image_pil, heatmap_zh)
    diff = diff_map(
        resize_heatmap(heatmap_en, (target_size, target_size)),
        resize_heatmap(heatmap_zh, (target_size, target_size)),
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    panels = [
        (img_np, "Original Image"),
        (overlay_en, f"English Attention\n({method_label})"),
        (overlay_zh, f"Chinese Attention\n({method_label})"),
    ]
    for ax, (img, title) in zip(axes[:3], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    axes[3].set_title("EN - ZH Difference\n(Red=EN focus, Blue=ZH focus)", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label="Delta attention")

    ax_caps = fig.add_subplot(gs[1, :2])
    ax_caps.axis("off")
    ax_caps.text(
        0.02,
        0.95,
        f"EN: {caption_en}\n\nZH: {caption_zh}",
        transform=ax_caps.transAxes,
        fontsize=9,
        va="top",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", alpha=0.9),
    )

    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.axis("off")
    text = (
        f"Entropy   EN={metrics.get('entropy_en', 0):.4f}   ZH={metrics.get('entropy_zh', 0):.4f}"
        f"   Delta={metrics.get('delta_entropy', 0):+.4f}\n"
        f"Area      EN={metrics.get('area_en', 0):.2%}      ZH={metrics.get('area_zh', 0):.2%}"
        f"      Delta={metrics.get('delta_area', 0):+.2%}\n"
        f"IoU (EN/ZH):  {metrics.get('iou', 0):.4f}\n"
        f"KL (EN->ZH):  {metrics.get('kl_en_to_zh', 0):.4f}   "
        f"KL (sym):  {metrics.get('kl_symmetric', 0):.4f}"
    )
    ax_metrics.text(
        0.02,
        0.95,
        text,
        transform=ax_metrics.transAxes,
        fontsize=9.5,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", alpha=0.95),
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metrics_summary(df, save_dir: Path, dpi: int = 150):
    """Generate aggregate metric figures for the full dataset."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [
        ("entropy_en", "entropy_zh", "Shannon Entropy (nats)", "Attention Entropy: EN vs ZH"),
        ("area_en", "area_zh", "High-Activation Area Ratio", "Activation Area: EN vs ZH"),
    ]
    for ax, (col_en, col_zh, ylabel, title) in zip(axes, specs):
        means = [df[col_en].mean(), df[col_zh].mean()]
        stds = [df[col_en].std(), df[col_zh].std()]
        bars = ax.bar(["English", "Chinese"], means, yerr=stds, color=["#1565C0", "#B71C1C"], capsize=6, width=0.4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.002, f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    fig1.suptitle("Bilingual Visual Attention: English vs Chinese (Mean +/- Std)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(save_dir / "fig1_en_zh_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    specs2 = [
        ("iou", "#2E7D32", "IoU (EN/ZH)"),
        ("kl_symmetric", "#6A1B9A", "Symmetric KL Divergence"),
    ]
    for ax, (col, color, xlabel) in zip(axes2, specs2):
        ax.hist(df[col].dropna(), bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(df[col].mean(), color="red", ls="--", lw=2, label=f"Mean={df[col].mean():.3f}")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {xlabel}", fontsize=12)
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(save_dir / "fig2_iou_kl_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    if "delta_entropy" in df.columns and "kl_symmetric" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        sc = ax3.scatter(df["delta_entropy"], df["kl_symmetric"], c=df["iou"], cmap="viridis_r", alpha=0.5, s=12)
        ax3.axvline(0, color="gray", ls="--", lw=1)
        ax3.axhline(df["kl_symmetric"].mean(), color="red", ls="--", lw=1)
        ax3.set_xlabel("Delta Entropy (ZH - EN)", fontsize=11)
        ax3.set_ylabel("Symmetric KL Divergence", fontsize=11)
        ax3.set_title("Attention Diversity vs Distribution Difference", fontsize=12, fontweight="bold")
        plt.colorbar(sc, ax=ax3, label="IoU")
        fig3.tight_layout()
        fig3.savefig(save_dir / "fig3_scatter_entropy_kl.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig3)

    logger.info("[visualize] summary figures saved to %s", save_dir)
