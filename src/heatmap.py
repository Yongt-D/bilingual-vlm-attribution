# ============================================================
# heatmap.py - visual attention heatmap extraction
# ============================================================
"""
This module provides two complementary attention-map extraction methods:

1. Text-conditioned GradCAM, which uses the gradient of the image-text
   similarity score with respect to the ViT patch-embedding activations.
2. Attention Rollout, which propagates visual self-attention through the
   ViT layers and is used as a language-free baseline.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_gradcam_heatmap(
    model,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute a text-conditioned GradCAM heatmap.

    Returns a normalized two-dimensional NumPy array whose shape matches
    the ViT patch grid.
    """
    model.eval()
    pixel_values = pixel_values.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    activation = {}

    def fwd_hook(module, inp, output):
        activation["patch_embed"] = output
        output.retain_grad()

    patch_conv = model.vision_model.embeddings.patch_embedding
    hook_handle = patch_conv.register_forward_hook(fwd_hook)

    try:
        vis_out = model.vision_model(pixel_values=pixel_values)
        img_feat = model.visual_projection(vis_out.pooler_output)
        txt_feat = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        score = (img_feat * txt_feat).sum()

        model.zero_grad()
        score.backward()
    finally:
        hook_handle.remove()

    act = activation.get("patch_embed")
    grad = activation["patch_embed"].grad if "patch_embed" in activation else None

    if act is None or grad is None:
        raise RuntimeError("GradCAM activation or gradient is missing; no map was produced")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=False)
    cam = F.relu(cam)
    cam = cam.squeeze(0).detach().cpu().numpy()
    if not np.isfinite(cam).all() or np.ptp(cam) == 0:
        raise ValueError("GradCAM is non-finite or constant; exclude and record this caption explicitly")
    return _normalize(cam)


@torch.no_grad()
def get_attention_rollout(
    model,
    pixel_values: torch.Tensor,
    device: str = "cuda",
    head_fusion: str = "mean",
) -> np.ndarray:
    """
    Compute a ViT Attention Rollout heatmap that does not condition on text.
    """
    model.eval()
    pixel_values = pixel_values.to(device)

    vis_out = model.vision_model(pixel_values=pixel_values, output_attentions=True)
    attentions = vis_out.attentions

    batch_size = pixel_values.shape[0]
    num_tokens = attentions[0].shape[-1]
    dev = pixel_values.device

    def fuse_heads(attn):
        if head_fusion == "mean":
            return attn.mean(dim=1)
        if head_fusion == "max":
            return attn.max(dim=1)[0]
        if head_fusion == "min":
            return attn.min(dim=1)[0]
        raise ValueError(f"Unsupported head_fusion: {head_fusion}")

    attn_mats = [fuse_heads(a) for a in attentions]
    identity = torch.eye(num_tokens, device=dev).unsqueeze(0).expand(batch_size, -1, -1)

    rollout = identity.clone()
    for attn in attn_mats:
        attn = attn + identity
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(attn, rollout)

    cls_attn = rollout[:, 0, 1:]
    grid_size = int(cls_attn.shape[-1] ** 0.5)
    heatmap = cls_attn.reshape(batch_size, grid_size, grid_size)[0].cpu().numpy()
    return _normalize(heatmap)


@torch.enable_grad()
def get_averaged_gradcam(
    model,
    processor,
    image_pil,
    captions: list,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute GradCAM for multiple captions of the same image and average the
    resulting heatmaps. All supplied captions must succeed; failures are
    propagated so the caller can record an excluded image rather than use
    an undocumented partial average.
    """
    if not captions:
        raise ValueError("At least one caption is required for averaged GradCAM")
    heatmaps = []
    for caption in captions:
        try:
            inputs = processor(
                text=[caption],
                images=[image_pil],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            heatmap = get_gradcam_heatmap(
                model,
                inputs["pixel_values"],
                inputs["input_ids"],
                inputs["attention_mask"],
                device=device,
            )
            heatmaps.append(heatmap)
        except Exception as exc:
            raise RuntimeError(
                f"GradCAM failed at caption {len(heatmaps) + 1}/{len(captions)}; "
                "no partial language average was returned"
            ) from exc

    avg = np.mean(np.stack(heatmaps, axis=0), axis=0)
    if not np.isfinite(avg).all() or np.ptp(avg) == 0:
        raise ValueError("Averaged GradCAM is non-finite or constant; no valid language map was produced")
    return _normalize(avg)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to the [0, 1] range."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)
