# ============================================================
# model_loader.py - model loading utilities
# ============================================================
"""
Load Chinese-CLIP, AltCLIP, and OpenAI CLIP checkpoints through a
consistent interface.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str = "cuda", cache_dir=None):
    """
    Load a supported model and its processor.

    Parameters
    ----------
    model_name:
        Hugging Face model id.
    device:
        Target device, usually "cuda" or "cpu".
    cache_dir:
        Optional Hugging Face cache directory.
    """
    name_lower = model_name.lower()

    if "chinese-clip" in name_lower or "chinese_clip" in name_lower:
        return _load_chinese_clip(model_name, device, cache_dir)
    if "altclip" in name_lower:
        return _load_altclip(model_name, device, cache_dir)
    if "clip-vit" in name_lower or name_lower.startswith("openai/clip"):
        return _load_openai_clip(model_name, device, cache_dir)

    raise ValueError(
        f"Unsupported model: {model_name}. Supported examples include "
        "OFA-Sys/chinese-clip-vit-large-patch14, BAAI/AltCLIP, and OpenAI CLIP variants."
    )


def _load_chinese_clip(model_name: str, device: str, cache_dir):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    logger.info("[model_loader] loading Chinese-CLIP: %s", model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = ChineseCLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device).eval()

    assert hasattr(model, "vision_model"), "vision_model submodule is missing"
    assert hasattr(model.vision_model, "embeddings"), "vision embeddings are missing"
    assert hasattr(model.vision_model.embeddings, "patch_embedding"), (
        "patch_embedding layer is missing; GradCAM cannot be computed"
    )

    _log_model_info(model, device)
    return model, processor


def _load_altclip(model_name: str, device: str, cache_dir):
    from transformers import AltCLIPModel, AltCLIPProcessor

    logger.info("[model_loader] loading AltCLIP: %s", model_name)
    processor = AltCLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AltCLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device).eval()

    _log_model_info(model, device)
    return model, processor


def _load_openai_clip(model_name: str, device: str, cache_dir):
    from transformers import CLIPModel, CLIPProcessor

    logger.info("[model_loader] loading OpenAI CLIP: %s", model_name)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device).eval()

    _log_model_info(model, device)
    return model, processor


def _log_model_info(model, device):
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available() and device == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(
            "[model_loader] parameters=%.1fM (trainable=%.1fM) | GPU memory %.1f/%.1f GB",
            total_params,
            trainable,
            mem_alloc,
            mem_total,
        )
    else:
        logger.info("[model_loader] parameters=%.1fM | device=%s", total_params, device)


def get_patch_embedding_layer(model):
    """Return the ViT patch-embedding layer used for GradCAM hooks."""
    return model.vision_model.embeddings.patch_embedding


def get_vision_encoder_layers(model):
    """Return the ViT transformer layer list used for Attention Rollout."""
    return model.vision_model.encoder.layers


def get_image_features(model, pixel_values):
    """Return normalized image features through the model's public API."""
    return model.get_image_features(pixel_values=pixel_values)


def get_text_features(model, input_ids, attention_mask):
    """Return normalized text features through the model's public API."""
    return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
