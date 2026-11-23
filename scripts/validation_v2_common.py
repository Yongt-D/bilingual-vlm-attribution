"""Shared, version-pinned utilities for MVA v2 validation inference.

Isolated from v1 (submission_mva/validation/). Never overwrites v1 files.
Supports single-vision-forward multi-method attribution extraction:
  1. patch_gradcam      : Layer-0 patch_embedding Grad-CAM (v1 baseline)
  2. lastblock_gradcam  : Final Transformer block layer_norm1 Grad-CAM
  3. chefer_relevance   : All-layer gradient-attention relevance (Chefer et al., ICCV 2021)
"""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
DEST_V1 = BASE / 'submission_mva' / 'validation'
DEST_V2 = BASE / 'submission_mva' / 'validation_v2'
sys.path.insert(0, str(BASE))
from src import config as cfg
from src.data_loader import prepare_test_data, prepare_flickr30k_test_data

SEED = 20261115
FRACTIONS = np.array([0., 26 / 256, 51 / 256, 77 / 256, 102 / 256, 128 / 256], dtype=np.float64)
MASK_COUNTS = (26, 51, 77, 102, 128)
REPEATS = 5
METHODS = ('patch_gradcam', 'lastblock_gradcam', 'chefer_relevance')
NEW_METHODS = ('lastblock_gradcam', 'chefer_relevance')

MODEL_REPOS = {
    'ChineseCLIP': 'OFA-Sys/chinese-clip-vit-large-patch14',
    'AltCLIP': 'BAAI/AltCLIP',
}


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def save_npz_atomic(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def dataset_f8k() -> list:
    samples = sorted(prepare_test_data(cfg), key=lambda s: s['img_id'])
    assert len(samples) == 1000, len(samples)
    assert len(set(s['img_id'] for s in samples)) == 1000
    assert all(len(s[k]) == 5 for s in samples for k in ('en_caps', 'zh_caps'))
    return samples


def dataset_f30k() -> list:
    samples = sorted(prepare_flickr30k_test_data(cfg), key=lambda s: s['img_id'])
    assert len(samples) == 975, len(samples)
    assert len(set(s['img_id'] for s in samples)) == 975
    assert all(len(s[k]) == 5 for s in samples for k in ('en_caps', 'zh_caps'))
    return samples


def selection_f8k(samples: list) -> np.ndarray:
    """Identical 200-image occlusion sample to v1."""
    return np.random.default_rng(SEED).permutation(len(samples))[:200]


def load_model(model_key: str, device: str):
    """Load frozen FP32 eager-attention model with parameters requiring no grad."""
    assert model_key in MODEL_REPOS, model_key
    assert transformers.__version__ == '4.49.0', transformers.__version__
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    repo = MODEL_REPOS[model_key]
    snap_dir = BASE / 'models' / ('models--' + repo.replace('/', '--')) / 'snapshots'
    snaps = [p for p in snap_dir.iterdir() if (p / 'config.json').exists()]
    assert len(snaps) == 1, snaps
    path = snaps[0]

    if model_key == 'ChineseCLIP':
        Model, Processor = transformers.ChineseCLIPModel, transformers.ChineseCLIPProcessor
    else:
        Model, Processor = transformers.AltCLIPModel, transformers.AltCLIPProcessor

    processor = Processor.from_pretrained(str(path), local_files_only=True, use_fast=False)
    model, info = Model.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation='eager',
        output_loading_info=True,
    )
    assert not info.get('missing_keys'), info
    assert not info.get('mismatched_keys'), info
    assert not info.get('error_msgs'), info

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = dict(
        model_key=model_key,
        model=repo,
        revision=path.name,
        device=device,
        torch=torch.__version__,
        transformers=transformers.__version__,
        tokenizer=type(processor.tokenizer).__name__,
        dtype='float32',
        attention='eager',
        max_length=77,
        loading_info=info,
    )
    return model, processor, path, meta


def pixels(processor, sample) -> torch.Tensor:
    with Image.open(sample['img_path']) as img:
        x = processor.image_processor(images=img.convert('RGB'), return_tensors='pt')['pixel_values']
    assert x.shape == (1, 3, 224, 224) and torch.isfinite(x).all()
    return x


def text_features(model, processor, texts: list, device: str) -> torch.Tensor:
    inputs = processor.tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77,
    )
    features = model.get_text_features(
        input_ids=inputs['input_ids'].to(device),
        attention_mask=inputs['attention_mask'].to(device),
    )
    return F.normalize(features.float(), dim=-1)


def image_features(model, x: torch.Tensor, device: str) -> torch.Tensor:
    return F.normalize(model.get_image_features(pixel_values=x.to(device)).float(), dim=-1)


def normalize_map(arr: np.ndarray):
    """Validate and min-max normalize a 2D map to [0, 1]. Returns (map, ok, err)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (16, 16):
        return np.full((16, 16), np.nan, dtype=np.float32), False, f'bad_shape:{arr.shape}'
    if not np.isfinite(arr).all():
        return np.full((16, 16), np.nan, dtype=np.float32), False, 'non_finite'
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.full((16, 16), np.nan, dtype=np.float32), False, 'constant_map'
    return ((arr - mn) / (mx - mn)).astype(np.float32), True, ''


def extract_image_attributions(
    model,
    processor,
    pixel_values: torch.Tensor,
    en_caps: list,
    zh_caps: list,
    device: str,
) -> dict:
    """Extract all 3 attribution methods for 5 EN + 5 ZH captions in one vision forward.

    Mathematical invariance:
      - Text features f_t(C) are independent of visual activations, so computing them
        under torch.no_grad() produces identical partial derivatives d s / d F_v.
      - Freezing model parameters (requires_grad=False) with pixel_values.requires_grad=True
        computes exact activation gradients via torch.autograd.grad without allocating
        parameter gradients.
    """
    assert len(en_caps) == 5 and len(zh_caps) == 5
    all_caps = list(en_caps) + list(zh_caps)

    with torch.no_grad():
        # Match legacy single-caption tokenization for exact zero-padding parity
        # with src/heatmap.py (also identical to batched features within FP32 eps).
        txt_list = []
        for cap in all_caps:
            tok = processor.tokenizer(
                [cap],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77,
            )
            tf = model.get_text_features(
                input_ids=tok['input_ids'].to(device),
                attention_mask=tok['attention_mask'].to(device),
            )
            txt_list.append(F.normalize(tf.float(), dim=-1))
        txt_feats = torch.cat(txt_list, dim=0)  # (10, D)

    acts = {}

    def hook_patch(_module, _inp, output):
        acts['patch'] = output

    def hook_last_norm(_module, _inp, output):
        acts['last_norm'] = output

    patch_mod = model.vision_model.embeddings.patch_embedding
    last_norm_mod = model.vision_model.encoder.layers[-1].layer_norm1
    h1 = patch_mod.register_forward_hook(hook_patch)
    h2 = last_norm_mod.register_forward_hook(hook_last_norm)

    pv = pixel_values.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
    try:
        with torch.enable_grad():
            vis_out = model.vision_model(pixel_values=pv, output_attentions=True)
            img_feat = F.normalize(model.visual_projection(vis_out.pooler_output).float(), dim=-1)
    finally:
        h1.remove()
        h2.remove()

    patch_act = acts['patch']          # (1, C, 16, 16)
    last_norm_act = acts['last_norm']  # (1, 257, C)
    attentions = vis_out.attentions    # tuple of L tensors: (1, H, 257, 257)
    assert patch_act.shape[2:] == (16, 16), patch_act.shape
    assert last_norm_act.shape[1] == 257, last_norm_act.shape
    assert attentions is not None and len(attentions) > 0

    targets = [patch_act, last_norm_act, *attentions]
    patch_det = patch_act.detach()
    last_spatial_det = last_norm_act.detach()[:, 1:, :]  # (1, 256, C)
    attn_dets = [a.detach() for a in attentions]
    num_tokens = attentions[0].shape[-1]
    eye = torch.eye(num_tokens, dtype=torch.float32, device=device)

    per_cap_maps = {m: [[], []] for m in METHODS}
    caption_ok = np.zeros((len(METHODS), 2, 5), dtype=bool)
    failures = []

    for idx in range(len(all_caps)):
        lang = 0 if idx < 5 else 1
        cap_i = idx if idx < 5 else idx - 5
        score = (img_feat * txt_feats[idx:idx + 1]).sum()
        grads = torch.autograd.grad(
            score,
            targets,
            retain_graph=(idx < len(all_caps) - 1),
            allow_unused=False,
        )

        # 1. Layer-0 patch_embedding Grad-CAM (exact v1 rule)
        g_patch = grads[0].detach()
        w_patch = g_patch.mean(dim=(2, 3), keepdim=True)
        raw_patch = F.relu((w_patch * patch_det).sum(dim=1)).squeeze(0).cpu().numpy()

        # 2. Final Transformer block layer_norm1 Grad-CAM (spatial tokens 1..256)
        g_last = grads[1].detach()[:, 1:, :]
        w_last = g_last.mean(dim=1, keepdim=True)
        raw_last = F.relu((w_last * last_spatial_det).sum(dim=-1)).reshape(16, 16).cpu().numpy()

        # 3. Chefer et al. (ICCV 2021, Eq. 5-7) all-layer gradient-attention relevance
        R = eye.clone()
        for a_det, g_a in zip(attn_dets, grads[2:]):
            cam_a = (g_a.detach() * a_det).clamp(min=0).mean(dim=1).squeeze(0)
            R = R + torch.matmul(cam_a, R)
        raw_chefer = R[0, 1:].reshape(16, 16).cpu().numpy()

        for mi, (m_name, raw) in enumerate(zip(METHODS, (raw_patch, raw_last, raw_chefer))):
            norm_m, ok, err = normalize_map(raw)
            caption_ok[mi, lang, cap_i] = ok
            if ok:
                per_cap_maps[m_name][lang].append(norm_m)
            else:
                failures.append(dict(method=m_name, language=('EN', 'ZH')[lang], caption=cap_i, error=err))

    lang_maps = {}
    lang_complete = np.zeros((len(METHODS), 2), dtype=bool)
    for mi, m_name in enumerate(METHODS):
        pair = []
        for lang in range(2):
            if len(per_cap_maps[m_name][lang]) == 5:
                avg_raw = np.mean(np.stack(per_cap_maps[m_name][lang], axis=0), axis=0)
                avg_norm, ok, err = normalize_map(avg_raw)
                lang_complete[mi, lang] = ok
                pair.append(avg_norm)
                if not ok:
                    failures.append(dict(method=m_name, language=('EN', 'ZH')[lang], caption=-1, error=err))
            else:
                lang_complete[mi, lang] = False
                pair.append(np.full((16, 16), np.nan, dtype=np.float32))
        lang_maps[m_name] = np.stack(pair, axis=0)  # (2, 16, 16)

    return dict(
        maps=lang_maps,
        caption_ok=caption_ok,
        lang_complete=lang_complete,
        failures=failures,
        img_feat=img_feat.detach().cpu().numpy()[0],
        text_feat=txt_feats.detach().cpu().numpy().reshape(2, 5, -1),
    )


def patch_order(heatmap: np.ndarray, seed: int, high: bool = True) -> np.ndarray:
    a = np.asarray(heatmap).ravel()
    assert len(a) == 256 and np.isfinite(a).all() and np.ptp(a) > 0
    tie = np.random.default_rng(seed).random(256)
    return np.lexsort((tie, -a if high else a))


def mask_pixels(x: torch.Tensor, order: np.ndarray, k: int) -> torch.Tensor:
    assert x.shape == (1, 3, 224, 224)
    mask = torch.zeros(256, dtype=torch.bool)
    mask[np.asarray(order[:k]).copy()] = True
    mask = mask.reshape(16, 16).repeat_interleave(14, 0).repeat_interleave(14, 1)
    return x.masked_fill(mask[None, None], 0.)


def ranks(similarity: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """One-based ranks; exact ties conservatively receive the worst tied rank."""
    score = similarity[np.arange(len(correct)), correct]
    return (similarity >= score[:, None]).sum(1)


def outcome_arrays(image: np.ndarray, text: np.ndarray):
    """T2I ranks (N, 2, 5) and I2T best-positive ranks (N, 2)."""
    n = len(image)
    assert text.shape[:3] == (n, 2, 5)
    sim = text.reshape(n * 10, -1).astype(np.float64) @ image.astype(np.float64).T
    t2i = ranks(sim, np.repeat(np.arange(n), 10)).reshape(n, 2, 5)
    i2t = []
    for lang in range(2):
        matrix = image.astype(np.float64) @ text[:, lang].reshape(n * 5, -1).astype(np.float64).T
        best = np.stack([matrix[i, i * 5:(i + 1) * 5].max() for i in range(n)])
        competitors = matrix >= best[:, None]
        for i in range(n):
            competitors[i, i * 5:(i + 1) * 5] = False
        i2t.append(1 + competitors.sum(1))
    return sim, t2i, np.stack(i2t, axis=1)
