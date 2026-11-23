"""Resumable MVA v2 inference runner (ROI #1 + ROI #2).

Run with .venv_mva/Scripts/python.exe:
  --mode prepare
  --mode smoke --device cpu
  --mode full --device cuda --batch 4

Isolated under submission_mva/validation_v2/. Never modifies v1 outputs.
"""
import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from validation_v2_common import (
    BASE,
    DEST_V1,
    DEST_V2,
    FRACTIONS,
    MASK_COUNTS,
    METHODS,
    NEW_METHODS,
    REPEATS,
    SEED,
    dataset_f8k,
    dataset_f30k,
    extract_image_attributions,
    image_features,
    load_model,
    mask_pixels,
    outcome_arrays,
    patch_order,
    pixels,
    save_json,
    save_npz_atomic,
    selection_f8k,
    sha,
    text_features,
)


def archived_map_pair(source: str, sample: dict) -> np.ndarray:
    folder = BASE / ('outputs/heatmaps' if source.startswith('Chinese') else 'outputs/altclip/heatmaps')
    stem = Path(sample['img_id']).stem
    arr = np.stack([np.load(folder / f'{stem}_{lang}.npy') for lang in ('en', 'zh')])
    assert arr.shape == (2, 16, 16) and np.isfinite(arr).all()
    assert all(np.ptp(a) > 0 for a in arr)
    return arr


def prepare():
    f8k = dataset_f8k()
    f30k = dataset_f30k()
    selected = selection_f8k(f8k)

    # Verify exact identity with v1 F8k manifest and existing F30k Chinese-CLIP IDs.
    v1_manifest = json.loads((DEST_V1 / 'input_manifest.json').read_text(encoding='utf-8'))
    assert v1_manifest['selected_indices'] == selected.tolist(), 'F8k occlusion selection mismatch with v1'
    assert [s['img_id'] for s in v1_manifest['samples']] == [s['img_id'] for s in f8k], 'F8k sample ID mismatch with v1'
    existing_scores = pd.read_csv(BASE / 'submission_mva/results/per_image_scores.csv')
    cc_f30k_ids = sorted(existing_scores.query('source == "ChineseCLIP_F30k"')['img_id'].tolist())
    assert cc_f30k_ids == [s['img_id'] for s in f30k], 'F30k ID mismatch with existing ChineseCLIP_F30k'

    protocol_path = DEST_V2 / 'protocol_v2.txt'
    assert protocol_path.exists(), f'Missing fixed protocol: {protocol_path}'

    manifest = dict(
        seed=SEED,
        methods=list(METHODS),
        f8k_selected_indices=selected.tolist(),
        f8k_occlusion_ids=[f8k[i]['img_id'] for i in selected],
        protocol_sha256=sha(protocol_path),
        code_sha256={
            p.name: sha(p)
            for p in [Path(__file__), Path(__file__).with_name('validation_v2_common.py')]
        },
        v1_manifest_sha256=sha(DEST_V1 / 'input_manifest.json'),
        v1_retrieval_sha256={
            s: sha(DEST_V1 / s / 'retrieval_arrays.npz')
            for s in ('ChineseCLIP_F8k', 'AltCLIP_F8k')
        },
        f8k_samples=[
            dict(img_id=s['img_id'], image_sha256=sha(s['img_path']),
                 en_caps=s['en_caps'], zh_caps=s['zh_caps'])
            for s in f8k
        ],
        f30k_samples=[
            dict(img_id=s['img_id'], image_sha256=sha(s['img_path']),
                 en_caps=s['en_caps'], zh_caps=s['zh_caps'])
            for s in f30k
        ],
    )
    target = DEST_V2 / 'input_manifest_v2.json'
    if target.exists():
        existing = json.loads(target.read_text(encoding='utf-8'))
        assert existing == manifest, (
            'Inputs, protocol, or core inference code changed relative to input_manifest_v2.json; '
            'do not mix runs in the same directory.'
        )
    else:
        save_json(target, manifest)
    print(
        f'Prepared v2 manifest: F8k={len(f8k)} (occ={len(selected)}), F30k={len(f30k)}',
        flush=True,
    )
    return f8k, f30k, selected


def record_runtime(path: Path, meta: dict, out: Path) -> None:
    files = [
        p for p in path.iterdir()
        if p.suffix in ('.json', '.model', '.txt') or p.name == 'model.safetensors'
    ]
    rec = dict(meta)
    rec.update(
        python=platform.python_version(),
        numpy=np.__version__,
        files_sha256={p.name: sha(p) for p in sorted(files)},
        input_manifest_v2_sha256=sha(DEST_V2 / 'input_manifest_v2.json'),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    if torch.cuda.is_available() and rec['device'] == 'cuda':
        rec['gpu'] = torch.cuda.get_device_name(0)
    save_json(out / 'runtime.json', rec)


def run_attributions(
    model,
    processor,
    samples: list,
    source: str,
    out: Path,
    device: str,
    compare_archived: bool = False,
) -> None:
    folder = out / 'attributions'
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n = len(samples)
    done_before = 0

    for i, sample in enumerate(samples):
        stem = Path(sample['img_id']).stem
        target = folder / f'{stem}.npz'
        if target.exists():
            done_before += 1
            continue
        x = pixels(processor, sample)
        res = extract_image_attributions(
            model, processor, x, sample['en_caps'], sample['zh_caps'], device
        )
        save_npz_atomic(
            target,
            patch_gradcam=res['maps']['patch_gradcam'],
            lastblock_gradcam=res['maps']['lastblock_gradcam'],
            chefer_relevance=res['maps']['chefer_relevance'],
            caption_ok=res['caption_ok'],
            lang_complete=res['lang_complete'],
            img_feat=res['img_feat'],
            text_feat=res['text_feat'],
        )
        if res['failures']:
            save_json(folder / f'{stem}_failures.json', res['failures'])
        completed = i + 1
        if completed % 50 == 0 or completed == n:
            elapsed = time.time() - start
            new_done = completed - done_before
            rate = new_done / max(elapsed, 1e-6)
            eta = (n - completed) / max(rate, 1e-6) if new_done > 0 else 0.0
            print(
                f'{source} attributions {completed}/{n} | elapsed {elapsed:.1f}s | '
                f'rate {rate:.2f} img/s | ETA {eta/60:.1f} min',
                flush=True,
            )

    # Consolidate per-image files into fast-loading maps_all.npz and execution logs.
    stacked = {m: [] for m in METHODS}
    lang_complete_all = []
    caption_ok_all = []
    log_rows = []
    arch_rows = []

    for sample in samples:
        stem = Path(sample['img_id']).stem
        with np.load(folder / f'{stem}.npz') as z:
            for m in METHODS:
                stacked[m].append(z[m])
            lc = z['lang_complete']
            co = z['caption_ok']
            lang_complete_all.append(lc)
            caption_ok_all.append(co)
            for mi, m in enumerate(METHODS):
                for li, lname in enumerate(('EN', 'ZH')):
                    log_rows.append(dict(
                        source=source,
                        img_id=sample['img_id'],
                        method=m,
                        language=lname,
                        n_caption_ok=int(co[mi, li].sum()),
                        lang_complete=bool(lc[mi, li]),
                    ))
            if compare_archived:
                arch = archived_map_pair(source, sample)
                cur = z['patch_gradcam']
                for li, lname in enumerate(('EN', 'ZH')):
                    ok = bool(lc[0, li])
                    if ok:
                        diff = np.abs(cur[li] - arch[li])
                        mae = float(diff.mean())
                        mx = float(diff.max())
                        r = float(np.corrcoef(cur[li].ravel(), arch[li].ravel())[0, 1])
                    else:
                        mae, mx, r = np.nan, np.nan, np.nan
                    arch_rows.append(dict(
                        source=source,
                        img_id=sample['img_id'],
                        language=lname,
                        complete=ok,
                        mean_abs_error=mae,
                        max_abs_error=mx,
                        pearson=r,
                    ))

    save_npz_atomic(
        out / 'maps_all.npz',
        ids=np.array([s['img_id'] for s in samples]),
        patch_gradcam=np.stack(stacked['patch_gradcam'], axis=0),
        lastblock_gradcam=np.stack(stacked['lastblock_gradcam'], axis=0),
        chefer_relevance=np.stack(stacked['chefer_relevance'], axis=0),
        lang_complete=np.stack(lang_complete_all, axis=0),
        caption_ok=np.stack(caption_ok_all, axis=0),
    )
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out / 'caption_log.csv', index=False)

    if compare_archived:
        arch_df = pd.DataFrame(arch_rows)
        arch_df.to_csv(out / 'patch_vs_archived.csv', index=False, float_format='%.12g')
        ok_arch = arch_df[arch_df['complete']]
        n_inc = int((~arch_df['complete']).sum())
        assert len(ok_arch) > 0, f'{source}: zero complete patch_gradcam maps'
        max_mae = float(ok_arch['mean_abs_error'].max())
        min_r = float(ok_arch['pearson'].min())
        print(
            f'{source} full {len(samples)}-img patch_gradcam vs archived: '
            f'complete_lang_maps={len(ok_arch)}/{len(arch_df)} (incomplete={n_inc}), '
            f'max_MAE={max_mae:.6e}, min_r={min_r:.8f}',
            flush=True,
        )
        assert max_mae < 0.02 and min_r > 0.98, f'{source}: archived mismatch MAE={max_mae}, r={min_r}'


def run_occlusion_new_methods(
    model,
    processor,
    samples: list,
    selected: np.ndarray,
    source: str,
    out: Path,
    device: str,
    batch: int,
    reuse_v1_random: bool = True,
) -> None:
    """Run top/bottom occlusion for lastblock_gradcam and chefer_relevance on 200 shared images."""
    method_dirs = {m: out / f'occlusion_{m}' for m in NEW_METHODS}
    for d in method_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    start = time.time()
    n_sel = len(selected)
    for counter, index in enumerate(selected):
        sample = samples[int(index)]
        stem = Path(sample['img_id']).stem
        targets = {m: method_dirs[m] / f'{stem}.npz' for m in NEW_METHODS}
        if all(t.exists() for t in targets.values()):
            continue

        with np.load(out / 'attributions' / f'{stem}.npz') as z:
            lc = z['lang_complete']
            method_maps = {m: z[m] for m in NEW_METHODS}

        # Verify v1 random baselines and original score match.
        v1_path = DEST_V1 / source / 'occlusion' / f'{stem}.npz'
        v1_data = None
        if reuse_v1_random:
            assert v1_path.exists(), f'Missing v1 occlusion file for random baseline reuse: {v1_path}'
            with np.load(v1_path) as vz:
                v1_data = dict(
                    original=vz['original'].copy(),
                    masked_random=vz['masked'][:, 2:, :].copy(),
                    drop_random=vz['drop'][:, 2:, :].copy(),
                    fractions=vz['fractions'].copy(),
                )
            np.testing.assert_allclose(v1_data['fractions'], FRACTIONS, atol=0, rtol=0)

        x = pixels(processor, sample)
        with torch.inference_mode():
            feats = text_features(model, processor, sample['en_caps'] + sample['zh_caps'], device).reshape(2, 5, -1)
            original = (image_features(model, x, device) @ feats.mean(1).T).cpu().numpy()[0]

        if v1_data is not None:
            orig_err = float(np.max(np.abs(original - v1_data['original'])))
            assert orig_err < 1e-5, f'{source} {stem}: original score drift vs v1 = {orig_err}'

        # Compute random orders directly only when reuse_v1_random is False (e.g., synthetic smoke).
        random_masked = None
        if v1_data is None:
            rng = np.random.default_rng(SEED + int(index) * 100)
            random_orders = [rng.permutation(256) for _ in range(REPEATS)]
            rmasks = [mask_pixels(x, order, k) for order in random_orders for k in MASK_COUNTS]
            with torch.inference_mode():
                r_img_feats = []
                for j in range(0, len(rmasks), batch):
                    r_img_feats.append(image_features(model, torch.cat(rmasks[j:j + batch]), device))
                r_img_feats = torch.cat(r_img_feats, dim=0)  # (25, D)
                # Same random pixel masks evaluated against EN and ZH captions.
                lang_means = feats.mean(1)  # (2, D)
                r_scores = (lang_means @ r_img_feats.T).cpu().numpy().reshape(2, REPEATS, len(MASK_COUNTS))
            random_masked = r_scores
        else:
            random_masked = v1_data['masked_random']

        # Batch top + bottom masks across both new methods in one loop per image.
        valid_methods = []
        masks = []
        meta_idx = []  # (method, lang, cond_idx 0=top/1=bottom, k_idx)
        for m in NEW_METHODS:
            mi = METHODS.index(m)
            if not bool(lc[mi].all()):
                # Record incomplete explicitly without substituting.
                nan_arr = np.full((2, 2 + REPEATS, len(MASK_COUNTS)), np.nan, dtype=np.float64)
                save_npz_atomic(
                    targets[m],
                    complete=np.array(False),
                    original=original,
                    masked=nan_arr,
                    drop=nan_arr,
                    fractions=FRACTIONS,
                    conditions=np.array(['top', 'bottom'] + [f'random{i}' for i in range(REPEATS)]),
                )
                continue
            valid_methods.append(m)
            for lang in range(2):
                ord_top = patch_order(method_maps[m][lang], SEED + int(index) * 100 + 70 + lang, True)
                ord_bot = patch_order(method_maps[m][lang], SEED + int(index) * 100 + 70 + lang, False)
                for ci, order in enumerate((ord_top, ord_bot)):
                    for ki, k in enumerate(MASK_COUNTS):
                        masks.append(mask_pixels(x, order, k))
                        meta_idx.append((m, lang, ci, ki))

        tb_scores = {m: np.zeros((2, 2, len(MASK_COUNTS)), dtype=np.float64) for m in valid_methods}
        if masks:
            with torch.inference_mode():
                lang_means = feats.mean(1)  # (2, D)
                out_scores = []
                for j in range(0, len(masks), batch):
                    z = image_features(model, torch.cat(masks[j:j + batch]), device)
                    langs = torch.tensor([meta_idx[t][1] for t in range(j, min(j + batch, len(masks)))], device=device)
                    out_scores.extend((z * lang_means[langs]).sum(-1).cpu().numpy().tolist())
            for val, (m, lang, ci, ki) in zip(out_scores, meta_idx):
                tb_scores[m][lang, ci, ki] = val

        for m in valid_methods:
            full_masked = np.concatenate([tb_scores[m], random_masked], axis=1)  # (2, 7, 5)
            assert np.isfinite(full_masked).all() and np.isfinite(original).all()
            full_drop = original[:, None, None] - full_masked
            if v1_data is not None:
                # Reassert exact equality with v1 random drops.
                np.testing.assert_allclose(full_drop[:, 2:, :], v1_data['drop_random'], atol=1e-5, rtol=0)
            save_npz_atomic(
                targets[m],
                complete=np.array(True),
                original=original,
                masked=full_masked,
                drop=full_drop,
                fractions=FRACTIONS,
                conditions=np.array(['top', 'bottom'] + [f'random{i}' for i in range(REPEATS)]),
            )

        if (counter + 1) % 10 == 0 or (counter + 1) == n_sel:
            print(
                f'{source} v2 occlusion {counter + 1}/{n_sel} elapsed {time.time() - start:.1f}s',
                flush=True,
            )


def run_retrieval(model, processor, samples: list, source: str, out: Path, device: str, batch: int) -> None:
    folder = out / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    missing = [i for i, s in enumerate(samples) if not (folder / f'{Path(s["img_id"]).stem}.npz').exists()]
    start = time.time()

    for offset in range(0, len(missing), batch):
        indexes = missing[offset:offset + batch]
        x = torch.cat([pixels(processor, samples[i]) for i in indexes])
        with torch.inference_mode():
            im = image_features(model, x, device).cpu().numpy()
            texts = [c for i in indexes for key in ('en_caps', 'zh_caps') for c in samples[i][key]]
            tx = np.concatenate([
                text_features(model, processor, texts[j:j + 20], device).cpu().numpy()
                for j in range(0, len(texts), 20)
            ]).reshape(len(indexes), 2, 5, -1)
        assert np.isfinite(im).all() and np.isfinite(tx).all()
        for j, index in enumerate(indexes):
            target = folder / f'{Path(samples[index]["img_id"]).stem}.npz'
            save_npz_atomic(target, image=im[j], text=tx[j])
        if offset % max(batch * 20, 80) == 0 or offset + len(indexes) == len(missing):
            print(
                f'{source} retrieval {offset + len(indexes)}/{len(missing)} elapsed {time.time() - start:.1f}s',
                flush=True,
            )

    ims, texts = [], []
    for sample in samples:
        with np.load(folder / f'{Path(sample["img_id"]).stem}.npz') as z:
            ims.append(z['image'])
            texts.append(z['text'])
    im, tx = np.stack(ims), np.stack(texts)
    similarity, t2i, i2t = outcome_arrays(im, tx)
    save_npz_atomic(
        out / 'retrieval_arrays.npz',
        ids=np.array([s['img_id'] for s in samples]),
        image=im,
        text=tx,
        similarity=similarity,
        t2i_rank=t2i,
        i2t_rank=i2t,
    )
    rows = []
    for i, s in enumerate(samples):
        row = dict(
            img_id=s['img_id'],
            failure_at1=float((t2i[i] > 1).mean()),
            bilingual_logrank_gap=float(abs(np.log(t2i[i, 0]).mean() - np.log(t2i[i, 1]).mean())),
        )
        for lang, lname in enumerate(('EN', 'ZH')):
            for k in (1, 5, 10):
                row[f'T2I_{lname}_R{k}'] = float((t2i[i, lang] <= k).mean())
                row[f'I2T_{lname}_R{k}'] = float(i2t[i, lang] <= k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'retrieval_per_image.csv', index=False)


def smoke_test(model_key: str, f8k: list, f30k: list, selected: np.ndarray, device: str) -> None:
    """CPU-safe end-to-end equivalence and sanity check on 1 F8k + 1 F30k image."""
    from src.heatmap import get_averaged_gradcam

    t0 = time.time()
    source_f8k = f'{model_key}_F8k'
    out = DEST_V2 / 'smoke' / model_key
    out.mkdir(parents=True, exist_ok=True)

    model, processor, _path, meta = load_model(model_key, device)
    sample = f8k[int(selected[0])]
    x = pixels(processor, sample)

    # 1. Extract all 3 methods via optimized single-forward autograd.grad.
    res = extract_image_attributions(model, processor, x, sample['en_caps'], sample['zh_caps'], device)
    assert res['caption_ok'].all(), f'{model_key}: caption failure in smoke: {res["failures"]}'
    assert res['lang_complete'].all(), f'{model_key}: incomplete lang map in smoke'

    # 2. Verify exact mathematical equivalence with legacy src/heatmap.py on patch_gradcam.
    for p in model.parameters():
        p.requires_grad_(True)
    from PIL import Image
    with Image.open(sample['img_path']) as im:
        pil_img = im.convert('RGB')
    leg_en = get_averaged_gradcam(model, processor, pil_img, sample['en_caps'], device=device)
    leg_zh = get_averaged_gradcam(model, processor, pil_img, sample['zh_caps'], device=device)
    for p in model.parameters():
        p.requires_grad_(False)

    err_leg_en = float(np.max(np.abs(res['maps']['patch_gradcam'][0] - leg_en)))
    err_leg_zh = float(np.max(np.abs(res['maps']['patch_gradcam'][1] - leg_zh)))
    assert err_leg_en < 1e-5 and err_leg_zh < 1e-5, (
        f'{model_key}: patch_gradcam mismatch vs src/heatmap.py: EN={err_leg_en}, ZH={err_leg_zh}'
    )

    # 3. Verify agreement with archived v1 maps.
    arch = archived_map_pair(source_f8k, sample)
    mae_arch_en = float(np.mean(np.abs(res['maps']['patch_gradcam'][0] - arch[0])))
    mae_arch_zh = float(np.mean(np.abs(res['maps']['patch_gradcam'][1] - arch[1])))
    r_arch_en = float(np.corrcoef(res['maps']['patch_gradcam'][0].ravel(), arch[0].ravel())[0, 1])
    r_arch_zh = float(np.corrcoef(res['maps']['patch_gradcam'][1].ravel(), arch[1].ravel())[0, 1])
    assert max(mae_arch_en, mae_arch_zh) < 1e-3 and min(r_arch_en, r_arch_zh) > 0.9999

    # 4. Verify occlusion matches v1 original score and produces valid finite arrays.
    save_npz_atomic(
        out / 'attributions' / f'{Path(sample["img_id"]).stem}.npz',
        patch_gradcam=res['maps']['patch_gradcam'],
        lastblock_gradcam=res['maps']['lastblock_gradcam'],
        chefer_relevance=res['maps']['chefer_relevance'],
        caption_ok=res['caption_ok'],
        lang_complete=res['lang_complete'],
    )
    # Run 1-image occlusion check reusing v1 random baselines.
    # Temporarily copy into smoke source folder layout.
    run_occlusion_new_methods(
        model, processor, f8k, selected[:1], source_f8k, out, device, batch=4, reuse_v1_random=True
    )

    # 5. Verify Flickr30k-CN sample extraction + feature normalization.
    s30 = f30k[0]
    x30 = pixels(processor, s30)
    res30 = extract_image_attributions(model, processor, x30, s30['en_caps'], s30['zh_caps'], device)
    assert res30['caption_ok'].all() and res30['lang_complete'].all()
    assert abs(float(np.linalg.norm(res30['img_feat'])) - 1.0) < 1e-5

    # Compute summary diagnostics for all 3 methods on smoke F8k image.
    from audit_existing_data import map_metrics
    diag = {}
    for m in METHODS:
        mm = map_metrics(res['maps'][m][0:1], res['maps'][m][1:2]).iloc[0].to_dict()
        diag[m] = {k: round(float(v), 6) for k, v in mm.items()}

    report = dict(
        model_key=model_key,
        device=device,
        f8k_img=sample['img_id'],
        f30k_img=s30['img_id'],
        max_abs_err_vs_legacy_heatmap=dict(EN=err_leg_en, ZH=err_leg_zh),
        vs_archived_v1=dict(mae_EN=mae_arch_en, mae_ZH=mae_arch_zh, r_EN=r_arch_en, r_ZH=r_arch_zh),
        f8k_smoke_metrics=diag,
        elapsed_seconds=round(time.time() - t0, 2),
    )
    save_json(out / 'smoke_report.json', report)
    print(f'{model_key} smoke OK in {report["elapsed_seconds"]}s', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    del model
    gc.collect()


def run_model_stage(
    model_key: str,
    f8k: list,
    f30k: list,
    selected: np.ndarray,
    device: str,
    batch: int,
    include_cc_f30k_maps: bool = False,
) -> None:
    manifest_sha = sha(DEST_V2 / 'input_manifest_v2.json')
    out_f8k = DEST_V2 / f'{model_key}_F8k'
    out_f30k = DEST_V2 / f'{model_key}_F30k'
    out_f8k.mkdir(parents=True, exist_ok=True)
    out_f30k.mkdir(parents=True, exist_ok=True)

    f8k_done = (out_f8k / 'COMPLETE.json').exists()
    f30k_done = (out_f30k / 'COMPLETE.json').exists()
    if f8k_done:
        assert json.loads((out_f8k / 'COMPLETE.json').read_text(encoding='utf-8'))['input_manifest_v2_sha256'] == manifest_sha
    if f30k_done:
        assert json.loads((out_f30k / 'COMPLETE.json').read_text(encoding='utf-8'))['input_manifest_v2_sha256'] == manifest_sha
    if f8k_done and f30k_done:
        print(f'{model_key}: both F8k and F30k already complete.', flush=True)
        return

    model, processor, path, meta = load_model(model_key, device)

    if not f8k_done:
        record_runtime(path, meta, out_f8k)
        run_attributions(model, processor, f8k, f'{model_key}_F8k', out_f8k, device, compare_archived=True)
        if device == 'cuda':
            torch.cuda.empty_cache()
        run_occlusion_new_methods(
            model, processor, f8k, selected, f'{model_key}_F8k', out_f8k, device, batch=batch, reuse_v1_random=True
        )
        save_json(
            out_f8k / 'COMPLETE.json',
            dict(
                input_manifest_v2_sha256=manifest_sha,
                n_images=len(f8k),
                n_occlusion=len(selected),
                methods=list(METHODS),
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )

    if not f30k_done:
        record_runtime(path, meta, out_f30k)
        extract_f30k_maps = (model_key == 'AltCLIP') or include_cc_f30k_maps
        if extract_f30k_maps:
            run_attributions(model, processor, f30k, f'{model_key}_F30k', out_f30k, device, compare_archived=False)
            if device == 'cuda':
                torch.cuda.empty_cache()
        run_retrieval(model, processor, f30k, f'{model_key}_F30k', out_f30k, device, batch=batch)
        save_json(
            out_f30k / 'COMPLETE.json',
            dict(
                input_manifest_v2_sha256=manifest_sha,
                n_images=len(f30k),
                has_attribution_maps=extract_f30k_maps,
                has_retrieval=True,
                methods=list(METHODS) if extract_f30k_maps else [],
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['prepare', 'smoke', 'full'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--model', choices=['ChineseCLIP', 'AltCLIP'], default=None)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--include-cc-f30k-maps', action='store_true')
    args = p.parse_args()

    f8k, f30k, selected = prepare()
    if args.mode == 'prepare':
        return

    model_keys = [args.model] if args.model else ['ChineseCLIP', 'AltCLIP']
    for mk in model_keys:
        if args.mode == 'smoke':
            smoke_test(mk, f8k, f30k, selected, args.device)
        else:
            run_model_stage(mk, f8k, f30k, selected, args.device, args.batch, args.include_cc_f30k_maps)


if __name__ == '__main__':
    main()
