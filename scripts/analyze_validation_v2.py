"""Pure-CPU statistical analysis for completed MVA v2 runs (ROI #1 + ROI #2).

Strictly implements submission_mva/validation_v2/protocol_v2.txt without any
GPU inference or outcome-dependent selection. Writes all outputs under
submission_mva/validation_v2/analysis/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

from audit_existing_data import COMP, components, fit, holm, map_metrics
from validation_v2_common import (
    BASE,
    DEST_V1,
    DEST_V2,
    FRACTIONS,
    METHODS,
    NEW_METHODS,
    SEED,
    dataset_f8k,
    dataset_f30k,
    save_json,
    selection_f8k,
)

OUT = DEST_V2 / 'analysis'
F8K_SOURCES = ('ChineseCLIP_F8k', 'AltCLIP_F8k')
F30K_SOURCES = ('ChineseCLIP_F30k', 'AltCLIP_F30k')
PREDICTORS = ('cvad', 'inv_iou', 'kl_symmetric', 'equal_weight', 'cos_en_zh')


def save_csv(name: str, rows) -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / f'{name}.csv', index=False, float_format='%.12g')
    return df


def bootstrap_mean(x, n: int = 10000, alpha: float = 0.05):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    vals = []
    for offset in range(0, n, 200):
        idx = rng.integers(len(x), size=(min(200, n - offset), len(x)))
        vals.extend(x[idx].mean(1))
    return np.quantile(vals, [alpha / 2, 1 - alpha / 2])


def bootstrap_rho(x, y, n: int = 2000):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    vals = []
    for offset in range(0, n, 100):
        idx = rng.integers(len(x), size=(min(100, n - offset), len(x)))
        a = stats.rankdata(x[idx], axis=1)
        b = stats.rankdata(y[idx], axis=1)
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        vals.extend(np.divide((a * b).sum(1), denom, out=np.zeros(len(denom)), where=denom > 0))
    return np.quantile(vals, [0.025, 0.975])


def paired_stats(delta) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    lo, hi = bootstrap_mean(delta)
    if np.all(delta == 0):
        w, p, dz = 0.0, 1.0, 0.0
    else:
        w, p = stats.wilcoxon(delta, zero_method='wilcox', alternative='two-sided', method='approx')
        sd = delta.std(ddof=1)
        dz = float(delta.mean() / sd) if sd > 0 else 0.0
    return dict(
        n=int(len(delta)),
        mean_difference=float(delta.mean()),
        median_difference=float(np.median(delta)),
        ci95_low=float(lo),
        ci95_high=float(hi),
        cohen_dz=float(dz),
        fraction_positive=float((delta > 0).mean()),
        w=float(w),
        p_raw=float(p),
    )


def compute_method_metrics_table(ids, maps_3d, complete_2d, source: str, method: str, threshold: float = 0.5):
    """Compute metrics for all complete (EN & ZH) pairs in maps_3d (N, 2, 16, 16)."""
    valid_mask = complete_2d.all(axis=1)
    valid_ids = np.asarray(ids)[valid_mask]
    en = maps_3d[valid_mask, 0]
    zh = maps_3d[valid_mask, 1]
    df = components(map_metrics(en, zh, threshold=threshold, eps=1e-9))
    df.insert(0, 'img_id', valid_ids)
    df.insert(0, 'method', method)
    df.insert(0, 'source', source)
    return df


def cv_predict_5(X_comp, cos_vec, y, folds):
    """5-fold image-level CV linear regression for 5 predictors + intercept baseline."""
    n = len(y)
    preds = {k: np.zeros(n, dtype=np.float64) for k in (*PREDICTORS, 'intercept')}
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        mn, span, w, *_ = fit(X_comp[train])
        scaled = (X_comp - mn) / span
        score = dict(
            cvad=scaled @ w,
            inv_iou=X_comp[:, 2],
            kl_symmetric=X_comp[:, 3],
            equal_weight=scaled.mean(1),
            cos_en_zh=cos_vec,
        )
        preds['intercept'][test] = y[train].mean()
        for key, x in score.items():
            design = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            preds[key][test] = design[test] @ beta
    return preds


def analyze_f8k_provenance():
    rows = []
    for source in F8K_SOURCES:
        clog = pd.read_csv(DEST_V2 / source / 'caption_log.csv')
        arch = pd.read_csv(DEST_V2 / source / 'patch_vs_archived.csv')
        for m in METHODS:
            sub = clog[clog['method'] == m]
            both_ok = sub.groupby('img_id')['lang_complete'].all()
            row = dict(
                source=source,
                method=m,
                n_images=int(len(both_ok)),
                n_complete_pairs=int(both_ok.sum()),
                n_captions_total=int(len(sub) * 5),
                n_captions_ok=int(sub['n_caption_ok'].sum()),
            )
            if m == 'patch_gradcam':
                row.update(
                    archived_max_mae=float(arch['mean_abs_error'].max()),
                    archived_median_mae=float(arch['mean_abs_error'].median()),
                    archived_min_pearson=float(arch['pearson'].min()),
                )
            rows.append(row)
    return save_csv('f8k_provenance_summary', rows)


def analyze_f8k_spatial():
    loaded = {}
    for source in F8K_SOURCES:
        with np.load(DEST_V2 / source / 'maps_all.npz', allow_pickle=False) as z:
            loaded[source] = dict(
                ids=z['ids'],
                lang_complete=z['lang_complete'],
                maps={m: z[m] for m in METHODS},
            )

    per_img_frames = []
    pca_rows = []
    summary_rows = []
    contrast_rows = []
    thresh_rows = []

    for mi, m in enumerate(METHODS):
        m_frames = []
        for source in F8K_SOURCES:
            df = compute_method_metrics_table(
                loaded[source]['ids'],
                loaded[source]['maps'][m],
                loaded[source]['lang_complete'][:, mi, :],
                source,
                m,
                threshold=0.5,
            )
            m_frames.append(df)
        comb = pd.concat(m_frames, ignore_index=True)
        mn, span, w, var_exp, signed = fit(comb[COMP].to_numpy())
        scaled = (comb[COMP].to_numpy() - mn) / span
        comb['cvad'] = scaled @ w
        comb['equal_weight'] = scaled.mean(1)
        per_img_frames.append(comb)

        for j, cname in enumerate(COMP):
            pca_rows.append(dict(
                method=m,
                component=cname,
                minimum=float(mn[j]),
                maximum=float(mn[j] + span[j]),
                weight=float(w[j]),
                signed_pc1_loading=float(signed[j]),
                explained_variance=float(var_exp),
                n_calibration_rows=int(len(comb)),
            ))

        for source in F8K_SOURCES:
            sub = comb[comb['source'] == source]
            rho_iou = float(stats.spearmanr(sub['cvad'], sub['inv_iou']).statistic)
            summary_rows.append(dict(
                source=source,
                method=m,
                n=int(len(sub)),
                cvad_mean=float(sub['cvad'].mean()),
                cvad_sd=float(sub['cvad'].std(ddof=1)),
                iou_mean=float(sub['iou'].mean()),
                iou_sd=float(sub['iou'].std(ddof=1)),
                kl_mean=float(sub['kl_symmetric'].mean()),
                kl_sd=float(sub['kl_symmetric'].std(ddof=1)),
                cos_mean=float(sub['cos_en_zh'].mean()),
                cos_sd=float(sub['cos_en_zh'].std(ddof=1)),
                rho_cvad_inv_iou=rho_iou,
            ))

        # Paired Chinese-CLIP minus AltCLIP on common valid images.
        cc = comb[comb['source'] == 'ChineseCLIP_F8k'].set_index('img_id')
        alt = comb[comb['source'] == 'AltCLIP_F8k'].set_index('img_id')
        common = sorted(cc.index.intersection(alt.index))
        for metric in ('inv_iou', 'cos_en_zh', 'cvad', 'kl_symmetric'):
            res = paired_stats(cc.loc[common, metric].to_numpy() - alt.loc[common, metric].to_numpy())
            contrast_rows.append(dict(
                method=m,
                metric=metric,
                primary_family=(metric in ('inv_iou', 'cos_en_zh')),
                **res,
            ))

        # Threshold sensitivity across tau in {0.3, 0.4, 0.5, 0.6, 0.7} holding method PCA fixed.
        ref_cvad = {
            s: comb[comb['source'] == s].set_index('img_id')['cvad']
            for s in F8K_SOURCES
        }
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            for source in F8K_SOURCES:
                df_t = compute_method_metrics_table(
                    loaded[source]['ids'],
                    loaded[source]['maps'][m],
                    loaded[source]['lang_complete'][:, mi, :],
                    source,
                    m,
                    threshold=tau,
                ).set_index('img_id')
                sc_t = (df_t[COMP].to_numpy() - mn) / span
                cv_t = sc_t @ w
                common_t = ref_cvad[source].index.intersection(df_t.index)
                rho_ref = float(stats.spearmanr(
                    ref_cvad[source].loc[common_t].to_numpy(),
                    pd.Series(cv_t, index=df_t.index).loc[common_t].to_numpy(),
                ).statistic)
                thresh_rows.append(dict(
                    method=m,
                    source=source,
                    tau=tau,
                    n=int(len(df_t)),
                    cvad_mean=float(cv_t.mean()),
                    iou_mean=float(df_t['iou'].mean()),
                    cvad_spearman_vs_05=rho_ref,
                ))

    per_img_df = save_csv('f8k_spatial_per_image', pd.concat(per_img_frames, ignore_index=True))
    save_csv('f8k_spatial_pca_fit', pca_rows)
    save_csv('f8k_spatial_summary', summary_rows)

    cdf = pd.DataFrame(contrast_rows)
    cdf['p_holm_primary_6'] = np.nan
    pmask = cdf['primary_family'].to_numpy()
    cdf.loc[pmask, 'p_holm_primary_6'] = holm(cdf.loc[pmask, 'p_raw'].to_numpy())
    cdf['p_holm_all_12'] = holm(cdf['p_raw'].to_numpy())
    save_csv('f8k_spatial_paired_contrasts', cdf)
    save_csv('f8k_threshold_sensitivity', thresh_rows)
    return per_img_df


def analyze_f8k_occlusion(f8k_samples, selected):
    auc_records = []
    curve_records = []
    contrast_rows = []
    lang_rows = []
    per_source_method_auc = {}  # (source, method) -> dict(img_id -> (2, 3) array [top, bottom, random])

    for source in F8K_SOURCES:
        for method in METHODS:
            img_map = {}
            for index in selected:
                s = f8k_samples[int(index)]
                stem = Path(s['img_id']).stem
                if method == 'patch_gradcam':
                    path = DEST_V1 / source / 'occlusion' / f'{stem}.npz'
                else:
                    path = DEST_V2 / source / f'occlusion_{method}' / f'{stem}.npz'
                with np.load(path) as z:
                    if 'complete' in z and not bool(z['complete']):
                        continue
                    drops = np.concatenate([np.zeros((2, 7, 1)), z['drop']], axis=2)
                    np.testing.assert_allclose(z['fractions'], FRACTIONS, atol=0, rtol=0)
                    assert drops.shape == (2, 7, 6) and np.isfinite(drops).all()
                    np.testing.assert_allclose(z['original'][:, None, None] - z['masked'], z['drop'], atol=1e-6)
                reduced = np.stack([drops[:, 0], drops[:, 1], drops[:, 2:].mean(1)], axis=1)  # (2, 3, 6)
                auc = np.trapz(reduced, FRACTIONS, axis=2) / FRACTIONS[-1]  # (2, 3)
                img_map[s['img_id']] = auc
                for li, lname in enumerate(('EN', 'ZH')):
                    for ci, cname in enumerate(('top', 'bottom', 'random')):
                        auc_records.append(dict(
                            source=source, method=method, img_id=s['img_id'],
                            language=lname, condition=cname, auc=float(auc[li, ci]),
                        ))
                        for fi, frac in enumerate(FRACTIONS):
                            curve_records.append(dict(
                                source=source, method=method, img_id=s['img_id'],
                                language=lname, condition=cname, fraction=float(frac),
                                drop=float(reduced[li, ci, fi]),
                            ))
            per_source_method_auc[(source, method)] = img_map

            ids_ok = sorted(img_map.keys())
            arr = np.stack([img_map[k] for k in ids_ok], axis=0)  # (n, 2, 3)
            for other, col in (('random', 2), ('bottom', 1)):
                # EN/ZH averaged within image (primary endpoint)
                diff_avg = (arr[:, :, 0] - arr[:, :, col]).mean(axis=1)
                contrast_rows.append(dict(
                    source=source,
                    method=method,
                    comparison=f'top_minus_{other}',
                    is_v2_primary=(method in NEW_METHODS),
                    **paired_stats(diff_avg),
                ))
                # Language-stratified secondary diagnostic
                for li, lname in enumerate(('EN', 'ZH')):
                    diff_lang = arr[:, li, 0] - arr[:, li, col]
                    lang_rows.append(dict(
                        source=source,
                        method=method,
                        language=lname,
                        comparison=f'top_minus_{other}',
                        **paired_stats(diff_lang),
                    ))

    cdf = pd.DataFrame(contrast_rows)
    cdf['p_holm_v2_8'] = np.nan
    vmask = cdf['is_v2_primary'].to_numpy()
    cdf.loc[vmask, 'p_holm_v2_8'] = holm(cdf.loc[vmask, 'p_raw'].to_numpy())
    cdf['p_holm_all_12'] = holm(cdf['p_raw'].to_numpy())
    save_csv('f8k_occlusion_contrasts', cdf)

    ldf = pd.DataFrame(lang_rows)
    ldf['p_holm_24'] = holm(ldf['p_raw'].to_numpy())
    save_csv('f8k_occlusion_by_language', ldf)

    # Cross-method paired faithfulness improvement tests on common valid images.
    cross_rows = []
    method_pairs = [
        ('lastblock_gradcam', 'patch_gradcam'),
        ('chefer_relevance', 'patch_gradcam'),
        ('chefer_relevance', 'lastblock_gradcam'),
    ]
    for source in F8K_SOURCES:
        for m_a, m_b in method_pairs:
            map_a = per_source_method_auc[(source, m_a)]
            map_b = per_source_method_auc[(source, m_b)]
            common = sorted(set(map_a).intersection(map_b))
            arr_a = np.stack([map_a[k] for k in common], axis=0)
            arr_b = np.stack([map_b[k] for k in common], axis=0)
            for other, col in (('random', 2), ('bottom', 1)):
                d_a = (arr_a[:, :, 0] - arr_a[:, :, col]).mean(axis=1)
                d_b = (arr_b[:, :, 0] - arr_b[:, :, col]).mean(axis=1)
                cross_rows.append(dict(
                    source=source,
                    method_a=m_a,
                    method_b=m_b,
                    contrast=f'top_minus_{other}',
                    is_primary_4=(m_b == 'patch_gradcam' and other == 'random'),
                    **paired_stats(d_a - d_b),
                ))
    xdf = pd.DataFrame(cross_rows)
    xdf['p_holm_primary_4'] = np.nan
    pmask = xdf['is_primary_4'].to_numpy()
    xdf.loc[pmask, 'p_holm_primary_4'] = holm(xdf.loc[pmask, 'p_raw'].to_numpy())
    xdf['p_holm_all_12'] = holm(xdf['p_raw'].to_numpy())
    save_csv('f8k_occlusion_cross_method', xdf)

    save_csv('f8k_occlusion_auc_per_image_language', auc_records)
    curves_df = pd.DataFrame(curve_records)
    summary_curves = []
    for (source, method, language, condition, fraction), g in curves_df.groupby(
        ['source', 'method', 'language', 'condition', 'fraction']
    ):
        lo, hi = bootstrap_mean(g['drop'].to_numpy(), n=2000)
        summary_curves.append(dict(
            source=source, method=method, language=language,
            condition=condition, fraction=fraction,
            mean_drop=float(g['drop'].mean()), ci95_low=float(lo), ci95_high=float(hi),
            n=int(len(g)),
        ))
    save_csv('f8k_occlusion_curves_summary', summary_curves)


def analyze_f8k_retrieval(f8k_samples, f8k_spatial_df: pd.DataFrame):
    ids = [s['img_id'] for s in f8k_samples]
    fold_df = pd.read_csv(DEST_V1 / 'analysis' / 'retrieval_cv_folds.csv').set_index('img_id').loc[ids]
    fold_id_map = fold_df['fold'].to_numpy()

    assoc_rows = []
    perf_rows = []
    for method in METHODS:
        method_assoc = []
        for source in F8K_SOURCES:
            yframe = pd.read_csv(DEST_V1 / source / 'retrieval_per_image.csv').set_index('img_id')
            sframe = f8k_spatial_df.query('source == @source and method == @method').set_index('img_id')
            common = [img_id for img_id in ids if img_id in sframe.index and img_id in yframe.index]
            yframe = yframe.loc[common]
            sframe = sframe.loc[common]
            # Preserve exact v1 fold membership for common images.
            common_fold_ids = fold_df.loc[common, 'fold'].to_numpy()
            folds = [np.where(common_fold_ids == k)[0] for k in range(5)]

            for outcome in ('failure_at1', 'bilingual_logrank_gap'):
                y = yframe[outcome].to_numpy()
                for key in PREDICTORS:
                    x = sframe[key].to_numpy()
                    r, p = stats.spearmanr(x, y)
                    lo, hi = bootstrap_rho(x, y)
                    method_assoc.append(dict(
                        dataset='Flickr8k-CN',
                        method=method,
                        source=source,
                        outcome=outcome,
                        predictor=key,
                        n=int(len(common)),
                        rho=float(r),
                        ci95_low=float(lo),
                        ci95_high=float(hi),
                        p_raw=float(p),
                    ))
                preds = cv_predict_5(sframe[COMP].to_numpy(), sframe['cos_en_zh'].to_numpy(), y, folds)
                base_mse = float(np.mean((preds['intercept'] - y) ** 2))
                for key, yp in preds.items():
                    err = yp - y
                    mse = float(np.mean(err ** 2))
                    perf_rows.append(dict(
                        dataset='Flickr8k-CN',
                        method=method,
                        source=source,
                        outcome=outcome,
                        predictor=key,
                        n=int(len(common)),
                        mse=mse,
                        mae=float(np.mean(np.abs(err))),
                        r2_vs_fold_intercept=float(1.0 - mse / base_mse),
                    ))
        mdf = pd.DataFrame(method_assoc)
        mdf['p_holm_20'] = holm(mdf['p_raw'].to_numpy())
        assoc_rows.append(mdf)

    save_csv('f8k_retrieval_associations', pd.concat(assoc_rows, ignore_index=True))
    save_csv('f8k_retrieval_cv_performance', perf_rows)


def analyze_f30k(f30k_samples):
    ids = [s['img_id'] for s in f30k_samples]
    n = len(ids)

    # Original v1 calibration parameters for patch_gradcam and F8k method PCA parameters
    # so Flickr30k-CN scores sit on the exact same calibration scales as Flickr8k-CN.
    v1_pca = pd.read_csv(BASE / 'submission_mva/results/pca_fit.csv').set_index('component').loc[COMP]
    v1_mn = v1_pca['minimum'].to_numpy()
    v1_span = (v1_pca['maximum'] - v1_pca['minimum']).to_numpy()
    v1_w = v1_pca['weight'].to_numpy()

    f8k_pca_df = pd.read_csv(OUT / 'f8k_spatial_pca_fit.csv')
    f8k_pca = {}
    for m in METHODS:
        sub = f8k_pca_df.query('method == @m').set_index('component').loc[COMP]
        f8k_pca[m] = (
            sub['minimum'].to_numpy(),
            (sub['maximum'] - sub['minimum']).to_numpy(),
            sub['weight'].to_numpy(),
        )

    has_cc_v2 = (DEST_V2 / 'ChineseCLIP_F30k' / 'maps_all.npz').exists()
    loaded_f30k = {}
    prov_rows = []
    for source in F30K_SOURCES:
        if source == 'ChineseCLIP_F30k' and not has_cc_v2:
            continue
        with np.load(DEST_V2 / source / 'maps_all.npz', allow_pickle=False) as z:
            loaded_f30k[source] = dict(
                ids=z['ids'],
                lang_complete=z['lang_complete'],
                maps={m: z[m] for m in METHODS},
            )
        clog = pd.read_csv(DEST_V2 / source / 'caption_log.csv')
        for m in METHODS:
            sub = clog[clog['method'] == m]
            both_ok = sub.groupby('img_id')['lang_complete'].all()
            prov_rows.append(dict(
                source=source,
                method=m,
                n_images=int(len(both_ok)),
                n_complete_pairs=int(both_ok.sum()),
                n_captions_total=int(len(sub) * 5),
                n_captions_ok=int(sub['n_caption_ok'].sum()),
            ))
    if prov_rows:
        save_csv('f30k_provenance_summary', prov_rows)

    # Verify ChineseCLIP_F30k v2 patch_gradcam matches v1 archived maps when present.
    if has_cc_v2:
        with np.load(BASE / 'submission_mva/results/ChineseCLIP_F30k_maps.npz') as az:
            a_ids = list(az['ids'])
            assert a_ids == list(loaded_f30k['ChineseCLIP_F30k']['ids'])
            cur_p = loaded_f30k['ChineseCLIP_F30k']['maps']['patch_gradcam']
            lc_p = loaded_f30k['ChineseCLIP_F30k']['lang_complete'][:, 0, :]
            mae_en = float(np.mean(np.abs(cur_p[lc_p[:, 0], 0] - az['en'][lc_p[:, 0]])))
            mae_zh = float(np.mean(np.abs(cur_p[lc_p[:, 1], 1] - az['zh'][lc_p[:, 1]])))
            assert max(mae_en, mae_zh) < 0.01, (mae_en, mae_zh)

    v1_scores = pd.read_csv(BASE / 'submission_mva/results/per_image_scores.csv')
    frames = {}  # (source, method) -> DataFrame indexed by img_id
    summary_rows = []

    for mi, m in enumerate(METHODS):
        mn_m, span_m, w_m = f8k_pca[m]
        for source in F30K_SOURCES:
            if source in loaded_f30k:
                df = compute_method_metrics_table(
                    loaded_f30k[source]['ids'],
                    loaded_f30k[source]['maps'][m],
                    loaded_f30k[source]['lang_complete'][:, mi, :],
                    source,
                    m,
                    threshold=0.5,
                )
            elif source == 'ChineseCLIP_F30k' and m == 'patch_gradcam':
                df = v1_scores.query('source == "ChineseCLIP_F30k"').set_index('img_id').loc[ids].reset_index().copy()
                df['method'] = m
            else:
                continue

            if m == 'patch_gradcam':
                sc = (df[COMP].to_numpy() - v1_mn) / v1_span
                df['cvad'] = sc @ v1_w
                df['equal_weight'] = sc.mean(1)
            else:
                sc = (df[COMP].to_numpy() - mn_m) / span_m
                df['cvad'] = sc @ w_m
                df['equal_weight'] = sc.mean(1)

            frames[(source, m)] = df.set_index('img_id')
            summary_rows.append(dict(
                source=source,
                method=m,
                n=int(len(df)),
                cvad_mean=float(df['cvad'].mean()),
                cvad_sd=float(df['cvad'].std(ddof=1)),
                iou_mean=float(df['iou'].mean()),
                iou_sd=float(df['iou'].std(ddof=1)),
                kl_mean=float(df['kl_symmetric'].mean()),
                kl_sd=float(df['kl_symmetric'].std(ddof=1)),
                cos_mean=float(df['cos_en_zh'].mean()),
                cos_sd=float(df['cos_en_zh'].std(ddof=1)),
            ))

    save_csv('f30k_spatial_summary', summary_rows)

    # Paired cross-dataset replication on Flickr30k-CN (Chinese-CLIP minus AltCLIP) for all available methods.
    rep_rows = []
    for m in METHODS:
        if ('ChineseCLIP_F30k', m) not in frames or ('AltCLIP_F30k', m) not in frames:
            continue
        cc_idx = frames[('ChineseCLIP_F30k', m)]
        alt_idx = frames[('AltCLIP_F30k', m)]
        common = sorted(cc_idx.index.intersection(alt_idx.index))
        for metric in ('cvad', 'inv_iou', 'kl_symmetric', 'cos_en_zh'):
            res = paired_stats(cc_idx.loc[common, metric].to_numpy() - alt_idx.loc[common, metric].to_numpy())
            rep_rows.append(dict(dataset='Flickr30k-CN', method=m, metric=metric, **res))
    rep_df = pd.DataFrame(rep_rows)
    rep_df['p_holm_method_4'] = np.nan
    for m in rep_df['method'].unique():
        mask = rep_df['method'] == m
        rep_df.loc[mask, 'p_holm_method_4'] = holm(rep_df.loc[mask, 'p_raw'].to_numpy())
    rep_df['p_holm_all'] = holm(rep_df['p_raw'].to_numpy())
    save_csv('f30k_spatial_paired_replication', rep_df)

    # Bilingual retrieval Recall@K + within-model association + 5-fold CV across all available methods.
    folds = np.array_split(np.random.default_rng(SEED).permutation(n), 5)
    foldid = np.empty(n, dtype=int)
    for fi, f in enumerate(folds):
        foldid[f] = fi
    save_csv('f30k_retrieval_cv_folds', [dict(img_id=ids[i], fold=int(foldid[i])) for i in range(n)])

    recall_rows = []
    for source in F30K_SOURCES:
        yframe = pd.read_csv(DEST_V2 / source / 'retrieval_per_image.csv').set_index('img_id').loc[ids]
        for col in yframe.columns:
            if '_R' not in col:
                continue
            lo, hi = bootstrap_mean(yframe[col].to_numpy())
            recall_rows.append(dict(
                source=source, metric=col, n=n,
                recall=float(yframe[col].mean()), ci95_low=float(lo), ci95_high=float(hi),
            ))
    save_csv('f30k_retrieval_recall', recall_rows)

    assoc_frames = []
    perf_rows = []
    for m in METHODS:
        m_assoc = []
        for source in F30K_SOURCES:
            if (source, m) not in frames:
                continue
            yframe = pd.read_csv(DEST_V2 / source / 'retrieval_per_image.csv').set_index('img_id').loc[ids]
            sframe = frames[(source, m)]
            common_r = [img_id for img_id in ids if img_id in sframe.index]
            y_sub = yframe.loc[common_r]
            s_sub = sframe.loc[common_r]
            fold_map = pd.Series(foldid, index=ids).loc[common_r].to_numpy()
            r_folds = [np.where(fold_map == k)[0] for k in range(5)]

            for outcome in ('failure_at1', 'bilingual_logrank_gap'):
                y = y_sub[outcome].to_numpy()
                for key in PREDICTORS:
                    x = s_sub[key].to_numpy()
                    r, p = stats.spearmanr(x, y)
                    lo, hi = bootstrap_rho(x, y)
                    m_assoc.append(dict(
                        dataset='Flickr30k-CN',
                        method=m,
                        source=source,
                        outcome=outcome,
                        predictor=key,
                        n=int(len(common_r)),
                        rho=float(r),
                        ci95_low=float(lo),
                        ci95_high=float(hi),
                        p_raw=float(p),
                    ))
                preds = cv_predict_5(s_sub[COMP].to_numpy(), s_sub['cos_en_zh'].to_numpy(), y, r_folds)
                base_mse = float(np.mean((preds['intercept'] - y) ** 2))
                for key, yp in preds.items():
                    err = yp - y
                    mse = float(np.mean(err ** 2))
                    perf_rows.append(dict(
                        dataset='Flickr30k-CN',
                        method=m,
                        source=source,
                        outcome=outcome,
                        predictor=key,
                        n=int(len(common_r)),
                        mse=mse,
                        mae=float(np.mean(np.abs(err))),
                        r2_vs_fold_intercept=float(1.0 - mse / base_mse),
                    ))
        if m_assoc:
            mdf = pd.DataFrame(m_assoc)
            mdf['p_holm_method'] = holm(mdf['p_raw'].to_numpy())
            assoc_frames.append(mdf)

    save_csv('f30k_retrieval_associations', pd.concat(assoc_frames, ignore_index=True))
    save_csv('f30k_retrieval_cv_performance', perf_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        DEST_V2 / 'ChineseCLIP_F8k' / 'COMPLETE.json',
        DEST_V2 / 'AltCLIP_F8k' / 'COMPLETE.json',
        DEST_V2 / 'ChineseCLIP_F30k' / 'COMPLETE.json',
        DEST_V2 / 'AltCLIP_F30k' / 'COMPLETE.json',
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f'Incomplete v2 inference runs: {missing}'

    f8k = dataset_f8k()
    f30k = dataset_f30k()
    selected = selection_f8k(f8k)

    prov = analyze_f8k_provenance()
    f8k_sp = analyze_f8k_spatial()
    analyze_f8k_occlusion(f8k, selected)
    analyze_f8k_retrieval(f8k, f8k_sp)
    analyze_f30k(f30k)

    save_json(
        OUT / 'analysis_v2_config.json',
        dict(
            seed=SEED,
            methods=list(METHODS),
            f8k_n=len(f8k),
            f8k_occlusion_n=len(selected),
            f30k_n=len(f30k),
            predictors=list(PREDICTORS),
        ),
    )

    print('=== F8k Provenance ===')
    print(prov.to_string(index=False))
    print('\n=== F8k Spatial Paired Contrasts ===')
    print(pd.read_csv(OUT / 'f8k_spatial_paired_contrasts.csv').to_string(index=False))
    print('\n=== F8k Occlusion Contrasts (All 3 Methods) ===')
    print(pd.read_csv(OUT / 'f8k_occlusion_contrasts.csv').to_string(index=False))
    print('\n=== F8k Cross-Method Faithfulness Improvement ===')
    print(pd.read_csv(OUT / 'f8k_occlusion_cross_method.csv').to_string(index=False))
    print('\n=== F30k Paired Replication ===')
    print(pd.read_csv(OUT / 'f30k_spatial_paired_replication.csv').to_string(index=False))
    print('\n=== F30k Retrieval Recall ===')
    print(pd.read_csv(OUT / 'f30k_retrieval_recall.csv').to_string(index=False))


if __name__ == '__main__':
    main()
