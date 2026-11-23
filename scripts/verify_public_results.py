"""CPU-only verification of all headline statistics reported in the manuscript.

Run from the repository root:
    python scripts/verify_public_results.py
Requires only numpy and pandas (no datasets, model weights, GPU, or network).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_RES = ROOT / 'results' / 'baseline_layer0'
CM_RES = ROOT / 'results' / 'cross_method_and_replication'


def main():
    # 1. Four-checkpoint Layer-0 baseline on Flickr8k-CN
    s = pd.read_csv(BASE_RES / 'summary.csv').set_index(['source', 'metric'])
    assert abs(s.loc[('ChineseCLIP_F8k', 'cvad'), 'mean'] - 0.482) < 0.001
    assert abs(s.loc[('AltCLIP_F8k', 'cvad'), 'mean'] - 0.259) < 0.001
    assert abs(s.loc[('OpenAI_CLIP_B32_F8k', 'cvad'), 'mean'] - 0.634) < 0.001
    assert abs(s.loc[('OpenAI_CLIP_L14_F8k', 'cvad'), 'mean'] - 0.623) < 0.001

    # 2. Cross-method provenance: 100% caption success on deep methods across both datasets
    for fname in ('f8k_provenance_summary.csv', 'f30k_provenance_summary.csv'):
        df = pd.read_csv(CM_RES / fname)
        deep = df[df['method'].isin(['lastblock_gradcam', 'chefer_relevance'])]
        assert (deep['n_captions_ok'] == deep['n_captions_total']).all(), fname
        assert (deep['n_complete_pairs'] == deep['n_images']).all(), fname

    # 3. Cross-method spatial ordering & Flickr30k-CN replication
    f8 = pd.read_csv(CM_RES / 'f8k_spatial_paired_contrasts.csv').set_index(['method', 'metric'])
    f30 = pd.read_csv(CM_RES / 'f30k_spatial_paired_replication.csv').set_index(['method', 'metric'])
    assert abs(f30.loc[('patch_gradcam', 'cvad'), 'mean_difference'] - 0.225) < 0.001
    for m in ('patch_gradcam', 'lastblock_gradcam', 'chefer_relevance'):
        assert f8.loc[(m, 'cos_en_zh'), 'mean_difference'] < 0.0
        assert f30.loc[(m, 'cos_en_zh'), 'mean_difference'] < 0.0

    # 4. Occlusion faithfulness jump (200 shared Flickr8k-CN images)
    occ = pd.read_csv(CM_RES / 'f8k_occlusion_contrasts.csv').set_index(['source', 'method', 'comparison'])
    for src in ('ChineseCLIP_F8k', 'AltCLIP_F8k'):
        l0 = occ.loc[(src, 'patch_gradcam', 'top_minus_random'), 'mean_difference']
        l24 = occ.loc[(src, 'lastblock_gradcam', 'top_minus_random'), 'mean_difference']
        ch = occ.loc[(src, 'chefer_relevance', 'top_minus_random'), 'mean_difference']
        assert l24 / l0 > 8.0 and ch / l0 > 13.0
        assert occ.loc[(src, 'lastblock_gradcam', 'top_minus_random'), 'cohen_dz'] > 1.0
        assert occ.loc[(src, 'chefer_relevance', 'top_minus_random'), 'cohen_dz'] > 1.2

    # 5. Deep retrieval rank-gap links on both benchmarks
    for fname in ('f8k_retrieval_cv_performance.csv', 'f30k_retrieval_cv_performance.csv'):
        cv = pd.read_csv(CM_RES / fname)
        deep_gap = cv[
            (cv['method'].isin(['lastblock_gradcam', 'chefer_relevance']))
            & (cv['outcome'] == 'bilingual_logrank_gap')
            & (cv['predictor'] == 'cos_en_zh')
        ]
        assert len(deep_gap) == 4
        assert (deep_gap['r2_vs_fold_intercept'] > 0.025).all(), fname

    print('All headline manuscript statistics verified against released CSV tables.')


if __name__ == '__main__':
    main()
