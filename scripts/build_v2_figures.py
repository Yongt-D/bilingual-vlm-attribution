"""Generate all publication-grade vector figures for the revised MVA manuscript (v2).

Reads only verified CSV/NPZ outputs from submission_mva/results/,
submission_mva/validation/, and submission_mva/validation_v2/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RES_V1 = ROOT / 'results'
VAL_V1 = ROOT / 'validation' / 'analysis'
VAL_V2 = ROOT / 'validation_v2'
ANA_V2 = VAL_V2 / 'analysis'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

COLORS = {
    'ChineseCLIP_F8k': '#0072B2',
    'AltCLIP_F8k': '#009E73',
    'OpenAI_CLIP_B32_F8k': '#D55E00',
    'OpenAI_CLIP_L14_F8k': '#CC79A7',
}
METHOD_LABELS = {
    'patch_gradcam': 'Layer-0 Patch Grad-CAM',
    'lastblock_gradcam': 'Layer-24 Last-Block Grad-CAM',
    'chefer_relevance': 'All-Layer Chefer Relevance',
}
METHOD_SHORT = {
    'patch_gradcam': 'Layer-0 Grad-CAM',
    'lastblock_gradcam': 'Layer-24 Grad-CAM',
    'chefer_relevance': 'Chefer Relevance',
}
METHOD_COLORS = {
    'patch_gradcam': '#7F7F7F',
    'lastblock_gradcam': '#0072B2',
    'chefer_relevance': '#D55E00',
}


def fig_model_comparison():
    """Fig. 2: Four-model baseline distributions + F8k vs F30k paired replication."""
    d = pd.read_csv(RES_V1 / 'per_image_scores.csv')
    models = ['ChineseCLIP_F8k', 'AltCLIP_F8k', 'OpenAI_CLIP_B32_F8k', 'OpenAI_CLIP_L14_F8k']
    labels = ['Chinese-CLIP', 'AltCLIP', 'CLIP-B/32', 'CLIP-L/14']

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), layout='constrained')

    # (a) Four-model baseline CVAD distributions
    groups = [d[d.source == s]['cvad'].to_numpy() for s in models]
    bp = axes[0].boxplot(
        groups, showfliers=False, patch_artist=True, widths=0.52,
        medianprops=dict(color='#222222', lw=1.4),
        whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9),
    )
    for patch, s in zip(bp['boxes'], models):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.72)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(0.8)
    axes[0].set_xticks(range(1, 5), labels, rotation=16, ha='right')
    axes[0].set_ylabel('CVAD (Layer-0 Grad-CAM)')
    axes[0].grid(axis='y', alpha=0.18)
    axes[0].text(0.02, 0.97, '(a)', transform=axes[0].transAxes, va='top', fontweight='bold')

    # (b) Paired Chinese-CLIP minus AltCLIP CVAD on Flickr8k-CN vs Flickr30k-CN
    p8 = d[d.source.isin(models[:2])].pivot(index='img_id', columns='source', values='cvad')
    delta8 = (p8['ChineseCLIP_F8k'] - p8['AltCLIP_F8k']).to_numpy()

    # Load Flickr30k-CN paired patch_gradcam CVAD (same v1 scale)
    v1_pca = pd.read_csv(RES_V1 / 'pca_fit.csv').set_index('component')
    comp = ['abs_delta_entropy', 'abs_delta_area', 'inv_iou', 'kl_symmetric']
    mn = v1_pca.loc[comp, 'minimum'].to_numpy()
    span = (v1_pca.loc[comp, 'maximum'] - v1_pca.loc[comp, 'minimum']).to_numpy()
    w = v1_pca.loc[comp, 'weight'].to_numpy()

    from audit_existing_data import map_metrics, components
    cc30 = d[d.source == 'ChineseCLIP_F30k'].set_index('img_id')
    with np.load(VAL_V2 / 'AltCLIP_F30k' / 'maps_all.npz', allow_pickle=False) as z:
        ok = z['lang_complete'][:, 0, :].all(1)
        alt_ids = z['ids'][ok]
        alt_df = components(map_metrics(z['patch_gradcam'][ok, 0], z['patch_gradcam'][ok, 1]))
        alt_df['img_id'] = alt_ids
        alt_df['cvad'] = ((alt_df[comp].to_numpy() - mn) / span) @ w
        alt30 = alt_df.set_index('img_id')
    common30 = sorted(cc30.index.intersection(alt30.index))
    delta30 = (cc30.loc[common30, 'cvad'] - alt30.loc[common30, 'cvad']).to_numpy()

    bins = np.linspace(-0.6, 0.8, 29)
    axes[1].hist(
        delta8, bins=bins, density=True, color='#0072B2', alpha=0.52,
        edgecolor='white', linewidth=0.4,
        label=f'Flickr8k-CN (n={len(delta8)}, mean={delta8.mean():.3f})',
    )
    axes[1].hist(
        delta30, bins=bins, density=True, histtype='step', color='#D55E00',
        linewidth=1.6,
        label=f'Flickr30k-CN (n={len(delta30)}, mean={delta30.mean():.3f})',
    )
    axes[1].axvline(0, color='#222222', linestyle='--', linewidth=0.9)
    axes[1].set_xlabel('Paired CVAD difference (Chinese-CLIP minus AltCLIP)')
    axes[1].set_ylabel('Density')
    axes[1].set_ylim(0, 2.25)
    axes[1].grid(axis='y', alpha=0.18)
    axes[1].legend(fontsize=7.2, frameon=False, loc='upper right')
    axes[1].text(0.02, 0.97, '(b)', transform=axes[1].transAxes, va='top', fontweight='bold')

    fig.savefig(ROOT / 'fig_model_comparison.pdf', bbox_inches='tight')
    plt.close(fig)


def overlay_heatmap(rgb_224: np.ndarray, heat_16: np.ndarray) -> np.ndarray:
    """Resize 16x16 heatmap to 224x224 (nearest/bilinear) and blend cleanly over RGB."""
    cmap = plt.get_cmap('turbo')
    # Upsample cleanly with bilinear interpolation for crisp visual inspection while preserving 16x16 peaks.
    h_img = Image.fromarray(np.clip(heat_16 * 255.0, 0, 255).astype(np.uint8))
    h_up = np.asarray(h_img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    rgb_f = rgb_224.astype(np.float32) / 255.0
    colored = cmap(h_up)[..., :3]
    alpha = 0.20 + 0.55 * h_up[..., None]
    out = (1.0 - alpha) * rgb_f + alpha * colored
    return np.clip(out, 0.0, 1.0)


def fig_qualitative():
    """Fig. 3: Qualitative EN/ZH heatmaps across 3 attribution methods on 2 CC BY 2.0 photos."""
    cases = [
        ('2764178773_d63b502812.jpg', 'Andrieu Lecondeguy'),
        ('2542662402_d781dd7f7c.jpg', 'DSC00079'),
    ]
    sp = pd.read_csv(ANA_V2 / 'f8k_spatial_per_image.csv').set_index(['source', 'method', 'img_id'])
    methods = ['patch_gradcam', 'lastblock_gradcam', 'chefer_relevance']
    sources = [('ChineseCLIP_F8k', 'Chinese-CLIP'), ('AltCLIP_F8k', 'AltCLIP')]

    fig = plt.figure(figsize=(7.15, 5.15))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.22,
                              left=0.06, right=0.985, top=0.89, bottom=0.05)
    sub_grids = [
        gridspec.GridSpecFromSubplotSpec(
            2, 7, subplot_spec=outer[0], width_ratios=[1.05, 1, 1, 1, 1, 1, 1], wspace=0.06, hspace=0.26
        ),
        gridspec.GridSpecFromSubplotSpec(
            2, 7, subplot_spec=outer[1], width_ratios=[1.05, 1, 1, 1, 1, 1, 1], wspace=0.06, hspace=0.26
        ),
    ]

    for c_idx, (img_id, short_title) in enumerate(cases):
        stem = Path(img_id).stem
        with Image.open(BASE / 'data/Flickr8k/train' / img_id) as im:
            rgb_224 = np.asarray(im.convert('RGB').resize((224, 224), Image.BICUBIC))

        for s_idx, (source, s_label) in enumerate(sources):
            row = c_idx * 2 + s_idx
            with np.load(VAL_V2 / source / 'attributions' / f'{stem}.npz') as z:
                maps = {m: z[m] for m in methods}

            ax0 = fig.add_subplot(sub_grids[c_idx][s_idx, 0])
            ax0.imshow(rgb_224)
            ax0.set_xticks([])
            ax0.set_yticks([])
            for spn in ax0.spines.values():
                spn.set_visible(True)
                spn.set_linewidth(0.5)
                spn.set_color('#444444')
            ax0.set_ylabel(s_label, fontsize=7.8, fontweight='bold', labelpad=4)
            if s_idx == 0:
                ax0.set_title(short_title, fontsize=7.5, pad=3, style='italic')

            for mi, m in enumerate(methods):
                rec = sp.loc[(source, m, img_id)]
                for li, lname in enumerate(('EN', 'ZH')):
                    col = 1 + mi * 2 + li
                    ax = fig.add_subplot(sub_grids[c_idx][s_idx, col])
                    ax.imshow(overlay_heatmap(rgb_224, maps[m][li]))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spn in ax.spines.values():
                        spn.set_visible(True)
                        spn.set_linewidth(0.5)
                        spn.set_color('#0072B2' if li == 0 else '#D55E00')
                    if row == 0:
                        ax.set_title(lname, fontsize=7.5, pad=2,
                                     color='#0072B2' if li == 0 else '#D55E00', fontweight='bold')
                ax_l = fig.axes[-2]
                ax_r = fig.axes[-1]
                pos_l = ax_l.get_position()
                pos_r = ax_r.get_position()
                xc = 0.5 * (pos_l.x0 + pos_r.x1)
                yb = pos_l.y0 - 0.008
                fig.text(
                    xc, yb,
                    f'cos={rec["cos_en_zh"]:.2f}, IoU={rec["iou"]:.2f}',
                    ha='center', va='top', fontsize=6.6, color='#222222',
                )

    pos0 = fig.axes[0].get_position()
    fig.text(0.5 * (pos0.x0 + pos0.x1), 0.945, 'Input view', ha='center', va='bottom',
             fontsize=8.0, fontweight='bold')
    for mi, m in enumerate(methods):
        al = fig.axes[1 + mi * 2].get_position()
        ar = fig.axes[2 + mi * 2].get_position()
        fig.text(0.5 * (al.x0 + ar.x1), 0.945, METHOD_SHORT[m], ha='center', va='bottom',
                 fontsize=8.0, fontweight='bold')

    p_top = fig.axes[7].get_position()
    p_bot = fig.axes[14].get_position()
    ymid = p_bot.y1 + 0.030
    fig.add_artist(plt.Line2D([0.04, 0.985], [ymid, ymid], color='#CCCCCC', lw=0.8, ls='--'))

    fig.savefig(ROOT / 'fig_qualitative.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_sensitivity():
    """Fig. 4: Fixed legend (no overlap!) + cross-method threshold rank stability."""
    sen_v1 = pd.read_csv(RES_V1 / 'map_sensitivity.csv').query('setting == "threshold"')
    sen_v2 = pd.read_csv(ANA_V2 / 'f8k_threshold_sensitivity.csv')
    models = ['ChineseCLIP_F8k', 'AltCLIP_F8k', 'OpenAI_CLIP_B32_F8k', 'OpenAI_CLIP_L14_F8k']
    names = {'ChineseCLIP_F8k': 'Chinese-CLIP', 'AltCLIP_F8k': 'AltCLIP',
             'OpenAI_CLIP_B32_F8k': 'CLIP-B/32', 'OpenAI_CLIP_L14_F8k': 'CLIP-L/14'}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), layout='constrained')

    # (a) Mean CVAD across thresholds for 4 baseline models - legend cleanly above axis
    for s, marker in zip(models, ['o', 's', '^', 'D']):
        sub = sen_v1[sen_v1.source == s].sort_values('value')
        axes[0].plot(sub.value, sub.cvad_mean, color=COLORS[s], marker=marker,
                     label=names[s], markersize=4, lw=1.4)
    axes[0].set_ylabel('Mean CVAD (Layer-0 Grad-CAM)')
    axes[0].set_xlabel('Activation threshold ($\\tau$)')
    axes[0].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axes[0].set_ylim(0.18, 0.72)
    axes[0].grid(alpha=0.16)
    axes[0].legend(fontsize=7.2, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
    axes[0].text(0.02, 0.97, '(a)', transform=axes[0].transAxes, va='top', fontweight='bold')

    # (b) Image-ranking stability across ALL 3 attribution methods (Chinese-CLIP solid, AltCLIP dashed)
    for m, marker in zip(['patch_gradcam', 'lastblock_gradcam', 'chefer_relevance'], ['o', 's', '^']):
        for source, ls, short_s in [('ChineseCLIP_F8k', '-', 'CC'), ('AltCLIP_F8k', '--', 'Alt')]:
            sub = sen_v2[(sen_v2.method == m) & (sen_v2.source == source)].sort_values('tau')
            lbl = f'{METHOD_SHORT[m]} ({short_s})'
            axes[1].plot(
                sub.tau, sub.cvad_spearman_vs_05,
                color=METHOD_COLORS[m], linestyle=ls, marker=marker,
                markersize=3.8, lw=1.35, label=lbl,
            )
    axes[1].set_ylabel('Image-rank Spearman $\\rho$ vs. $\\tau=0.5$')
    axes[1].set_xlabel('Activation threshold ($\\tau$)')
    axes[1].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axes[1].set_ylim(0.15, 1.06)
    axes[1].grid(alpha=0.16)
    axes[1].legend(fontsize=6.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False)
    axes[1].text(0.02, 0.97, '(b)', transform=axes[1].transAxes, va='top', fontweight='bold')

    fig.savefig(ROOT / 'fig_sensitivity.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_occlusion():
    """Fig. 5: 2x3 occlusion faithfulness curves across models and all 3 attribution methods."""
    # Recompute EN/ZH averaged per-image curves directly from v1/v2 per-image occlusion files
    # so bootstrap bands match the primary EN/ZH-averaged image-level analysis.
    from validation_v2_common import DEST_V1, DEST_V2, FRACTIONS, dataset_f8k, selection_f8k
    from analyze_validation_v2 import bootstrap_mean

    samples = dataset_f8k()
    selected = selection_f8k(samples)
    contrasts = pd.read_csv(ANA_V2 / 'f8k_occlusion_contrasts.csv').set_index(['source', 'method', 'comparison'])

    methods = ['patch_gradcam', 'lastblock_gradcam', 'chefer_relevance']
    sources = [('ChineseCLIP_F8k', 'Chinese-CLIP'), ('AltCLIP_F8k', 'AltCLIP')]
    cond_colors = {'top': '#D55E00', 'random': '#0072B2', 'bottom': '#009E73'}

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), sharex=True, sharey='row', layout='constrained')

    for r_idx, (source, s_label) in enumerate(sources):
        for c_idx, method in enumerate(methods):
            ax = axes[r_idx, c_idx]
            per_img = []
            for idx in selected:
                stem = Path(samples[int(idx)]['img_id']).stem
                p = (DEST_V1 / source / 'occlusion' / f'{stem}.npz'
                     if method == 'patch_gradcam'
                     else DEST_V2 / source / f'occlusion_{method}' / f'{stem}.npz')
                with np.load(p) as z:
                    drops = np.concatenate([np.zeros((2, 7, 1)), z['drop']], axis=2)
                # Reduce 5 random repeats in exact order [top, random, bottom], then average EN & ZH -> (3, 6)
                red = np.stack([drops[:, 0], drops[:, 2:].mean(1), drops[:, 1]], axis=1).mean(0)
                per_img.append(red)
            arr = np.stack(per_img, axis=0)  # (200, 3, 6)

            for ci, cname in enumerate(('top', 'random', 'bottom')):
                means = arr[:, ci, :].mean(0)
                los, his = [], []
                for fi in range(len(FRACTIONS)):
                    if fi == 0:
                        los.append(0.0); his.append(0.0)
                    else:
                        lo, hi = bootstrap_mean(arr[:, ci, fi], n=2000)
                        los.append(lo); his.append(hi)
                ax.plot(FRACTIONS * 100, means, color=cond_colors[cname],
                        marker='o', markersize=2.8, lw=1.5, label=cname.capitalize())
                ax.fill_between(FRACTIONS * 100, los, his, color=cond_colors[cname], alpha=0.14)

            ax.axhline(0, color='#666666', lw=0.6)
            ax.grid(alpha=0.15)
            tr = contrasts.loc[(source, method, 'top_minus_random')]
            tb = contrasts.loc[(source, method, 'top_minus_bottom')]
            ax.text(
                0.04, 0.95,
                f'$\\Delta U_{{\\rm T-R}}={tr["mean_difference"]:.4f}$ ($d_z={tr["cohen_dz"]:.2f}$, {100*tr["fraction_positive"]:.0f}%)\n'
                f'$\\Delta U_{{\\rm T-B}}={tb["mean_difference"]:.4f}$ ($d_z={tb["cohen_dz"]:.2f}$, {100*tb["fraction_positive"]:.0f}%)',
                transform=ax.transAxes, va='top', fontsize=6.4,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', alpha=0.84, edgecolor='#DDDDDD', lw=0.6),
            )
            if r_idx == 0:
                ax.set_title(METHOD_SHORT[method], fontsize=8.8, fontweight='bold')
            if r_idx == 1:
                ax.set_xlabel('Masked patches (%)')
            if c_idx == 0:
                ax.set_ylabel(f'{s_label}\nMean cosine drop')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.055),
               ncol=3, frameon=False, fontsize=8.0)
    fig.savefig(ROOT / 'fig_occlusion.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_retrieval_association():
    """Fig. 6: Retrieval associations across all 3 attribution methods (aligned divergence direction)."""
    d = pd.read_csv(ANA_V2 / 'f8k_retrieval_associations.csv')
    # Align cos_en_zh direction as cosine divergence (1 - cos_en_zh) -> negate rho and swap CI endpoints.
    d = d.copy()
    mask_cos = d['predictor'] == 'cos_en_zh'
    rho_old = d.loc[mask_cos, 'rho'].to_numpy()
    lo_old = d.loc[mask_cos, 'ci95_low'].to_numpy()
    hi_old = d.loc[mask_cos, 'ci95_high'].to_numpy()
    d.loc[mask_cos, 'rho'] = -rho_old
    d.loc[mask_cos, 'ci95_low'] = -hi_old
    d.loc[mask_cos, 'ci95_high'] = -lo_old

    keys = ['cvad', 'inv_iou', 'kl_symmetric', 'cos_en_zh']
    key_labels = ['CVAD', 'Inverse IoU', 'Symmetric KL', '1 − Map cosine']
    methods = ['patch_gradcam', 'lastblock_gradcam', 'chefer_relevance']
    offsets = [-0.22, 0.0, 0.22]
    markers = ['o', 's', '^']

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharex=True, sharey=True, layout='constrained')
    outcomes = [('bilingual_logrank_gap', 'Bilingual rank gap'), ('failure_at1', 'R@1 failure fraction')]
    sources = [('ChineseCLIP_F8k', 'Chinese-CLIP'), ('AltCLIP_F8k', 'AltCLIP')]

    for r_idx, (outcome, o_label) in enumerate(outcomes):
        for c_idx, (source, s_label) in enumerate(sources):
            ax = axes[r_idx, c_idx]
            ax.axvline(0, color='#555555', lw=0.75, ls='--')
            for m, off, mk in zip(methods, offsets, markers):
                sub = d[(d.source == source) & (d.outcome == outcome) & (d.method == m)].set_index('predictor').loc[keys]
                ys = np.arange(len(keys)) + off
                xerr = np.vstack([sub['rho'] - sub['ci95_low'], sub['ci95_high'] - sub['rho']])
                ax.errorbar(
                    sub['rho'], ys, xerr=xerr, fmt=mk, color=METHOD_COLORS[m],
                    markersize=4.0, capsize=2.0, lw=1.15, label=METHOD_SHORT[m],
                )
            ax.set_yticks(range(len(keys)), key_labels)
            ax.grid(axis='x', alpha=0.16)
            if r_idx == 0:
                ax.set_title(s_label, fontsize=9.0, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(o_label, fontsize=8.5, fontweight='bold')

    axes[0, 0].invert_yaxis()
    fig.supxlabel(
        'Spearman correlation $\\rho$ (positive = higher cross-language divergence $\\rightarrow$ worse retrieval parity / accuracy)',
        fontsize=8.2,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.055),
               ncol=3, frameon=False, fontsize=8.0)
    fig.savefig(ROOT / 'fig_retrieval_association.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    fig_model_comparison()
    print('Built fig_model_comparison.pdf')
    fig_qualitative()
    print('Built fig_qualitative.pdf')
    fig_sensitivity()
    print('Built fig_sensitivity.pdf')
    fig_occlusion()
    print('Built fig_occlusion.pdf')
    fig_retrieval_association()
    print('Built fig_retrieval_association.pdf')


if __name__ == '__main__':
    main()
