# Bilingual Visual Attribution in Vision Language Models

Official code, evaluation protocol, and numerical results for:

> **Evaluating Bilingual Visual Attribution and Faithfulness in Vision Language Models for Educational Applications**

## Overview

This repository implements an image-paired evaluation protocol linking cross-language spatial attribution consistency, patch-occlusion faithfulness, and bilingual image-text retrieval parity across **Flickr8k-CN** ($n=1000$) and **Flickr30k-CN** ($n=975$), supporting pre-deployment auditing of bilingual vision-language models for educational and digital-library applications.

We compare four contrastive checkpoints (`OFA-Sys/chinese-clip-vit-large-patch14`, `BAAI/AltCLIP`, `openai/clip-vit-base-patch32`, and `openai/clip-vit-large-patch14`) and three text-conditioned Vision Transformer (ViT) attribution mechanisms:

1. **Layer-0 Patch-Embedding Grad-CAM** (`patch_gradcam`)
2. **Layer-24 Final-Block Grad-CAM** (`lastblock_gradcam`, hooked at `encoder.layers[-1].layer_norm1`)
3. **All-Layer Gradient-Attention Relevance** (`chefer_relevance`, Chefer et al., ICCV 2021)

## Key Empirical Results

### Cross-Method Spatial Consistency (Chinese-CLIP vs. AltCLIP)

| Dataset | Attribution Method | $n$ | Map Cosine (CC / Alt) | $\Delta$Cos ($d_z$) | Alt > CC |
| :--- | :--- | ---: | :---: | :---: | ---: |
| Flickr8k-CN | Layer-0 Grad-CAM | 998 | 0.807 / 0.963 | -0.156 (-0.98) | 90.9% |
| Flickr8k-CN | Layer-24 Grad-CAM | 1000 | 0.860 / 0.975 | -0.114 (-1.17) | 98.4% |
| Flickr8k-CN | Chefer relevance | 1000 | 0.981 / 0.997 | -0.016 (-1.07) | 98.1% |
| Flickr30k-CN | Layer-0 Grad-CAM | 972 | 0.815 / 0.962 | -0.148 (-0.95) | 90.9% |
| Flickr30k-CN | Layer-24 Grad-CAM | 975 | 0.873 / 0.976 | -0.102 (-1.17) | 97.2% |
| Flickr30k-CN | Chefer relevance | 975 | 0.981 / 0.997 | -0.016 (-1.01) | 97.7% |

### Patch-Occlusion Faithfulness (200 Shared Flickr8k-CN Images)

| Model | Method | $\Delta U_{\text{T-R}}$ | Cohen's $d_z$ | Positive Images | $\Delta U_{\text{T-B}}$ ($d_z$) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| Chinese-CLIP | Layer-0 Grad-CAM | 0.0026 | 0.24 | 57.0% | 0.0022 (0.13) |
| Chinese-CLIP | Layer-24 Grad-CAM | 0.0224 | 1.11 | 93.0% | 0.0307 (1.13) |
| Chinese-CLIP | Chefer relevance | 0.0354 | 1.49 | 97.5% | 0.0446 (1.52) |
| AltCLIP | Layer-0 Grad-CAM | 0.0013 | 0.11 | 50.5% | 0.0002 (0.01) |
| AltCLIP | Layer-24 Grad-CAM | 0.0208 | 1.07 | 92.0% | 0.0306 (1.30) |
| AltCLIP | Chefer relevance | 0.0283 | 1.24 | 91.5% | 0.0398 (1.43) |

## Repository Structure

```text
.
├── protocol.txt                        # Full evaluation protocol and hyperparameters
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── src/                                # Core loaders, Grad-CAM, Attention Rollout, and spatial metrics
├── scripts/                            # Multi-method ViT inference, analysis, unit tests, and verification
├── results/
│   ├── baseline_layer0/                # 4-checkpoint Layer-0 baseline, controls, and ablations (CSV)
│   └── cross_method_and_replication/   # 2x2x3 spatial, occlusion, sensitivity, and retrieval tables (CSV)
└── figures/                            # CC BY 2.0 photograph license provenance for Figs. 1 and 3
```

## Quick Verification (CPU Only)

All published headline numbers can be verified directly from the included CSV tables without datasets, pretrained weights, or a GPU:

```bash
pip install numpy pandas
python scripts/verify_public_results.py
```

## Datasets and Licensing

Raw Flickr8k-CN and Flickr30k-CN photographs and pretrained model weights are not redistributed in this repository. Please obtain the source datasets from their original authors:

- **Flickr8k-CN**: Li et al., *Adding Chinese captions to images*, ACM ICMR 2016 (`10.1145/2911996.2912049`)
- **Flickr30k-CN**: Lan et al., *Fluency-Guided Cross-Lingual Image Captioning*, ACM MM 2017 (`10.1145/3123266.3123366`)

Code in this repository is released under the MIT License (see `LICENSE`). License records for the five CC BY 2.0 Flickr photographs displayed in manuscript Figs. 1 and 3 are provided in `figures/LICENSE_PHOTOGRAPHS.txt` and `figures/photo_license.json`.
