"""Pure-CPU synthetic and integrity unit tests for MVA v2 pipeline."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'

import unittest
import numpy as np
import torch

from validation_v2_common import (
    SEED,
    FRACTIONS,
    MASK_COUNTS,
    mask_pixels,
    normalize_map,
    outcome_arrays,
    patch_order,
)
from analyze_validation_v2 import cv_predict_5, paired_stats


class ValidationV2Tests(unittest.TestCase):
    def test_normalize_map_validation(self):
        m, ok, err = normalize_map(np.zeros((16, 16), dtype=np.float32))
        self.assertFalse(ok)
        self.assertEqual(err, 'constant_map')
        self.assertTrue(np.isnan(m).all())

        bad = np.arange(256, dtype=np.float32).reshape(16, 16)
        bad[0, 0] = np.nan
        m, ok, err = normalize_map(bad)
        self.assertFalse(ok)
        self.assertEqual(err, 'non_finite')

        good = np.arange(256, dtype=np.float32).reshape(16, 16) * 2.5 - 10.0
        m, ok, err = normalize_map(good)
        self.assertTrue(ok)
        self.assertEqual(err, '')
        self.assertAlmostEqual(float(m.min()), 0.0, places=6)
        self.assertAlmostEqual(float(m.max()), 1.0, places=6)

    def test_chefer_rollout_update_equation(self):
        # 3 tokens (CLS + 2 patches), 2 heads, 2 layers
        # Layer 0: positive grad only on head 0, token 1 -> token 2
        # Layer 1: positive grad on CLS -> token 1 (so 2-hop CLS -> 1 -> 2 must receive relevance)
        T = 3
        a0 = torch.tensor([[[
            [0.5, 0.5, 0.0],
            [0.0, 0.2, 0.8],
            [0.0, 0.0, 1.0],
        ], [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]]], dtype=torch.float32)  # (1, 2, 3, 3)
        g0 = torch.tensor([[[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ], [
            [-5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
        ]]], dtype=torch.float32)

        a1 = torch.tensor([[[
            [0.2, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], [
            [0.2, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]]], dtype=torch.float32)
        g1 = torch.tensor([[[
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]]], dtype=torch.float32)

        R = torch.eye(T, dtype=torch.float32)
        for a, g in ((a0, g0), (a1, g1)):
            cam = (g * a).clamp(min=0).mean(dim=1).squeeze(0)
            R = R + torch.matmul(cam, R)
        # Layer 0: cam0[1,2] = mean(1.6, 0) = 0.8 -> R0[1,2] = 0.8
        # Layer 1: cam1[0,1] = 0.8 -> R1[0,1] = 0.8, R1[0,2] = 0.8 * 0.8 = 0.64
        self.assertAlmostEqual(float(R[0, 1]), 0.8, places=6)
        self.assertAlmostEqual(float(R[0, 2]), 0.64, places=6)

    def test_cv_predict_5_no_target_leakage(self):
        rng = np.random.default_rng(SEED)
        X = rng.uniform(0.1, 2.0, size=(60, 4))
        cos_vec = rng.uniform(0.2, 0.99, size=60)
        y = rng.normal(size=60)
        folds = np.array_split(rng.permutation(60), 5)
        before = cv_predict_5(X, cos_vec, y, folds)
        altered = y.copy()
        altered[folds[0]] += 100.0
        after = cv_predict_5(X, cos_vec, altered, folds)
        for key in before:
            np.testing.assert_allclose(before[key][folds[0]], after[key][folds[0]], rtol=0, atol=0)

    def test_mask_and_ranks_consistency(self):
        x = torch.ones(1, 3, 224, 224)
        order = patch_order(np.arange(256).reshape(16, 16), SEED, high=True)
        prev = torch.zeros(224, 224, dtype=torch.bool)
        for k in MASK_COUNTS:
            m = mask_pixels(x, order, k)
            changed = (m[0] == 0).all(0)
            self.assertEqual(int(changed.sum()), k * 14 * 14)
            self.assertTrue(torch.all(changed[prev]))
            prev = changed

        image = np.eye(4, dtype=np.float32)
        text = np.tile(image[:, None, None, :], (1, 2, 5, 1))
        _, t2i, i2t = outcome_arrays(image, text)
        np.testing.assert_array_equal(t2i, np.ones((4, 2, 5)))
        np.testing.assert_array_equal(i2t, np.ones((4, 2)))

    def test_paired_stats_zero_and_positive(self):
        z = paired_stats(np.zeros(20))
        self.assertEqual(z['p_raw'], 1.0)
        self.assertEqual(z['mean_difference'], 0.0)
        pos = paired_stats(np.linspace(0.1, 1.0, 20))
        self.assertEqual(pos['fraction_positive'], 1.0)
        self.assertLess(pos['p_raw'], 0.001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
