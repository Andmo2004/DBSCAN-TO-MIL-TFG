"""
tests/distances/test_distance_matrix.py

Tests unitarios para `compute_distance_matrix` y paralelización con joblib.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances.hausdorff import hausdorff_distance
from miclustering.data.midata import MIData
from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.miknn import MIKnn
from tests.models.conftest import _make_binary_dataset, _make_bag_custom


class TestDistanceMatrixComputation:
    def test_empty_bags_returns_empty_matrix(self):
        m = compute_distance_matrix([], hausdorff_distance, n_jobs=1)
        assert m.shape == (0, 0)

    def test_single_bag_returns_1x1_zero(self):
        bag = _make_bag_custom("b0", 0, [[1.0, 2.0]])
        m = compute_distance_matrix([bag], hausdorff_distance, n_jobs=1)
        assert m.shape == (1, 1)
        assert m[0, 0] == 0.0

    def test_symmetry_and_diagonal_zeros(self):
        dataset = _make_binary_dataset(n_pos=10, n_neg=10)
        bags = dataset.bags
        m = compute_distance_matrix(bags, hausdorff_distance, n_jobs=1)

        assert m.shape == (20, 20)
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m), np.zeros(20), atol=1e-12)

    def test_parallel_matches_sequential_exact(self):
        # 25 bolsas para superar el umbral N > 15 y activar joblib
        dataset = _make_binary_dataset(n_pos=13, n_neg=12)
        bags = dataset.bags

        m_seq = compute_distance_matrix(bags, hausdorff_distance, n_jobs=1)
        m_par2 = compute_distance_matrix(bags, hausdorff_distance, n_jobs=2)
        m_par_all = compute_distance_matrix(bags, hausdorff_distance, n_jobs=-1)

        np.testing.assert_allclose(m_seq, m_par2, atol=1e-12)
        np.testing.assert_allclose(m_seq, m_par_all, atol=1e-12)


class TestModelParallelPredict:
    def test_midbscan_parallel_predict_matches_sequential(self):
        dataset = _make_binary_dataset(n_pos=10, n_neg=10)
        test_dataset = _make_binary_dataset(n_pos=8, n_neg=8)

        # n_jobs = 1
        model_seq = MIDBSCAN(epsilon=2.0, min_pts=2, n_jobs=1)
        model_seq.fit(dataset)
        preds_seq = model_seq.predict(test_dataset)

        # n_jobs = -1
        model_par = MIDBSCAN(epsilon=2.0, min_pts=2, n_jobs=-1)
        model_par.fit(dataset)
        preds_par = model_par.predict(test_dataset)

        assert preds_seq == preds_par

    def test_miknn_parallel_predict_matches_sequential(self):
        dataset = _make_binary_dataset(n_pos=10, n_neg=10)
        test_dataset = _make_binary_dataset(n_pos=8, n_neg=8)

        # n_jobs = 1
        model_seq = MIKnn(k=3, n_jobs=1)
        model_seq.fit(dataset)
        preds_seq = model_seq.predict(test_dataset)

        # n_jobs = -1
        model_par = MIKnn(k=3, n_jobs=-1)
        model_par.fit(dataset)
        preds_par = model_par.predict(test_dataset)

        assert preds_seq == preds_par
