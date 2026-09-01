"""
tests/distances/test_distance_matrix.py

Tests unitarios para `compute_distance_matrix` y paralelización con joblib.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances.hausdorff import hausdorff_distance
from miclustering.data.midata import MIData
from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.miknn import MIKnn
from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance


def _make_bag_custom(bag_id: str, label: int, matrix: list[list[float]]) -> Bag:
    schema = [Attribute(f"f{i}", "real") for i in range(len(matrix[0]))]
    insts = [Instance(list(row), schema) for row in matrix]
    return Bag(bag_id=bag_id, label=str(label), instances=insts)


def _make_binary_dataset(n_pos: int = 10, n_neg: int = 10, seed: int = 0) -> MIData:
    rng = np.random.RandomState(seed)
    schema = [Attribute(f"f{i}", "real") for i in range(4)]
    bags = []
    for i in range(n_pos):
        mat = (rng.rand(5, 4) + 2.0).tolist()
        bags.append(Bag(f"pos_{i}", "1", [Instance(r, schema) for r in mat]))
    for i in range(n_neg):
        mat = rng.rand(5, 4).tolist()
        bags.append(Bag(f"neg_{i}", "0", [Instance(r, schema) for r in mat]))
    return MIData(bags, "synthetic")


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

    def test_mahalanobis_dispatch_matches_direct(self):
        from miclustering.distances.probability_distribution import (
            mahalanobis_distance,
            compute_mahalanobis_matrix,
        )
        dataset = _make_binary_dataset(n_pos=5, n_neg=5)
        bags = dataset.bags

        m_dispatch = compute_distance_matrix(
            bags, mahalanobis_distance, metric_name="mahalanobis", device="cpu"
        )
        m_direct = compute_mahalanobis_matrix(bags)

        assert m_dispatch.shape == (10, 10)
        np.testing.assert_allclose(m_dispatch, m_direct, atol=1e-10)
        np.testing.assert_allclose(m_dispatch, m_dispatch.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m_dispatch), np.zeros(10), atol=1e-12)

    def test_earth_movers_matrix_computation(self):
        from miclustering.distances.probability_distribution import earth_movers_distance
        dataset = _make_binary_dataset(n_pos=4, n_neg=4)
        bags = dataset.bags

        m = compute_distance_matrix(
            bags, earth_movers_distance, metric_name="earth_movers", device="cpu"
        )
        assert m.shape == (8, 8)
        assert np.all(m >= 0.0)
        np.testing.assert_allclose(m, m.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(m), np.zeros(8), atol=1e-10)


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
