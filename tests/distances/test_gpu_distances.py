"""
tests/distances/test_gpu_distances.py

Tests unitarios para la aceleración por hardware (PyTorch / GPU / MPS / CPU) de
métricas de distancia y matrices completas.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import unittest

from miclustering.distances.torch_backend import (
    is_torch_available,
    get_torch_device,
    hausdorff_torch,
    cauchy_schwarz_torch,
    mahalanobis_torch,
    sinkhorn_emd_torch,
    compute_distance_matrix_torch,
)
from miclustering.distances.hausdorff import (
    hausdorff_distance,
    hausdorff_distance_min,
    hausdorff_distance_avg,
)
from miclustering.distances.probability_distribution import (
    cauchy_schwarz_distance,
    mahalanobis_distance,
)
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.midata import MIData


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


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestTorchAvailabilityAndDevice:
    def test_torch_is_available(self):
        assert is_torch_available() is True

    def test_get_device_auto_returns_valid_device(self):
        dev = get_torch_device("auto")
        assert dev is not None
        assert dev.type in {"cuda", "mps", "cpu"}

    def test_get_device_cpu_returns_cpu(self):
        dev = get_torch_device("cpu")
        assert dev is not None
        assert dev.type == "cpu"


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestHausdorffGPUvsCPU:
    def test_hausdorff_max_matches_cpu(self):
        bag1 = _make_bag_custom("b1", 1, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bag2 = _make_bag_custom("b2", 0, [[2.0, 2.5], [4.0, 4.5]])

        cpu_val = hausdorff_distance(bag1, bag2)
        gpu_val = hausdorff_torch(bag1.as_matrix(), bag2.as_matrix(), mode="max")

        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-5)

    def test_hausdorff_min_matches_cpu(self):
        bag1 = _make_bag_custom("b1", 1, [[1.0, 2.0], [3.0, 4.0]])
        bag2 = _make_bag_custom("b2", 0, [[2.0, 2.5], [4.0, 4.5]])

        cpu_val = hausdorff_distance_min(bag1, bag2)
        gpu_val = hausdorff_torch(bag1.as_matrix(), bag2.as_matrix(), mode="min")

        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-5)

    def test_hausdorff_avg_matches_cpu(self):
        bag1 = _make_bag_custom("b1", 1, [[1.0, 2.0], [3.0, 4.0]])
        bag2 = _make_bag_custom("b2", 0, [[2.0, 2.5], [4.0, 4.5]])

        cpu_val = hausdorff_distance_avg(bag1, bag2)
        gpu_val = hausdorff_torch(bag1.as_matrix(), bag2.as_matrix(), mode="avg")

        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-5)

    def test_empty_bag_returns_inf(self):
        val = hausdorff_torch(np.empty((0, 2)), np.array([[1.0, 2.0]]))
        assert np.isinf(val)


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestCauchySchwarzGPUvsCPU:
    def test_cauchy_schwarz_matches_cpu(self):
        bag1 = _make_bag_custom("b1", 1, [[1.0, 2.0], [3.0, 4.0]])
        bag2 = _make_bag_custom("b2", 0, [[2.0, 3.0], [4.0, 5.0]])

        cpu_val = cauchy_schwarz_distance(bag1, bag2)
        gpu_val = cauchy_schwarz_torch(bag1.as_matrix(), bag2.as_matrix())

        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-5)

    def test_cauchy_schwarz_empty_returns_inf(self):
        val = cauchy_schwarz_torch(np.empty((0, 2)), np.array([[1.0, 2.0]]))
        assert np.isinf(val)


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestMahalanobisGPUvsCPU:
    def test_mahalanobis_matches_cpu(self):
        rng = np.random.RandomState(42)
        mat1 = rng.randn(10, 4) + 2.0
        mat2 = rng.randn(10, 4) - 1.0

        bag1 = _make_bag_custom("b1", 1, mat1.tolist())
        bag2 = _make_bag_custom("b2", 0, mat2.tolist())

        cpu_val = mahalanobis_distance(bag1, bag2)
        gpu_val = mahalanobis_torch(mat1, mat2)

        np.testing.assert_allclose(cpu_val, gpu_val, atol=1e-3)


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestSinkhornEMDGPU:
    def test_sinkhorn_identical_is_near_zero(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        val = sinkhorn_emd_torch(mat, mat)
        assert val >= 0.0
        assert val < 0.5

    def test_sinkhorn_distinct_is_positive(self):
        mat1 = np.array([[0.0, 0.0]])
        mat2 = np.array([[10.0, 10.0]])
        val = sinkhorn_emd_torch(mat1, mat2)
        assert val > 10.0


@unittest.skipIf(not is_torch_available(), "PyTorch no está instalado")
class TestComputeDistanceMatrixGPU:
    def test_matrix_gpu_matches_cpu_hausdorff(self):
        dataset = _make_binary_dataset(n_pos=8, n_neg=8)
        bags = dataset.bags

        m_cpu = compute_distance_matrix(bags, hausdorff_distance, metric_name="hausdorff", device="cpu")
        m_gpu = compute_distance_matrix(bags, hausdorff_distance, metric_name="hausdorff", device="auto")

        assert m_gpu.shape == (16, 16)
        np.testing.assert_allclose(m_cpu, m_gpu, atol=1e-5)
        np.testing.assert_allclose(m_gpu, m_gpu.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(m_gpu), np.zeros(16), atol=1e-12)

    def test_matrix_gpu_all_supported_metrics(self):
        dataset = _make_binary_dataset(n_pos=4, n_neg=4)
        bags = dataset.bags

        for metric in ["hausdorff", "hausdorff_min", "hausdorff_avg", "cauchy_schwarz", "mahalanobis", "earth_movers"]:
            m = compute_distance_matrix_torch(bags, metric_name=metric, device="auto")
            assert m.shape == (8, 8)
            assert np.all(np.isfinite(m))
            np.testing.assert_allclose(m, m.T, atol=1e-10)

    def test_compute_mahalanobis_matrix_torch_direct(self):
        from miclustering.distances.torch_backend import compute_mahalanobis_matrix_torch
        dataset = _make_binary_dataset(n_pos=5, n_neg=5)
        bags = dataset.bags

        m = compute_mahalanobis_matrix_torch(bags, device="auto")
        assert m.shape == (10, 10)
        np.testing.assert_allclose(m, m.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(m), np.zeros(10), atol=1e-10)
