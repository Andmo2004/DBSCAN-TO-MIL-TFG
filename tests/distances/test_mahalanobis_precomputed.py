"""
tests/distances/test_mahalanobis_precomputed.py

Tests para la optimización de Mahalanobis con estadísticos precomputados.

Verifica que ``compute_mahalanobis_matrix`` (CPU) y
``compute_mahalanobis_matrix_torch`` (GPU/MPS si disponible) producen
resultados numéricamente idénticos al bucle par-a-par original
``mahalanobis_distance`` / ``mahalanobis_torch``.
"""

import unittest
import numpy as np

from miclustering.data.attribute import Attribute
from miclustering.data.instance import Instance
from miclustering.data.bag import Bag
from miclustering.distances.probability_distribution import (
    mahalanobis_distance,
    compute_mahalanobis_matrix,
    _bag_gaussian_stats,
    _mahalanobis_from_stats,
)



# Helpers


def _make_schema(n_features: int) -> list:
    return [Attribute(name=f"feat_{i}", attr_type="real") for i in range(n_features)]


def _make_bag(matrix: np.ndarray, bag_id: str = "bag") -> Bag:
    schema = _make_schema(matrix.shape[1])
    instances = [Instance(values=row.tolist(), schema=schema) for row in matrix]
    return Bag(bag_id=bag_id, label=0, instances=instances)


def _make_random_bags(n_bags: int, n_features: int = 4, seed: int = 42) -> list:
    """Crea bolsas sintéticas con instancias aleatorias."""
    rng = np.random.RandomState(seed)
    bags = []
    for i in range(n_bags):
        n_inst = rng.randint(5, 30)
        mat = rng.randn(n_inst, n_features)
        bags.append(_make_bag(mat, bag_id=f"bag_{i}"))
    return bags


def _pairwise_mahalanobis_matrix(bags: list) -> np.ndarray:
    """Calcula la matriz Mahalanobis con el método original par-a-par."""
    n = len(bags)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = mahalanobis_distance(bags[i], bags[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix



# Tests: funciones auxiliares


class TestBagGaussianStats(unittest.TestCase):
    """Tests para _bag_gaussian_stats."""

    def test_empty_bag_returns_none(self):
        bag = Bag(bag_id="empty", label=0, instances=[])
        mu, cov = _bag_gaussian_stats(bag)
        self.assertIsNone(mu)
        self.assertIsNone(cov)

    def test_singleton_bag_uses_identity_cov(self):
        """Una bolsa con 1 instancia no puede calcular covarianza → I."""
        bag = _make_bag(np.array([[1.0, 2.0, 3.0]]))
        mu, cov = _bag_gaussian_stats(bag)
        self.assertIsNotNone(mu)
        self.assertIsNotNone(cov)
        assert mu is not None and cov is not None
        np.testing.assert_array_almost_equal(mu, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(cov, np.eye(3))

    def test_multi_instance_bag(self):
        """Covarianza coincide con np.cov para bolsas con ≥2 instancias."""
        mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bag = _make_bag(mat)
        mu, cov = _bag_gaussian_stats(bag)
        self.assertIsNotNone(mu)
        self.assertIsNotNone(cov)
        assert mu is not None and cov is not None
        np.testing.assert_array_almost_equal(mu, np.mean(mat, axis=0))
        np.testing.assert_array_almost_equal(cov, np.cov(mat, rowvar=False))


class TestMahalanobisFromStats(unittest.TestCase):
    """Tests para _mahalanobis_from_stats."""

    def test_none_stats_returns_inf(self):
        self.assertEqual(_mahalanobis_from_stats(None, None, np.array([1.0]), np.eye(1)), float('inf'))
        self.assertEqual(_mahalanobis_from_stats(np.array([1.0]), np.eye(1), None, None), float('inf'))

    def test_identical_stats_returns_zero(self):
        mu = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        d = _mahalanobis_from_stats(mu, cov, mu, cov)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_matches_original_function(self):
        """Comparar con mahalanobis_distance para un par concreto."""
        mat1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        mat2 = np.array([[3.0, 3.0], [4.0, 4.0], [3.5, 3.5]])
        bag1 = _make_bag(mat1)
        bag2 = _make_bag(mat2)

        expected = mahalanobis_distance(bag1, bag2)

        mu1, cov1 = _bag_gaussian_stats(bag1)
        mu2, cov2 = _bag_gaussian_stats(bag2)
        actual = _mahalanobis_from_stats(mu1, cov1, mu2, cov2)

        self.assertAlmostEqual(actual, expected, places=10)



# Tests: equivalencia numérica de la matriz completa (CPU)


class TestComputeMahalanobisMatrix(unittest.TestCase):
    """Verifica que compute_mahalanobis_matrix produce resultados idénticos
    al bucle par-a-par con mahalanobis_distance."""

    def test_equivalence_small(self):
        """Equivalencia con 10 bolsas, 4 features."""
        bags = _make_random_bags(10, n_features=4, seed=42)
        matrix_old = _pairwise_mahalanobis_matrix(bags)
        matrix_new = compute_mahalanobis_matrix(bags)
        np.testing.assert_allclose(matrix_new, matrix_old, atol=1e-10)

    def test_equivalence_medium(self):
        """Equivalencia con 30 bolsas, 8 features."""
        bags = _make_random_bags(30, n_features=8, seed=99)
        matrix_old = _pairwise_mahalanobis_matrix(bags)
        matrix_new = compute_mahalanobis_matrix(bags)
        np.testing.assert_allclose(matrix_new, matrix_old, atol=1e-10)

    def test_symmetry(self):
        bags = _make_random_bags(15, seed=7)
        matrix = compute_mahalanobis_matrix(bags)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)

    def test_diagonal_zero(self):
        bags = _make_random_bags(15, seed=7)
        matrix = compute_mahalanobis_matrix(bags)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)

    def test_single_bag(self):
        bags = _make_random_bags(1)
        matrix = compute_mahalanobis_matrix(bags)
        self.assertEqual(matrix.shape, (1, 1))
        self.assertEqual(matrix[0, 0], 0.0)

    def test_empty_list(self):
        matrix = compute_mahalanobis_matrix([])
        self.assertEqual(matrix.shape, (0, 0))

    def test_non_negative(self):
        bags = _make_random_bags(20, seed=123)
        matrix = compute_mahalanobis_matrix(bags)
        self.assertTrue(np.all(matrix >= 0))



# Tests: equivalencia GPU (torch backend) — skip si no disponible


class TestComputeMahalanobisMatrixTorch(unittest.TestCase):
    """Verifica compute_mahalanobis_matrix_torch contra el resultado CPU."""

    @classmethod
    def setUpClass(cls):
        try:
            from miclustering.distances.torch_backend import (
                is_torch_available,
                compute_mahalanobis_matrix_torch,
            )
            cls._torch_available = is_torch_available()
            cls._compute_fn = staticmethod(compute_mahalanobis_matrix_torch)
        except ImportError:
            cls._torch_available = False
            cls._compute_fn = None

    def test_equivalence_torch_vs_cpu(self):
        """Resultado GPU/MPS debe coincidir con CPU (tolerancia float32)."""
        if not self._torch_available or self._compute_fn is None:
            self.skipTest("PyTorch no disponible")

        fn = self._compute_fn
        bags = _make_random_bags(15, n_features=4, seed=42)
        matrix_cpu = compute_mahalanobis_matrix(bags)
        matrix_torch = fn(bags, device="cpu")
        # float32 en MPS tiene menor precisión, usar atol más amplio
        np.testing.assert_allclose(matrix_torch, matrix_cpu, atol=1e-6)

    def test_symmetry_torch(self):
        if not self._torch_available or self._compute_fn is None:
            self.skipTest("PyTorch no disponible")

        fn = self._compute_fn
        bags = _make_random_bags(10, seed=99)
        matrix = fn(bags, device="cpu")
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)

    def test_diagonal_zero_torch(self):
        if not self._torch_available or self._compute_fn is None:
            self.skipTest("PyTorch no disponible")

        fn = self._compute_fn
        bags = _make_random_bags(10, seed=99)
        matrix = fn(bags, device="cpu")
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
