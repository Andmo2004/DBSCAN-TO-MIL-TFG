"""
tests/test_probability_distribution.py

Tests unitarios para las tres distancias basadas en distribuciones de probabilidad:
  - cauchy_schwarz_distance   (ec. basada en similitud coseno)
  - earth_movers_distance     (EMD, ec. 3.27)
  - mahalanobis_distance      (ec. 3.28)

Cada test incluye el cálculo manual esperado en el docstring.
"""

import os
import sys
import math
import unittest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# pyrefly: ignore [missing-import]
from miclustering.distances.probability_distribution import (
    cauchy_schwarz_distance,
    earth_movers_distance,
    mahalanobis_distance,
)

from miclustering.data.bag import Bag # pyrefly: ignore [missing-import]
from miclustering.data.instance import Instance # pyrefly: ignore [missing-import]
from miclustering.data.attribute import Attribute # pyrefly: ignore [missing-import]s

# ── Helper ────────────────────────────────────────────────────────────────────

def make_schema(n_features: int) -> list:
    return [Attribute(name=f"feat_{i}", attr_type="real") for i in range(n_features)]

def make_bag(matrix, bag_id="bag"):
    schema = make_schema(matrix.shape[1])
    instances = [Instance(values=row.tolist(), schema=schema) for row in matrix]
    return Bag(bag_id=bag_id, label=0, instances=instances)

def make_empty_bag(bag_id: str = "empty") -> Bag:
    """Crea una Bag vacía (sin instancias)."""
    return Bag(bag_id=bag_id, label=0, instances=[])


# ══════════════════════════════════════════════════════════════════════════════
# Tests compartidos por las tres distancias
# ══════════════════════════════════════════════════════════════════════════════

class TestProbDistCommonProperties(unittest.TestCase):
    """Propiedades que deben cumplir las tres distancias."""

    def _all_funcs(self):
        return [cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance]

    def test_identical_bags_return_zero(self):
        """
        d(A, A) = 0 para las tres distancias.
        Bolsa: [[1, 2], [3, 4]]
        """
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(A, A), 0.0, places=5)

    def test_non_negative(self):
        """Toda distancia debe ser >= 0."""
        A = make_bag(np.array([[0.0, 0.0], [1.0, 1.0]]))
        B = make_bag(np.array([[5.0, 5.0], [6.0, 6.0]]))
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertGreaterEqual(fn(A, B), 0.0)

    def test_empty_bag_returns_inf(self):
        """Una bolsa vacía debe devolver inf."""
        A = make_bag(np.array([[1.0, 2.0]]))
        empty = make_empty_bag()
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertEqual(fn(A, empty), float('inf'))
                self.assertEqual(fn(empty, A), float('inf'))

    def test_both_empty_returns_inf(self):
        """Dos bolsas vacías → inf."""
        e1, e2 = make_empty_bag("e1"), make_empty_bag("e2")
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertEqual(fn(e1, e2), float('inf'))


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: cauchy_schwarz_distance
# ══════════════════════════════════════════════════════════════════════════════

class TestCauchySchwarz(unittest.TestCase):

    def test_identical_bags(self):
        """
        A = [[1, 0], [0, 1]]
        vec_A = mean = [0.5, 0.5]
        cos(vec_A, vec_A) = 1  → d = 1 - 1 = 0
        """
        A = make_bag(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(cauchy_schwarz_distance(A, A), 0.0, places=10)

    def test_orthogonal_bags(self):
        """
        A = [[1, 0]] → vec_A = [1, 0]
        B = [[0, 1]] → vec_B = [0, 1]
        dot = 0, norms = 1 → cos = 0 → d = 1 - 0 = 1
        """
        A = make_bag(np.array([[1.0, 0.0]]))
        B = make_bag(np.array([[0.0, 1.0]]))
        self.assertAlmostEqual(cauchy_schwarz_distance(A, B), 1.0, places=10)

    def test_opposite_bags(self):
        """
        A = [[1, 0]] → vec_A = [1, 0]
        B = [[-1, 0]] → vec_B = [-1, 0]
        dot = -1, norms = 1 → cos = -1 → d = 1 - (-1) = 2
        """
        A = make_bag(np.array([[1.0, 0.0]]))
        B = make_bag(np.array([[-1.0, 0.0]]))
        self.assertAlmostEqual(cauchy_schwarz_distance(A, B), 2.0, places=10)

    def test_parallel_same_direction(self):
        """
        A = [[2, 0]] → vec = [2, 0]
        B = [[4, 0]] → vec = [4, 0]
        dot = 8, norm1 = 2, norm2 = 4 → cos = 8/8 = 1 → d = 0
        Vectores paralelos misma dirección → distancia 0.
        """
        A = make_bag(np.array([[2.0, 0.0]]))
        B = make_bag(np.array([[4.0, 0.0]]))
        self.assertAlmostEqual(cauchy_schwarz_distance(A, B), 0.0, places=10)

    def test_symmetry(self):
        """d(A, B) == d(B, A)."""
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        B = make_bag(np.array([[5.0, 6.0], [7.0, 8.0]]))
        self.assertAlmostEqual(
            cauchy_schwarz_distance(A, B),
            cauchy_schwarz_distance(B, A),
            places=10,
        )

    def test_manual_45_degrees(self):
        """
        A = [[1, 0]] → vec = [1, 0]
        B = [[1, 1]] → vec = [1, 1]
        dot = 1, norm1 = 1, norm2 = √2
        cos = 1/√2 ≈ 0.7071
        d = 1 - 1/√2 ≈ 0.2929
        """
        A = make_bag(np.array([[1.0, 0.0]]))
        B = make_bag(np.array([[1.0, 1.0]]))
        expected = 1.0 - 1.0 / math.sqrt(2)
        self.assertAlmostEqual(cauchy_schwarz_distance(A, B), expected, places=10)

    def test_multi_instance_uses_mean(self):
        """
        A = [[2, 0], [0, 2]] → vec_A = mean = [1, 1]
        B = [[1, 0]]         → vec_B = [1, 0]
        dot = 1, norm_A = √2, norm_B = 1
        cos = 1/√2  → d = 1 - 1/√2 ≈ 0.2929
        """
        A = make_bag(np.array([[2.0, 0.0], [0.0, 2.0]]))
        B = make_bag(np.array([[1.0, 0.0]]))
        expected = 1.0 - 1.0 / math.sqrt(2)
        self.assertAlmostEqual(cauchy_schwarz_distance(A, B), expected, places=10)

    def test_zero_vector_returns_inf(self):
        """
        A = [[0, 0]] → vec = [0, 0], norm = 0 → devuelve inf.
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[1.0, 1.0]]))
        self.assertEqual(cauchy_schwarz_distance(A, B), float('inf'))

    def test_range_bounded(self):
        """La distancia Cauchy-Schwarz debe estar en [0, 2]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            A = make_bag(rng.standard_normal((3, 4)))
            B = make_bag(rng.standard_normal((3, 4)))
            d = cauchy_schwarz_distance(A, B)
            if math.isfinite(d):
                self.assertGreaterEqual(d, -1e-10)
                self.assertLessEqual(d, 2.0 + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: earth_movers_distance
# ══════════════════════════════════════════════════════════════════════════════

class TestEarthMoversDistance(unittest.TestCase):

    def test_single_instance_each(self):
        """
        A = [[0, 0]], B = [[3, 4]]
        Masas uniformes: w_A = 1, w_B = 1.
        Flujo óptimo: f(A0, B0) = 1.
        EMD = 1 * d(A0, B0) = sqrt(9+16) = 5.
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(earth_movers_distance(A, B), 5.0, places=5)

    def test_identical_bags(self):
        """
        A = B = [[1, 0], [0, 1]]
        Flujo óptimo: cada instancia se mapea a sí misma con coste 0.
        EMD = 0.
        """
        A = make_bag(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(earth_movers_distance(A, A), 0.0, places=5)

    def test_symmetry(self):
        """EMD(A, B) == EMD(B, A)."""
        A = make_bag(np.array([[0.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[3.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(
            earth_movers_distance(A, B),
            earth_movers_distance(B, A),
            places=5,
        )

    def test_manual_2x2_aligned(self):
        """
        A = [[0, 0], [4, 0]]  (masas 1/2 cada una)
        B = [[1, 0], [3, 0]]  (masas 1/2 cada una)

        Distancias:
          d(A0,B0)=1  d(A0,B1)=3
          d(A1,B0)=3  d(A1,B1)=1

        Flujo óptimo: f(A0,B0)=0.5, f(A1,B1)=0.5
        EMD = 0.5*1 + 0.5*1 = 1.0
        """
        A = make_bag(np.array([[0.0, 0.0], [4.0, 0.0]]))
        B = make_bag(np.array([[1.0, 0.0], [3.0, 0.0]]))
        self.assertAlmostEqual(earth_movers_distance(A, B), 1.0, places=5)

    def test_non_negative(self):
        """EMD siempre >= 0."""
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        B = make_bag(np.array([[5.0, 6.0], [7.0, 8.0]]))
        self.assertGreaterEqual(earth_movers_distance(A, B), 0.0)

    def test_translation(self):
        """
        A = [[0, 0]], B = [[d, 0]]
        EMD con una instancia cada una = d.
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[7.0, 0.0]]))
        self.assertAlmostEqual(earth_movers_distance(A, B), 7.0, places=5)

    def test_asymmetric_sizes(self):
        """
        A = [[0, 0]]  (n_a=1, masa=1)
        B = [[2, 0], [4, 0]]  (n_b=2, masa=1/2 cada una)

        Flujo: f(A0,B0) + f(A0,B1) = 1 (restricción igualdad)
               f(A0,B0) <= 1/2, f(A0,B1) <= 1/2
               f(A0,B0) <= 1, f(A0,B1) <= 1  (restricción fila)
        Flujo óptimo: f(A0,B0) = 0.5, f(A0,B1) = 0.5
        EMD = 0.5*2 + 0.5*4 = 3.0
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[2.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(earth_movers_distance(A, B), 3.0, places=5)


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: mahalanobis_distance
# ══════════════════════════════════════════════════════════════════════════════

class TestMahalanobisDistance(unittest.TestCase):

    def test_identical_bags(self):
        """
        A = B → μ_a = μ_b → diff = 0 → d = 0.
        """
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertAlmostEqual(mahalanobis_distance(A, A), 0.0, places=10)

    def test_single_instance_each_identity_cov(self):
        """
        A = [[0, 0]], B = [[3, 4]]
        n < 2 → covarianza = identidad para ambas.
        cov_combined = 0.5*I + 0.5*I = I
        cov_inv = I
        diff = [-3, -4]
        maha_sq = (-3)^2 + (-4)^2 = 25
        d = sqrt(25) = 5
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(mahalanobis_distance(A, B), 5.0, places=10)

    def test_symmetry(self):
        """d(A, B) == d(B, A)."""
        A = make_bag(np.array([[0.0, 0.0], [2.0, 2.0]]))
        B = make_bag(np.array([[5.0, 5.0], [7.0, 7.0]]))
        self.assertAlmostEqual(
            mahalanobis_distance(A, B),
            mahalanobis_distance(B, A),
            places=10,
        )

    def test_non_negative(self):
        """La distancia siempre es >= 0."""
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        B = make_bag(np.array([[5.0, 6.0], [7.0, 8.0]]))
        self.assertGreaterEqual(mahalanobis_distance(A, B), 0.0)

    def test_single_instance_vs_multi(self):
        """
        A = [[0, 0]]  (1 instancia → cov_a = I)
        B = [[4, 0], [6, 0]]  → μ_b = [5, 0], cov_b = [[2, 0],[0, 0]]
        
        cov_combined = 0.5*I + 0.5*[[2,0],[0,0]] = [[1.5, 0],[0, 0.5]]
        diff = [0,0] - [5,0] = [-5, 0]
        cov_inv = [[1/1.5, 0], [0, 1/0.5]] = [[2/3, 0], [0, 2]]
        maha_sq = (-5)^2 * (2/3) + 0 = 25 * 2/3 = 50/3
        d = sqrt(50/3) ≈ 4.0825
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[4.0, 0.0], [6.0, 0.0]]))
        expected = math.sqrt(50.0 / 3.0)
        self.assertAlmostEqual(mahalanobis_distance(A, B), expected, places=4)

    def test_isotropic_covariance_equals_scaled_euclidean(self):
        """
        Si ambas bolsas tienen la misma covarianza σ²I, la distancia de
        Mahalanobis es proporcional a la euclidiana entre medias:
        d_maha = ||μ_a - μ_b|| / σ

        A = [[0,0],[2,0]] → μ_a = [1,0], cov_a = [[2,0],[0,0]]
        B = [[6,0],[8,0]] → μ_b = [7,0], cov_b = [[2,0],[0,0]]
        cov_combined = [[2,0],[0,0]]  (singular → pinv)
        diff = [-6, 0]
        pinv de [[2,0],[0,0]] = [[0.5,0],[0,0]]
        maha_sq = 36*0.5 = 18
        d = sqrt(18) ≈ 4.2426
        """
        A = make_bag(np.array([[0.0, 0.0], [2.0, 0.0]]))
        B = make_bag(np.array([[6.0, 0.0], [8.0, 0.0]]))
        expected = math.sqrt(18.0)
        self.assertAlmostEqual(mahalanobis_distance(A, B), expected, places=4)

    def test_high_dimensional(self):
        """
        Bolsas unitarias en 5D con una sola instancia cada una.
        cov = I → maha = euclidiana.

        A = [[0,0,0,0,0]], B = [[1,1,1,1,1]]
        d = sqrt(5) ≈ 2.2361
        """
        A = make_bag(np.zeros((1, 5)))
        B = make_bag(np.ones((1, 5)))
        self.assertAlmostEqual(mahalanobis_distance(A, B), math.sqrt(5), places=5)

    def test_returns_float(self):
        """El resultado debe ser un float nativo de Python."""
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        B = make_bag(np.array([[5.0, 6.0], [7.0, 8.0]]))
        result = mahalanobis_distance(A, B)
        self.assertIsInstance(result, float)


# ══════════════════════════════════════════════════════════════════════════════
# Tests de robustez numérica
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalRobustness(unittest.TestCase):

    def test_cauchy_schwarz_near_zero_norm(self):
        """Vectores con norma muy pequeña no deben producir NaN."""
        A = make_bag(np.array([[1e-15, 0.0]]))
        B = make_bag(np.array([[0.0, 1e-15]]))
        result = cauchy_schwarz_distance(A, B)
        self.assertTrue(math.isfinite(result) or result == float('inf'))

    def test_mahalanobis_singular_covariance(self):
        """
        Si la covarianza combinada es singular, debe usar pseudoinversa
        sin error.
        A = [[1,0],[1,0]] → cov = [[0,0],[0,0]] (singular)
        """
        A = make_bag(np.array([[1.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[2.0, 0.0], [2.0, 0.0]]))
        result = mahalanobis_distance(A, B)
        self.assertTrue(math.isfinite(result))
        self.assertGreaterEqual(result, 0.0)

    def test_large_values_no_overflow(self):
        """No debe haber overflow con coordenadas grandes."""
        A = make_bag(np.array([[1e6, 1e6]]))
        B = make_bag(np.array([[2e6, 2e6]]))
        for fn in [cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance]:
            with self.subTest(fn=fn.__name__):
                result = fn(A, B)
                self.assertTrue(math.isfinite(result))

    def test_all_return_float(self):
        """Las tres funciones deben devolver float nativo."""
        A = make_bag(np.array([[1.0, 2.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        for fn in [cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance]:
            with self.subTest(fn=fn.__name__):
                self.assertIsInstance(fn(A, B), float)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
