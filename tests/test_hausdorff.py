"""
tests/test_hausdorff.py

Tests unitarios para las tres variantes de distancia de Hausdorff:
  - hausdorff_distance      (máxima / simétrica estándar)  ec. 3.19-3.20
  - hausdorff_distance_min  (mínima)                       ec. 3.18
  - hausdorff_distance_avg  (promedio)                     ec. 3.21

Cada test incluye el cálculo manual esperado en el docstring para que
sea fácil verificar la corrección sin ejecutar el código.
"""

import os
import sys
import math
import unittest
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from distances.hausdorff import (
    hausdorff_distance,
    hausdorff_distance_min,
    hausdorff_distance_avg,
)

from data.bag import Bag
from data.instance import Instance
from data.attribute import Attribute

# ── Helper ────────────────────────────────────────────────────────────────────

def make_schema(n_features: int) -> list:
    return [Attribute(name=f"feat_{i}", attr_type="real") for i in range(n_features)]

def make_bag(matrix, bag_id="bag"):
    schema = make_schema(matrix.shape[1])
    instances = [Instance(values=row.tolist(), schema=schema) for row in matrix]  # ✅
    return Bag(bag_id=bag_id, label=0, instances=instances)


def make_empty_bag(bag_id: str = "empty") -> Bag:
    """Crea una Bag vacía (sin instancias)."""
    return Bag(bag_id=bag_id, label=0, instances=[])


# ══════════════════════════════════════════════════════════════════════════════
# Tests compartidos por las tres variantes
# ══════════════════════════════════════════════════════════════════════════════

class TestHausdorffCommonProperties(unittest.TestCase):
    """Propiedades que deben cumplir las tres variantes."""

    def _all_funcs(self):
        return [hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg]

    def test_identical_bags_return_zero(self):
        """
        d(A, A) = 0 para las tres variantes.
        Bolsa: [[1, 2], [3, 4]]
        La matriz de distancias cruzadas es la propia distancia de cada punto
        consigo mismo → todos los mínimos son 0 → máximo, mínimo y promedio = 0.
        """
        A = make_bag(np.array([[1.0, 2.0], [3.0, 4.0]]))
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(A, A), 0.0, places=10)

    def test_symmetry(self):
        """
        d(A, B) == d(B, A) para las tres variantes.
        A = [[0, 0]], B = [[3, 4]]  → d_euclid = 5
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        for fn in self._all_funcs():
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(A, B), fn(B, A), places=10)

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

    def test_ordering_min_leq_avg_leq_max(self):
        """
        Invariante de orden: d_min <= d_avg <= d_max.

        A = [[0,0], [1,0]]
        B = [[3,0], [4,0]]

        Matriz de distancias cruzadas:
          d(A0,B0)=3  d(A0,B1)=4
          d(A1,B0)=2  d(A1,B1)=3

        min_A→B = [3, 2]  → sum = 5
        min_B→A = [2, 3]  → sum = 5
        d_min = 2
        d_avg = (5+5) / (2+2) = 2.5
        d_max: h(A,B) = max(3,2)=3, h(B,A) = max(2,3)=3  → max(3,3)=3
        2 <= 2.5 <= 3 ✓
        """
        A = make_bag(np.array([[0.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[3.0, 0.0], [4.0, 0.0]]))
        d_min = hausdorff_distance_min(A, B)
        d_avg = hausdorff_distance_avg(A, B)
        d_max = hausdorff_distance(A, B)
        self.assertLessEqual(d_min, d_avg + 1e-10)
        self.assertLessEqual(d_avg, d_max + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: hausdorff_distance  (MÁXIMA)
# ══════════════════════════════════════════════════════════════════════════════

class TestHausdorffMax(unittest.TestCase):

    def test_single_instance_bags(self):
        """
        A = [[0, 0]], B = [[3, 4]]
        d_euclid(A0, B0) = sqrt(9+16) = 5
        h(A,B) = 5, h(B,A) = 5  → max = 5
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(hausdorff_distance(A, B), 5.0, places=10)

    def test_asymmetric_bags_takes_max_direction(self):
        """
        A tiene un outlier lejano; la Hausdorff máxima lo penaliza.

        A = [[0,0], [10,0]]
        B = [[1,0]]

        Matriz (2x1):
          d(A0,B0) = 1
          d(A1,B0) = 9

        min_A→B = [1, 9]  → h(A,B) = max = 9
        min_B→A = [1]     → h(B,A) = max = 1
        D_max = max(9, 1) = 9
        """
        A = make_bag(np.array([[0.0, 0.0], [10.0, 0.0]]))
        B = make_bag(np.array([[1.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance(A, B), 9.0, places=10)

    def test_symmetric_configuration(self):
        """
        A = [[0,0], [1,0]]
        B = [[3,0], [4,0]]

        Explicado en test_ordering: d_max = 3.
        """
        A = make_bag(np.array([[0.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[3.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance(A, B), 3.0, places=10)

    def test_nested_bags_direction_matters(self):
        """
        B está completamente dentro del rango de A.

        A = [[0,0], [10,0]]
        B = [[4,0], [6,0]]

        Matriz (2x2):
          d(A0,B0)=4  d(A0,B1)=6
          d(A1,B0)=6  d(A1,B1)=4

        min_A→B = [4, 4]  → h(A,B) = 4
        min_B→A = [4, 4]  → h(B,A) = 4
        D_max = 4
        """
        A = make_bag(np.array([[0.0, 0.0], [10.0, 0.0]]))
        B = make_bag(np.array([[4.0, 0.0], [6.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance(A, B), 4.0, places=10)

    def test_high_dimensional(self):
        """
        Dos bolsas unitarias en 5D. Distancia euclídea = sqrt(5*(2^2)) = sqrt(20).
        """
        A = make_bag(np.zeros((1, 5)))
        B = make_bag(np.full((1, 5), 2.0))
        self.assertAlmostEqual(hausdorff_distance(A, B), math.sqrt(20), places=10)


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: hausdorff_distance_min  (MÍNIMA)
# ══════════════════════════════════════════════════════════════════════════════

class TestHausdorffMin(unittest.TestCase):

    def test_single_instance_bags(self):
        """
        A = [[0,0]], B = [[3,4]]
        Única celda de la matriz = 5 → d_min = 5
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(hausdorff_distance_min(A, B), 5.0, places=10)

    def test_picks_closest_pair(self):
        """
        A = [[0,0], [10,0]]
        B = [[1,0], [20,0]]

        Matriz (2x2):
          d(A0,B0)=1   d(A0,B1)=20
          d(A1,B0)=9   d(A1,B1)=10

        d_min = min(1, 20, 9, 10) = 1
        """
        A = make_bag(np.array([[0.0, 0.0], [10.0, 0.0]]))
        B = make_bag(np.array([[1.0, 0.0], [20.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance_min(A, B), 1.0, places=10)

    def test_coincident_instance_gives_zero(self):
        """
        Si una instancia de A coincide con una de B → d_min = 0.
        Esto ilustra por qué la distancia mínima NO es una métrica:
        d(A,B)=0 no implica A==B.
        """
        A = make_bag(np.array([[0.0, 0.0], [5.0, 5.0]]))
        B = make_bag(np.array([[0.0, 0.0], [9.0, 9.0]]))
        self.assertAlmostEqual(hausdorff_distance_min(A, B), 0.0, places=10)

    def test_min_leq_max_always(self):
        """d_min <= d_max en configuración general."""
        A = make_bag(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
        B = make_bag(np.array([[1.0, 0.0], [3.0, 0.0]]))
        self.assertLessEqual(
            hausdorff_distance_min(A, B),
            hausdorff_distance(A, B) + 1e-10
        )

    def test_symmetric_configuration(self):
        """
        A = [[0,0], [1,0]], B = [[3,0], [4,0]]
        Matriz:
          d(A0,B0)=3  d(A0,B1)=4
          d(A1,B0)=2  d(A1,B1)=3
        d_min = 2
        """
        A = make_bag(np.array([[0.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[3.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance_min(A, B), 2.0, places=10)


# ══════════════════════════════════════════════════════════════════════════════
# Tests específicos: hausdorff_distance_avg  (PROMEDIO)
# ══════════════════════════════════════════════════════════════════════════════

class TestHausdorffAvg(unittest.TestCase):

    def test_single_instance_bags(self):
        """
        A = [[0,0]], B = [[3,4]]
        sum_A→B = min(5) = 5
        sum_B→A = min(5) = 5
        d_avg = (5+5) / (1+1) = 5
        """
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(hausdorff_distance_avg(A, B), 5.0, places=10)

    def test_manual_2x2(self):
        """
        A = [[0,0], [1,0]], B = [[3,0], [4,0]]

        Matriz (2x2):
          d(A0,B0)=3  d(A0,B1)=4
          d(A1,B0)=2  d(A1,B1)=3

        min_A→B por fila: [3, 2]  → sum_A→B = 5
        min_B→A por col:  [2, 3]  → sum_B→A = 5
        d_avg = (5+5)/(2+2) = 2.5
        """
        A = make_bag(np.array([[0.0, 0.0], [1.0, 0.0]]))
        B = make_bag(np.array([[3.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(hausdorff_distance_avg(A, B), 2.5, places=10)

    def test_manual_asymmetric_sizes(self):
        """
        A = [[0,0], [10,0]]  (n_a=2)
        B = [[1,0]]          (n_b=1)

        Matriz (2x1):
          d(A0,B0)=1
          d(A1,B0)=9

        sum_A→B: min por fila = [1, 9] → sum = 10
        sum_B→A: min por col  = [1]    → sum = 1
        d_avg = (10+1) / (2+1) = 11/3 ≈ 3.6667
        """
        A = make_bag(np.array([[0.0, 0.0], [10.0, 0.0]]))
        B = make_bag(np.array([[1.0, 0.0]]))
        expected = 11.0 / 3.0
        self.assertAlmostEqual(hausdorff_distance_avg(A, B), expected, places=10)

    def test_avg_between_min_and_max(self):
        """d_min <= d_avg <= d_max en configuración general."""
        A = make_bag(np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]]))
        B = make_bag(np.array([[1.0, 1.0], [3.0, 3.0]]))
        d_min = hausdorff_distance_min(A, B)
        d_avg = hausdorff_distance_avg(A, B)
        d_max = hausdorff_distance(A, B)
        self.assertLessEqual(d_min, d_avg + 1e-10)
        self.assertLessEqual(d_avg, d_max + 1e-10)

    def test_normalization_by_total_instances(self):
        """
        Verifica que el denominador es |A|+|B| y no |A| ni |B| por separado.

        A = [[0,0], [0,0], [0,0]]  (n_a=3, todos en el origen)
        B = [[1,0], [1,0]]         (n_b=2, todos en x=1)

        sum_A→B = 3 * 1 = 3   (cada instancia de A tiene min dist 1)
        sum_B→A = 2 * 1 = 2   (cada instancia de B tiene min dist 1)
        d_avg = (3+2) / (3+2) = 1.0
        """
        A = make_bag(np.zeros((3, 2)))
        B = make_bag(np.ones((2, 2)) * [1.0, 0.0])
        self.assertAlmostEqual(hausdorff_distance_avg(A, B), 1.0, places=10)

    def test_large_bag_symmetry(self):
        """Simetría con bolsas más grandes (10 instancias cada una)."""
        rng = np.random.default_rng(42)
        A = make_bag(rng.random((10, 4)))
        B = make_bag(rng.random((10, 4)))
        self.assertAlmostEqual(
            hausdorff_distance_avg(A, B),
            hausdorff_distance_avg(B, A),
            places=10
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tests de robustez numérica
# ══════════════════════════════════════════════════════════════════════════════

class TestHausdorffNumericalRobustness(unittest.TestCase):

    def test_single_dimension(self):
        """Bolsas 1D: la distancia es simplemente |x_a - x_b|."""
        A = make_bag(np.array([[2.0]]))
        B = make_bag(np.array([[7.0]]))
        for fn in [hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg]:
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(A, B), 5.0, places=10)

    def test_very_close_bags(self):
        """Bolsas casi idénticas: distancia muy pequeña pero no negativa."""
        A = make_bag(np.array([[0.0, 0.0]]))
        B = make_bag(np.array([[1e-10, 0.0]]))
        for fn in [hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg]:
            with self.subTest(fn=fn.__name__):
                result = fn(A, B)
                self.assertGreaterEqual(result, 0.0)
                self.assertAlmostEqual(result, 1e-10, places=15)

    def test_large_values(self):
        """No debe haber overflow con coordenadas grandes."""
        A = make_bag(np.array([[1e6, 1e6]]))
        B = make_bag(np.array([[2e6, 2e6]]))
        expected = math.sqrt(2) * 1e6
        for fn in [hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg]:
            with self.subTest(fn=fn.__name__):
                result = fn(A, B)
                self.assertTrue(math.isfinite(result))
                self.assertAlmostEqual(result, expected, delta=1e-3)

    def test_single_instance_each_all_equal(self):
        """Con una instancia por bolsa, las tres variantes deben dar el mismo valor."""
        A = make_bag(np.array([[1.0, 2.0, 3.0]]))
        B = make_bag(np.array([[4.0, 6.0, 3.0]]))
        expected = math.sqrt(9 + 16)  # sqrt(25) = 5
        d_max = hausdorff_distance(A, B)
        d_min = hausdorff_distance_min(A, B)
        d_avg = hausdorff_distance_avg(A, B)
        self.assertAlmostEqual(d_max, expected, places=10)
        self.assertAlmostEqual(d_min, expected, places=10)
        self.assertAlmostEqual(d_avg, expected, places=10)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)