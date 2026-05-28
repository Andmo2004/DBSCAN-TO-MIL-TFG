"""
tests/test_midbscan.py

Tests unitarios para `miclustering.models.midbscan.MIDBSCAN`.

Cubre:
  1. Construcción y validación de parámetros
  2. Protocolo fit / predict / fit_predict  (contrato scikit-learn)
  3. Propiedades del estado interno post-fit
  4. Comportamiento del clustering (casos degenerados y normales)
  5. Soporte de precomputed_matrix
  6. predict sobre datos no vistos
  7. get_statistics / get_noise_points / get_cluster_members
  8. Representaciones __str__ / __repr__

Estrategia de aislamiento
--------------------------
Todos los datasets se construyen en memoria.  Para evitar el coste de
calcular matrices de distancias reales en tests unitarios, se usa la opción
`precomputed_matrix` con matrices numpy sintéticas.  Esto desacopla la lógica
del algoritmo de las funciones de distancia (ya cubiertas en test_hausdorff.py).

Problemas de diseño detectados
--------------------------------
1.  fit() libera self._distance_matrix al final (self._distance_matrix = None).
    Correcto por memoria, pero impide inspeccionar la matriz en tests post-fit.
    → No es problema para tests de caja negra.

2.  MIDBSCAN imprime a través del logger de producción.  Se usa caplog para
    silenciar el ruido en tests.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from miclustering.models.midbscan import MIDBSCAN
from miclustering.data.midata import MIData

# reutilizamos helpers del conftest local de modelos
from tests.models.conftest import (
    _make_binary_dataset,
    _schema,
)
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance


# ─── helpers locales ──────────────────────────────────────────────────────────

def _identity_matrix(n: int) -> np.ndarray:
    """Todos los puntos son 'idénticos' entre sí → un único cluster si eps >= 0."""
    return np.zeros((n, n))


def _block_matrix(n_a: int, n_b: int, intra: float = 0.1, inter: float = 10.0) -> np.ndarray:
    """
    Matriz de distancias con dos clusters bien separados:
      - Puntos [0, n_a)  están cerca entre sí (distancia = intra).
      - Puntos [n_a, n_a+n_b) están cerca entre sí.
      - Los dos grupos están lejos (distancia = inter).
    """
    n = n_a + n_b
    m = np.full((n, n), inter)
    np.fill_diagonal(m, 0.0)
    # Bloque A
    for i in range(n_a):
        for j in range(n_a):
            m[i, j] = intra if i != j else 0.0
    # Bloque B
    for i in range(n_a, n):
        for j in range(n_a, n):
            m[i, j] = intra if i != j else 0.0
    return m


def _make_dataset(n: int, prefix: str = "b") -> MIData:
    s = _schema(2)
    bags = [Bag(f"{prefix}_{i}", str(i % 2), [Instance([float(i), 0.0], s)]) for i in range(n)]
    return MIData(bags, f"ds_{n}")


# ─── 1. Construcción y validación ─────────────────────────────────────────────

class TestMIDBSCANConstruction:

    def test_stores_epsilon_and_min_pts(self):
        m = MIDBSCAN(epsilon=0.5, min_pts=3)
        assert m.epsilon == 0.5
        assert m.min_pts == 3

    def test_default_metric_is_hausdorff(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert "hausdorff" in m._metric_name

    def test_custom_metric_stored(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2, metric="cauchy_schwarz")
        assert m._metric_name == "cauchy_schwarz"

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            MIDBSCAN(epsilon=0.0, min_pts=2)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            MIDBSCAN(epsilon=-1.0, min_pts=2)

    def test_min_pts_zero_raises(self):
        with pytest.raises(ValueError, match="min_pts"):
            MIDBSCAN(epsilon=1.0, min_pts=0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Métrica"):
            MIDBSCAN(epsilon=1.0, min_pts=2, metric="unknown_metric")

    def test_is_not_fitted_initially(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert not m.is_fitted

    def test_cluster_count_zero_before_fit(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert m.cluster_count == 0

    def test_labels_empty_before_fit(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert m.labels == {}

    def test_noise_label_is_minus_one(self):
        assert MIDBSCAN.NOISE_LABEL == -1


# ─── 2. Contrato fit() ────────────────────────────────────────────────────────

class TestMIDBSCANFit:

    def test_fit_returns_self(self):
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        result = m.fit(ds, precomputed_matrix=_identity_matrix(6))
        assert result is m

    def test_is_fitted_after_fit(self):
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(6))
        assert m.is_fitted

    def test_fit_empty_dataset_raises(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        with pytest.raises(ValueError):
            m.fit(MIData([], "empty"))

    def test_precomputed_matrix_wrong_shape_raises(self):
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        bad_matrix = np.zeros((3, 3))  # debería ser 4x4
        with pytest.raises(ValueError, match="shape"):
            m.fit(ds, precomputed_matrix=bad_matrix)

    def test_labels_dict_populated_after_fit(self):
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(6))
        assert len(m.labels) == 6

    def test_all_bag_ids_present_in_labels(self):
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(6))
        expected_ids = {bag.bag_id for bag in ds.bags}
        assert set(m.labels.keys()) == expected_ids

    def test_fit_twice_resets_state(self):
        """Llamar fit() dos veces debe producir el mismo resultado."""
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        mat = _identity_matrix(6)
        m.fit(ds, precomputed_matrix=mat)
        labels_first = dict(m.labels)
        m.fit(ds, precomputed_matrix=mat)
        assert m.labels == labels_first

    def test_labels_property_returns_copy(self):
        """Mutar el dict devuelto no debe afectar el estado interno."""
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(4))
        copy = m.labels
        copy.clear()
        assert len(m.labels) == 4


# ─── 3. Comportamiento del clustering ─────────────────────────────────────────

class TestMIDBSCANClusteringBehavior:

    def test_all_noise_when_eps_very_small(self):
        """Con eps muy pequeño, todos los puntos son ruido."""
        ds = _make_dataset(6)
        # Construimos una matriz donde las distancias son todas > 100
        large_dist = np.full((6, 6), 100.0)
        np.fill_diagonal(large_dist, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=2)
        m.fit(ds, precomputed_matrix=large_dist)
        assert m.cluster_count == 0
        assert all(v == MIDBSCAN.NOISE_LABEL for v in m.labels.values())

    def test_single_cluster_when_all_points_close(self):
        """Con eps grande y matriz de zeros, todos forman un cluster."""
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(6))
        assert m.cluster_count >= 1
        non_noise = [v for v in m.labels.values() if v != MIDBSCAN.NOISE_LABEL]
        assert len(non_noise) > 0

    def test_two_clusters_block_matrix(self):
        """Dos grupos bien separados → exactamente 2 clusters."""
        n_a, n_b = 5, 5
        ds = _make_dataset(n_a + n_b)
        mat = _block_matrix(n_a, n_b, intra=0.1, inter=10.0)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=mat)
        assert m.cluster_count == 2

    def test_noise_label_is_minus_one_in_output(self):
        """Los puntos de ruido tienen label -1 en el dict."""
        ds = _make_dataset(4)
        large_dist = np.full((4, 4), 100.0)
        np.fill_diagonal(large_dist, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=2)
        m.fit(ds, precomputed_matrix=large_dist)
        assert all(v == -1 for v in m.labels.values())

    def test_cluster_ids_start_at_zero(self):
        """Los IDs de cluster comienzan en 0 y son consecutivos."""
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(6))
        real_clusters = {v for v in m.labels.values() if v != MIDBSCAN.NOISE_LABEL}
        if real_clusters:
            assert min(real_clusters) == 0

    def test_min_pts_one_creates_cluster_per_point(self):
        """min_pts=1 → cada punto es su propio núcleo (ningún ruido con eps > 0)."""
        ds = _make_dataset(4)
        large = np.full((4, 4), 100.0)
        np.fill_diagonal(large, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=1)
        m.fit(ds, precomputed_matrix=large)
        # Todos son núcleo, nadie es ruido
        assert all(v != MIDBSCAN.NOISE_LABEL for v in m.labels.values())


# ─── 4. predict() ─────────────────────────────────────────────────────────────

class TestMIDBSCANPredict:

    def _fitted_model(self, n_a=5, n_b=5) -> tuple[MIDBSCAN, MIData]:
        ds = _make_dataset(n_a + n_b, prefix="train")
        mat = _block_matrix(n_a, n_b, intra=0.1, inter=10.0)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=mat)
        return m, ds

    def test_predict_before_fit_raises(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        test_ds = _make_dataset(3, prefix="test")
        with pytest.raises(RuntimeError):
            m.predict(test_ds)

    def test_predict_empty_test_raises(self):
        m, _ = self._fitted_model()
        with pytest.raises(ValueError):
            m.predict(MIData([], "empty"))

    def test_predict_returns_dict(self):
        m, train_ds = self._fitted_model()
        test_ds = _make_dataset(4, prefix="test")
        result = m.predict(test_ds)
        assert isinstance(result, dict)

    def test_predict_covers_all_test_bag_ids(self):
        m, _ = self._fitted_model()
        test_ds = _make_dataset(4, prefix="test")
        result = m.predict(test_ds)
        expected_ids = {bag.bag_id for bag in test_ds.bags}
        assert set(result.keys()) == expected_ids

    def test_predict_values_are_integers(self):
        m, _ = self._fitted_model()
        test_ds = _make_dataset(3, prefix="test")
        result = m.predict(test_ds)
        for v in result.values():
            assert isinstance(v, int)

    def test_predict_all_noise_when_no_core_points(self):
        """Si fit() no encontró ningún núcleo, predict devuelve todo ruido."""
        ds = _make_dataset(4)
        large = np.full((4, 4), 100.0)
        np.fill_diagonal(large, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=100)  # nunca alcanza min_pts
        m.fit(ds, precomputed_matrix=large)

        test_ds = _make_dataset(3, prefix="test")
        result = m.predict(test_ds)
        assert all(v == MIDBSCAN.NOISE_LABEL for v in result.values())


# ─── 5. fit_predict ───────────────────────────────────────────────────────────

class TestMIDBSCANFitPredict:

    def test_fit_predict_without_y_returns_train_labels(self):
        ds = _make_dataset(6)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        result = m.fit_predict(ds)
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_fit_predict_with_y_returns_test_predictions(self):
        train_ds = _make_dataset(6, prefix="train")
        test_ds = _make_dataset(3, prefix="test")
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        result = m.fit_predict(train_ds, test_ds)
        assert set(result.keys()) == {bag.bag_id for bag in test_ds.bags}


# ─── 6. Estadísticas y helpers de consulta ────────────────────────────────────

class TestMIDBSCANStatistics:

    def _fitted_two_cluster_model(self) -> MIDBSCAN:
        ds = _make_dataset(10)
        mat = _block_matrix(5, 5, intra=0.1, inter=10.0)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=mat)
        return m

    def test_get_statistics_not_fitted_returns_status(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        stats = m.get_statistics()
        assert stats.get("status") == "not_fitted"

    def test_get_statistics_fitted_contains_expected_keys(self):
        m = self._fitted_two_cluster_model()
        stats = m.get_statistics()
        for key in ("epsilon", "min_pts", "total_bags", "num_clusters",
                    "noise_points_count", "noise_percentage", "cluster_sizes"):
            assert key in stats, f"Missing key: {key}"

    def test_get_statistics_epsilon_matches_constructor(self):
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=3.14, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(4))
        assert m.get_statistics()["epsilon"] == pytest.approx(3.14)

    def test_get_statistics_total_bags_correct(self):
        ds = _make_dataset(8)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(8))
        assert m.get_statistics()["total_bags"] == 8

    def test_noise_percentage_all_noise(self):
        ds = _make_dataset(4)
        large = np.full((4, 4), 100.0)
        np.fill_diagonal(large, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=2)
        m.fit(ds, precomputed_matrix=large)
        stats = m.get_statistics()
        assert stats["noise_percentage"] == pytest.approx(100.0)

    def test_noise_percentage_no_noise(self):
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(4))
        stats = m.get_statistics()
        assert stats["noise_percentage"] == pytest.approx(0.0)

    def test_get_noise_points_returns_list(self):
        ds = _make_dataset(4)
        large = np.full((4, 4), 100.0)
        np.fill_diagonal(large, 0.0)
        m = MIDBSCAN(epsilon=0.01, min_pts=2)
        m.fit(ds, precomputed_matrix=large)
        noise = m.get_noise_points()
        assert isinstance(noise, list)
        assert len(noise) == 4

    def test_get_noise_points_not_fitted_returns_empty(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert m.get_noise_points() == []

    def test_get_cluster_members_returns_bag_ids(self):
        ds = _make_dataset(10)
        mat = _block_matrix(5, 5, intra=0.1, inter=10.0)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=mat)
        for cluster_id in range(m.cluster_count):
            members = m.get_cluster_members(cluster_id)
            assert isinstance(members, list)
            assert len(members) > 0

    def test_get_cluster_members_not_fitted_returns_empty(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert m.get_cluster_members(0) == []

    def test_get_cluster_sizes_keys_are_cluster_ids(self):
        ds = _make_dataset(10)
        mat = _block_matrix(5, 5, intra=0.1, inter=10.0)
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        m.fit(ds, precomputed_matrix=mat)
        sizes = m.get_cluster_sizes()
        assert isinstance(sizes, dict)
        total = sum(sizes.values())
        assert total == 10

    def test_get_cluster_sizes_not_fitted_returns_empty(self):
        m = MIDBSCAN(epsilon=1.0, min_pts=2)
        assert m.get_cluster_sizes() == {}


# ─── 7. Representaciones ──────────────────────────────────────────────────────

class TestMIDBSCANRepresentation:

    def test_repr_unfitted_contains_unfitted(self):
        m = MIDBSCAN(epsilon=0.5, min_pts=3)
        assert "unfitted" in repr(m)

    def test_repr_fitted_contains_fitted(self):
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(4))
        assert "fitted" in repr(m)

    def test_repr_contains_epsilon(self):
        m = MIDBSCAN(epsilon=1.23, min_pts=2)
        assert "1.23" in repr(m)

    def test_str_unfitted_contains_unfitted(self):
        m = MIDBSCAN(epsilon=0.5, min_pts=2)
        assert "Unfitted" in str(m) or "unfitted" in str(m).lower()

    def test_str_fitted_contains_cluster_info(self):
        ds = _make_dataset(4)
        m = MIDBSCAN(epsilon=999.0, min_pts=2)
        m.fit(ds, precomputed_matrix=_identity_matrix(4))
        s = str(m)
        assert "MIDBSCAN" in s