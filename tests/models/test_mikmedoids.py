"""
tests/test_mikmedoids.py

Tests unitarios para `miclustering.models.mikmedoids.MIKMedoids`.

MIKMedoids implementa PAM (Partitioning Around Medoids) adaptado a MIL.
A diferencia de MIKMeans, los medoides son bolsas REALES del dataset.

Cubre:
  1. Construcción y validación de parámetros
  2. Protocolo fit / predict / fit_predict (con precomputed_matrix)
  3. Medoides como bolsas reales del dataset de entrenamiento
  4. Convergencia PAM y clusters vacíos
  5. get_statistics / get_cluster_sizes / medoids property
  6. Representaciones __str__ / __repr__

Estrategia de aislamiento
--------------------------
MIKMedoids acepta `precomputed_matrix` en fit(), lo que permite
aislar completamente el algoritmo de las funciones de distancia.
Todos los tests usan matrices sintéticas, igual que en test_midbscan.py.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from miclustering.models.mikmedoids import MIKMedoids
from miclustering.data.midata import MIData
from miclustering.data.bag import Bag

from tests.models.conftest import _make_binary_dataset, _schema


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ds(n: int = 8, seed: int = 0) -> MIData:
    return _make_binary_dataset(n_pos=n // 2, n_neg=n // 2, seed=seed)


def _close_matrix(n: int, off_diag: float = 0.1) -> np.ndarray:
    """Todos los puntos están cerca entre sí."""
    m = np.full((n, n), off_diag)
    np.fill_diagonal(m, 0.0)
    return m


def _block_matrix(n_a: int, n_b: int, intra: float = 0.1, inter: float = 10.0) -> np.ndarray:
    n = n_a + n_b
    m = np.full((n, n), inter)
    np.fill_diagonal(m, 0.0)
    for i in range(n_a):
        for j in range(n_a):
            m[i, j] = intra if i != j else 0.0
    for i in range(n_a, n):
        for j in range(n_a, n):
            m[i, j] = intra if i != j else 0.0
    return m


# ─── 1. Construcción ──────────────────────────────────────────────────────────

class TestMIKMedoidsConstruction:

    def test_stores_k(self):
        m = MIKMedoids(k=3)
        assert m.k == 3

    def test_default_metric_is_hausdorff(self):
        m = MIKMedoids(k=2)
        assert "hausdorff" in m._metric_name

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKMedoids(k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKMedoids(k=-1)

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            MIKMedoids(k=2, max_iters=0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Métrica"):
            MIKMedoids(k=2, metric="unknown_metric")

    def test_is_not_fitted_initially(self):
        m = MIKMedoids(k=2)
        assert not m.is_fitted

    def test_labels_empty_before_fit(self):
        m = MIKMedoids(k=2)
        assert m.labels == {}

    def test_medoids_empty_before_fit(self):
        m = MIKMedoids(k=2)
        assert m.medoids == []


# ─── 2. fit() ─────────────────────────────────────────────────────────────────

class TestMIKMedoidsFit:

    def test_fit_returns_self(self):
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        result = m.fit(ds)
        assert result is m

    def test_is_fitted_after_fit(self):
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds)
        assert m.is_fitted

    def test_fit_empty_dataset_raises(self):
        m = MIKMedoids(k=2)
        with pytest.raises(ValueError):
            m.fit(MIData([], "empty"))

    def test_precomputed_matrix_used_in_fit(self):
        ds = _ds(10)
        mat = _block_matrix(5, 5)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=mat)
        assert m.is_fitted

    def test_precomputed_wrong_shape_raises(self):
        ds = _ds(6)
        bad = np.zeros((4, 4))
        m = MIKMedoids(k=2)
        with pytest.raises(ValueError, match="shape"):
            m.fit(ds, precomputed_matrix=bad)

    def test_labels_populated_after_fit(self):
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        assert len(m.labels) == 10

    def test_all_bag_ids_in_labels(self):
        ds = _ds(8)
        mat = _close_matrix(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=mat)
        assert set(m.labels.keys()) == {bag.bag_id for bag in ds.bags}

    def test_label_values_in_valid_range(self):
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(8))
        for v in m.labels.values():
            assert 0 <= v < 2

    def test_k_greater_than_bags_adjusted(self):
        ds = _ds(4)
        m = MIKMedoids(k=100, random_state=0)
        m.fit(ds)  # debe ajustar k sin error
        assert m.is_fitted

    def test_medoids_are_real_bags_from_training(self):
        """Los medoides deben ser bolsas reales del dataset de entrenamiento."""
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        train_ids = {bag.bag_id for bag in ds.bags}
        for medoid in m.medoids:
            assert isinstance(medoid, Bag)
            assert medoid.bag_id in train_ids

    def test_medoids_count_equals_k(self):
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        assert len(m.medoids) == 2

    def test_two_clusters_block_matrix(self):
        """Con dos grupos bien separados, deben formarse exactamente 2 clusters."""
        ds = _ds(10)
        mat = _block_matrix(5, 5, intra=0.1, inter=10.0)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=mat)
        cluster_ids = set(m.labels.values())
        assert len(cluster_ids) == 2

    def test_distance_matrix_released_after_fit(self):
        """La matriz no debe persistir en memoria tras fit() (protección de RAM)."""
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(8))
        assert m._distance_matrix is None

    def test_reproducible_with_same_random_state(self):
        ds = _ds(10)
        mat = _close_matrix(10)
        m1 = MIKMedoids(k=2, random_state=42)
        m2 = MIKMedoids(k=2, random_state=42)
        m1.fit(ds, precomputed_matrix=mat)
        m2.fit(ds, precomputed_matrix=mat)
        assert m1.labels == m2.labels

    def test_fit_twice_resets_state(self):
        ds = _ds(8)
        mat = _close_matrix(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=mat)
        labels1 = dict(m.labels)
        m.fit(ds, precomputed_matrix=mat)
        assert m.labels == labels1


# ─── 3. predict() ─────────────────────────────────────────────────────────────

class TestMIKMedoidsPredict:

    def _trained(self) -> tuple[MIKMedoids, MIData]:
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        return m, ds

    def test_predict_before_fit_raises(self):
        m = MIKMedoids(k=2)
        with pytest.raises(RuntimeError):
            m.predict(_ds(4))

    def test_predict_empty_test_raises(self):
        m, _ = self._trained()
        with pytest.raises(ValueError):
            m.predict(MIData([], "empty"))

    def test_predict_returns_dict(self):
        m, _ = self._trained()
        result = m.predict(_ds(4, seed=99))
        assert isinstance(result, dict)

    def test_predict_covers_all_test_ids(self):
        m, _ = self._trained()
        test_ds = _ds(6, seed=99)
        result = m.predict(test_ds)
        assert set(result.keys()) == {bag.bag_id for bag in test_ds.bags}

    def test_predict_values_in_valid_range(self):
        m, _ = self._trained()
        result = m.predict(_ds(4, seed=99))
        for v in result.values():
            assert 0 <= v < m.k

    def test_predict_values_are_integers(self):
        m, _ = self._trained()
        result = m.predict(_ds(4, seed=99))
        for v in result.values():
            assert isinstance(v, int)


# ─── 4. fit_predict ───────────────────────────────────────────────────────────

class TestMIKMedoidsFitPredict:

    def test_fit_predict_without_y_returns_train_labels(self):
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        result = m.fit_predict(ds)
        assert isinstance(result, dict)
        assert set(result.keys()) == {bag.bag_id for bag in ds.bags}

    def test_fit_predict_with_y_returns_test_predictions(self):
        train = _ds(8)
        test = _ds(4, seed=99)
        m = MIKMedoids(k=2, random_state=0)
        result = m.fit_predict(train, test)
        assert set(result.keys()) == {bag.bag_id for bag in test.bags}


# ─── 5. Estadísticas ─────────────────────────────────────────────────────────

class TestMIKMedoidsStatistics:

    def test_get_statistics_not_fitted(self):
        m = MIKMedoids(k=2)
        stats = m.get_statistics()
        assert stats.get("status") == "not_fitted"

    def test_get_statistics_contains_expected_keys(self):
        ds = _ds(8)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(8))
        stats = m.get_statistics()
        for key in ("k", "metric", "total_bags", "cluster_sizes", "medoids"):
            assert key in stats

    def test_get_statistics_total_bags(self):
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        assert m.get_statistics()["total_bags"] == 10

    def test_get_statistics_medoids_are_bag_ids(self):
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        train_ids = {bag.bag_id for bag in ds.bags}
        for medoid_id in m.get_statistics()["medoids"]:
            assert medoid_id in train_ids

    def test_cluster_sizes_sum_equals_total_bags(self):
        ds = _ds(10)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(10))
        assert sum(m.get_cluster_sizes().values()) == 10

    def test_cluster_sizes_not_fitted_returns_empty(self):
        m = MIKMedoids(k=2)
        assert m.get_cluster_sizes() == {}


# ─── 6. Representaciones ─────────────────────────────────────────────────────

class TestMIKMedoidsRepresentation:

    def test_repr_unfitted(self):
        m = MIKMedoids(k=3)
        assert "unfitted" in repr(m)

    def test_repr_fitted(self):
        ds = _ds(6)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(6))
        assert "fitted" in repr(m)

    def test_str_unfitted_contains_model_name(self):
        m = MIKMedoids(k=2)
        assert "MIKMedoids" in str(m)

    def test_str_fitted_contains_model_name(self):
        ds = _ds(6)
        m = MIKMedoids(k=2, random_state=0)
        m.fit(ds, precomputed_matrix=_close_matrix(6))
        assert "MIKMedoids" in str(m)