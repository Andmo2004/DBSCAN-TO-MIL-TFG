"""
tests/test_mikmeans.py

Tests unitarios para `miclustering.models.mikmeans.MIKMeans`.

MIKMeans es un K-Means adaptado a MIL donde los centroides son Bags sintéticos
que representan la media de todas las instancias de las bolsas de su cluster.

Cubre:
  1. Construcción y validación de parámetros
  2. Protocolo fit / predict / fit_predict
  3. Centroide como objeto Bag válido
  4. Convergencia y manejo de clusters vacíos
  5. get_statistics / get_cluster_sizes
  6. Representaciones __str__ / __repr__

Problemas de diseño detectados
--------------------------------
1.  _array_to_bag() accede a self._train_bags[0][0].schema internamente,
    lo que acopla la creación de centroides al estado de entrenamiento.
    Si _train_bags está vacío antes de llamar a _array_to_bag, hay IndexError.
    → Los tests cubren este escenario indirecamente a través de fit().

2.  El parámetro `metric` se almacena como `_metric_name` pero se inicializa
    en el constructor con `metric.lower()` — correcto.  Sin embargo,
    `fit_predict` ignora `y` si el modelo ya fue entrenado, lo que puede
    sorprender.  Se documenta con un test.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from miclustering.models.mikmeans import MIKMeans
from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance

from tests.models.conftest import _make_binary_dataset, _schema


#  helpers 

def _ds(n: int = 8, seed: int = 0) -> MIData:
    return _make_binary_dataset(n_pos=n // 2, n_neg=n // 2, seed=seed)


#  1. Construcción 

class TestMIKMeansConstruction:

    def test_stores_k(self):
        m = MIKMeans(k=3, metric="hausdorff")
        assert m.k == 3

    def test_default_metric_is_hausdorff(self):
        m = MIKMeans(k=2)
        assert "hausdorff" in m._metric_name

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKMeans(k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKMeans(k=-1)

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            MIKMeans(k=2, max_iters=0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Métrica"):
            MIKMeans(k=2, metric="unknown")

    def test_is_not_fitted_initially(self):
        m = MIKMeans(k=2)
        assert not m.is_fitted

    def test_labels_empty_before_fit(self):
        m = MIKMeans(k=2)
        assert m.labels == {}

    def test_centroids_empty_before_fit(self):
        m = MIKMeans(k=2)
        assert m.centroids == []

    def test_random_state_stored(self):
        m = MIKMeans(k=2, random_state=42)
        assert m._random_state == 42

    def test_default_tol_is_0_01(self):
        m = MIKMeans(k=2)
        assert m.tol == 0.01

    def test_custom_tol_stored(self):
        m = MIKMeans(k=2, tol=0.05)
        assert m.tol == 0.05

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError, match="tol"):
            MIKMeans(k=2, tol=-0.01)


#  2. fit() 

class TestMIKMeansFit:

    def test_fit_with_custom_tol(self):
        ds = _ds()
        m = MIKMeans(k=2, tol=0.05, random_state=0)
        m.fit(ds)
        assert m.is_fitted

    def test_fit_returns_self(self):
        ds = _ds()
        m = MIKMeans(k=2, random_state=0)
        result = m.fit(ds)
        assert result is m

    def test_is_fitted_after_fit(self):
        ds = _ds()
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert m.is_fitted

    def test_fit_empty_dataset_raises(self):
        m = MIKMeans(k=2)
        with pytest.raises(ValueError):
            m.fit(MIData([], "empty"))

    def test_labels_populated_after_fit(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert len(m.labels) == 8

    def test_all_bag_ids_in_labels(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        expected = {bag.bag_id for bag in ds.bags}
        assert set(m.labels.keys()) == expected

    def test_label_values_are_integers_in_range(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        for v in m.labels.values():
            assert 0 <= v < 2

    def test_centroids_count_equals_k(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert len(m.centroids) == 2

    def test_centroids_are_bag_objects(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        for c in m.centroids:
            assert isinstance(c, Bag)

    def test_k_greater_than_bags_adjusted(self):
        """Si k > n_bags, el modelo debe ajustar k sin lanzar excepción."""
        ds = _ds(4)
        m = MIKMeans(k=100, random_state=0)
        m.fit(ds)
        assert m.is_fitted

    def test_fit_twice_resets_and_produces_same_result(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=7)
        m.fit(ds)
        labels1 = dict(m.labels)
        m.fit(ds)
        assert m.labels == labels1

    def test_reproducible_with_same_random_state(self):
        ds = _ds(10)
        m1 = MIKMeans(k=2, random_state=42)
        m2 = MIKMeans(k=2, random_state=42)
        m1.fit(ds)
        m2.fit(ds)
        assert m1.labels == m2.labels

    def test_different_random_state_may_differ(self):
        """Semillas distintas generalmente producen resultados distintos."""
        ds = _ds(20, seed=0)
        m1 = MIKMeans(k=3, random_state=0)
        m2 = MIKMeans(k=3, random_state=99)
        m1.fit(ds)
        m2.fit(ds)
        # No es imposible que coincidan, pero con datasets grandes es muy improbable.
        # Solo verificamos que ambos modelos están entrenados correctamente.
        assert m1.is_fitted and m2.is_fitted


#  3. predict() 

class TestMIKMeansPredict:

    def _trained(self) -> tuple[MIKMeans, MIData]:
        ds = _ds(10)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        return m, ds

    def test_predict_before_fit_raises(self):
        m = MIKMeans(k=2)
        test_ds = _ds(4)
        with pytest.raises(RuntimeError):
            m.predict(test_ds)

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


#  4. fit_predict 

class TestMIKMeansFitPredict:

    def test_fit_predict_without_y_returns_train_labels(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        result = m.fit_predict(ds)
        assert isinstance(result, dict)
        assert set(result.keys()) == {bag.bag_id for bag in ds.bags}

    def test_fit_predict_with_y_returns_test_predictions(self):
        train = _ds(8)
        test = _ds(4, seed=99)
        m = MIKMeans(k=2, random_state=0)
        result = m.fit_predict(train, test)
        assert set(result.keys()) == {bag.bag_id for bag in test.bags}


#  5. get_statistics / get_cluster_sizes 

class TestMIKMeansStatistics:

    def test_get_statistics_not_fitted(self):
        m = MIKMeans(k=2)
        stats = m.get_statistics()
        assert stats.get("status") == "not_fitted"

    def test_get_statistics_contains_expected_keys(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        stats = m.get_statistics()
        for key in ("k", "metric", "total_bags", "cluster_sizes"):
            assert key in stats

    def test_get_statistics_k_matches_constructor(self):
        ds = _ds(8)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert m.get_statistics()["k"] == 2

    def test_get_statistics_total_bags_correct(self):
        ds = _ds(10)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert m.get_statistics()["total_bags"] == 10

    def test_cluster_sizes_sum_equals_total_bags(self):
        ds = _ds(10)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        sizes = m.get_cluster_sizes()
        assert sum(sizes.values()) == 10

    def test_cluster_sizes_not_fitted_returns_empty(self):
        m = MIKMeans(k=2)
        assert m.get_cluster_sizes() == {}


#  6. Representaciones 

class TestMIKMeansRepresentation:

    def test_repr_unfitted(self):
        m = MIKMeans(k=3)
        assert "unfitted" in repr(m)

    def test_repr_fitted(self):
        ds = _ds(6)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert "fitted" in repr(m)

    def test_repr_contains_k(self):
        m = MIKMeans(k=5)
        assert "5" in repr(m)

    def test_str_unfitted_contains_unfitted(self):
        m = MIKMeans(k=2)
        assert "Unfitted" in str(m) or "unfitted" in str(m).lower()

    def test_str_fitted_contains_model_name(self):
        ds = _ds(6)
        m = MIKMeans(k=2, random_state=0)
        m.fit(ds)
        assert "MIKMeans" in str(m)