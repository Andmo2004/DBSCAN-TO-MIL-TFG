"""
tests/test_miknn.py

Tests unitarios para `miclustering.models.miknn.MIKnn`.

MIKnn es el único modelo supervisado de la librería.  A diferencia de los
modelos de clustering, predict() devuelve directamente etiquetas de clase (0/1),
sin necesidad de un mapeo Húngaro posterior.

Cubre:
  1. Construcción y validación de parámetros
  2. fit() — lazy learning (solo almacena datos)
  3. predict() — mayoría de votos con desempate por distancia
  4. predict_bag() — predicción de una sola bolsa
  5. predict_proba() — proporciones de votos
  6. get_neighbors() — k vecinos más cercanos con distancias
  7. get_statistics()
  8. Representaciones __str__ / __repr__

Problemas de diseño detectados
--------------------------------
1.  MIKnn._parse_label usa int(float(raw)) sin el mapa nominal de utils.parse_label.
    Esto significa que etiquetas como "positive" / "negative" en el dataset de test
    se pasarán a predict() correctamente (son las bag labels del DATASET, no del
    resultado de predict), pero la comparación en el evaluador puede fallar si
    las etiquetas del dataset de entrenamiento son nominales.
    → Se documenta con un test xfail explicativo.

2.  predict() recalcula las distancias de cada bolsa test a TODOS los bags de
    train en cada llamada.  Sin caché ni precomputed, es O(n_test * n_train).
    Para tests unitarios esto es aceptable con datasets pequeños.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from miclustering.models.miknn import MIKnn
from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.attribute import Attribute

from tests.models.conftest import _make_binary_dataset, _schema


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ds(n_pos: int = 6, n_neg: int = 6, seed: int = 0) -> MIData:
    return _make_binary_dataset(n_pos=n_pos, n_neg=n_neg, seed=seed)


def _single_bag(bag_id: str, label: str, values: list) -> Bag:
    s = _schema(len(values))
    return Bag(bag_id, label, [Instance(values, s)])


def _perfectly_separable_dataset() -> tuple[MIData, MIData]:
    """
    Dataset totalmente separable para tests deterministas:
      Train: positivos en [10, 10], negativos en [0, 0]
      Test:  un punto cerca de positivos, uno cerca de negativos
    """
    s = _schema(2)
    train_bags = [
        Bag("tp1", "1", [Instance([10.0, 10.0], s)]),
        Bag("tp2", "1", [Instance([10.1, 10.0], s)]),
        Bag("tp3", "1", [Instance([10.0, 10.1], s)]),
        Bag("tn1", "0", [Instance([0.0, 0.0], s)]),
        Bag("tn2", "0", [Instance([0.1, 0.0], s)]),
        Bag("tn3", "0", [Instance([0.0, 0.1], s)]),
    ]
    test_bags = [
        Bag("test_pos", "1", [Instance([10.05, 10.05], s)]),
        Bag("test_neg", "0", [Instance([0.05, 0.05], s)]),
    ]
    return MIData(train_bags, "sep_train"), MIData(test_bags, "sep_test")


# ─── 1. Construcción ──────────────────────────────────────────────────────────

class TestMIKnnConstruction:

    def test_stores_k(self):
        m = MIKnn(k=3)
        assert m.k == 3

    def test_default_k_is_three(self):
        m = MIKnn()
        assert m.k == 3

    def test_default_metric_is_hausdorff(self):
        m = MIKnn()
        assert m.metric_name == "hausdorff"

    def test_custom_metric_stored(self):
        m = MIKnn(k=1, metric="cauchy_schwarz")
        assert m.metric_name == "cauchy_schwarz"

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKnn(k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            MIKnn(k=-1)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Métrica"):
            MIKnn(k=1, metric="unknown_xyz")

    def test_is_not_fitted_initially(self):
        assert not MIKnn().is_fitted

    def test_n_train_bags_zero_before_fit(self):
        assert MIKnn().n_train_bags == 0


# ─── 2. fit() — lazy learning ─────────────────────────────────────────────────

class TestMIKnnFit:

    def test_fit_returns_self(self):
        ds = _ds()
        m = MIKnn(k=1)
        assert m.fit(ds) is m

    def test_is_fitted_after_fit(self):
        ds = _ds()
        m = MIKnn(k=1)
        m.fit(ds)
        assert m.is_fitted

    def test_n_train_bags_matches_dataset(self):
        ds = _ds(n_pos=5, n_neg=5)
        m = MIKnn(k=1)
        m.fit(ds)
        assert m.n_train_bags == 10

    def test_fit_empty_dataset_raises(self):
        m = MIKnn(k=1)
        with pytest.raises(ValueError):
            m.fit(MIData([], "empty"))

    def test_fit_stores_all_bags(self):
        ds = _ds(n_pos=4, n_neg=4)
        m = MIKnn(k=1)
        m.fit(ds)
        assert len(m._train_bags) == 8

    def test_fit_with_k_larger_than_dataset_warns_but_continues(self):
        """k > n_bags no debe lanzar excepción, solo warning."""
        ds = _ds(n_pos=2, n_neg=2)
        m = MIKnn(k=100)
        m.fit(ds)  # no debe fallar
        assert m.is_fitted


# ─── 3. predict() ─────────────────────────────────────────────────────────────

class TestMIKnnPredict:

    def test_predict_before_fit_raises(self):
        m = MIKnn(k=1)
        with pytest.raises(RuntimeError):
            m.predict(_ds())

    def test_predict_empty_test_raises(self):
        m = MIKnn(k=1)
        m.fit(_ds())
        with pytest.raises(ValueError):
            m.predict(MIData([], "empty"))

    def test_predict_returns_dict(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        result = m.predict(test)
        assert isinstance(result, dict)

    def test_predict_covers_all_test_ids(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        result = m.predict(test)
        assert set(result.keys()) == {bag.bag_id for bag in test.bags}

    def test_predict_values_are_integers(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        result = m.predict(test)
        for v in result.values():
            assert isinstance(v, int)

    def test_perfectly_separable_k1_correct_predictions(self):
        """Con k=1 y clusters perfectamente separados debe predecir 100% correcto."""
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        result = m.predict(test)
        assert result["test_pos"] == 1
        assert result["test_neg"] == 0

    def test_predict_output_only_contains_zero_and_one(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        result = m.predict(test)
        assert set(result.values()).issubset({0, 1})


# ─── 4. predict_bag() ─────────────────────────────────────────────────────────

class TestMIKnnPredictBag:

    def test_predict_bag_before_fit_raises(self):
        m = MIKnn(k=1)
        s = _schema(2)
        bag = Bag("b", "1", [Instance([1.0, 1.0], s)])
        with pytest.raises(RuntimeError):
            m.predict_bag(bag)

    def test_predict_bag_returns_int(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        result = m.predict_bag(test.bags[0])
        assert isinstance(result, int)

    def test_predict_bag_consistent_with_predict(self):
        """predict_bag y predict deben dar el mismo resultado para un mismo bag."""
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        bag = test.bags[0]
        assert m.predict_bag(bag) == m.predict(test)[bag.bag_id]


# ─── 5. predict_proba() ───────────────────────────────────────────────────────

class TestMIKnnPredictProba:

    def test_predict_proba_before_fit_raises(self):
        m = MIKnn(k=1)
        with pytest.raises(RuntimeError):
            m.predict_proba(_ds())

    def test_predict_proba_returns_dict_of_dicts(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        probas = m.predict_proba(test)
        assert isinstance(probas, dict)
        for v in probas.values():
            assert isinstance(v, dict)

    def test_probabilities_sum_to_one(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        probas = m.predict_proba(test)
        for bag_proba in probas.values():
            total = sum(bag_proba.values())
            assert total == pytest.approx(1.0)

    def test_probabilities_between_zero_and_one(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        probas = m.predict_proba(test)
        for bag_proba in probas.values():
            for p in bag_proba.values():
                assert 0.0 <= p <= 1.0

    def test_perfectly_separable_high_probability_for_correct_class(self):
        """Con datos separados y k=1, la proba de la clase correcta debe ser 1.0."""
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        probas = m.predict_proba(test)
        assert probas["test_pos"][1] == pytest.approx(1.0)
        assert probas["test_neg"][0] == pytest.approx(1.0)


# ─── 6. get_neighbors() ───────────────────────────────────────────────────────

class TestMIKnnGetNeighbors:

    def test_get_neighbors_before_fit_raises(self):
        m = MIKnn(k=1)
        s = _schema(2)
        bag = Bag("b", "1", [Instance([1.0, 1.0], s)])
        with pytest.raises(RuntimeError):
            m.get_neighbors(bag)

    def test_get_neighbors_returns_list_of_tuples(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=2, metric="hausdorff")
        m.fit(train)
        neighbors = m.get_neighbors(test.bags[0])
        assert isinstance(neighbors, list)
        for item in neighbors:
            assert len(item) == 3  # (bag_id, label, dist)

    def test_get_neighbors_count_is_k(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=2, metric="hausdorff")
        m.fit(train)
        neighbors = m.get_neighbors(test.bags[0])
        assert len(neighbors) == 2

    def test_get_neighbors_sorted_by_distance(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        neighbors = m.get_neighbors(test.bags[0])
        distances = [d for _, _, d in neighbors]
        assert distances == sorted(distances)

    def test_get_neighbors_distances_are_non_negative(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=3, metric="hausdorff")
        m.fit(train)
        neighbors = m.get_neighbors(test.bags[0])
        for _, _, d in neighbors:
            assert d >= 0.0

    def test_nearest_neighbor_is_correct_class(self):
        """El vecino más cercano a test_pos debe ser de clase 1."""
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        m.fit(train)
        nearest = m.get_neighbors(test.bags[0])
        _, label, _ = nearest[0]
        assert label == 1


# ─── 7. fit_predict ───────────────────────────────────────────────────────────

class TestMIKnnFitPredict:

    def test_fit_predict_with_train_only_returns_train_preds(self):
        train = _ds()
        m = MIKnn(k=1)
        result = m.fit_predict(train)
        assert set(result.keys()) == {bag.bag_id for bag in train.bags}

    def test_fit_predict_with_test_dataset_returns_test_preds(self):
        train, test = _perfectly_separable_dataset()
        m = MIKnn(k=1, metric="hausdorff")
        result = m.fit_predict(train, test)
        assert set(result.keys()) == {bag.bag_id for bag in test.bags}


# ─── 8. get_statistics() ─────────────────────────────────────────────────────

class TestMIKnnStatistics:

    def test_not_fitted_returns_fitted_false(self):
        m = MIKnn(k=3)
        stats = m.get_statistics()
        assert stats["fitted"] is False

    def test_fitted_contains_expected_keys(self):
        m = MIKnn(k=1)
        m.fit(_ds())
        stats = m.get_statistics()
        for key in ("k", "metric", "n_train_bags", "label_counts", "fitted"):
            assert key in stats

    def test_fitted_flag_true_after_fit(self):
        m = MIKnn(k=1)
        m.fit(_ds())
        assert m.get_statistics()["fitted"] is True

    def test_k_matches_constructor(self):
        m = MIKnn(k=5)
        m.fit(_ds())
        assert m.get_statistics()["k"] == 5

    def test_label_counts_sum_to_n_train_bags(self):
        ds = _ds(n_pos=4, n_neg=6)
        m = MIKnn(k=1)
        m.fit(ds)
        stats = m.get_statistics()
        assert sum(stats["label_counts"].values()) == 10


# ─── 9. Representaciones ─────────────────────────────────────────────────────

class TestMIKnnRepresentation:

    def test_repr_unfitted(self):
        m = MIKnn(k=3)
        assert "unfitted" in repr(m)

    def test_repr_fitted(self):
        m = MIKnn(k=1)
        m.fit(_ds())
        assert "fitted" in repr(m)

    def test_repr_contains_k(self):
        m = MIKnn(k=7)
        assert "7" in repr(m)

    def test_str_unfitted_contains_unfitted(self):
        m = MIKnn(k=2)
        assert "Unfitted" in str(m) or "unfitted" in str(m).lower()

    def test_str_fitted_contains_model_name(self):
        m = MIKnn(k=1)
        m.fit(_ds())
        assert "MIKnn" in str(m)


# ─── 10. Documentación de bug conocido (xfail) ───────────────────────────────

class TestMIKnnKnownIssues:

    @pytest.mark.xfail(
        reason=(
            "MIKnn._parse_label usa int(float(raw)) sin el mapa nominal. "
            "Etiquetas 'positive'/'negative' en _train_labels lanzarán ValueError. "
            "Refactor sugerido: delegar a miclustering.data.utils.parse_label."
        ),
        strict=False,
    )
    def test_nominal_string_labels_parsed_correctly(self):
        """MIKnn debería aceptar etiquetas 'positive'/'negative' como 1/0."""
        s = _schema(2)
        bags = [
            Bag("pos", "positive", [Instance([5.0, 5.0], s)]),
            Bag("neg", "negative", [Instance([0.0, 0.0], s)]),
        ]
        train = MIData(bags, "nominal_train")
        m = MIKnn(k=1)
        # Esto lanza ValueError en la implementación actual
        m.fit(train)
        assert m.is_fitted