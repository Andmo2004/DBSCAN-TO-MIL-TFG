"""
tests/test_models_common.py

Tests del contrato scikit-learn compartido por todos los modelos MIL.

Todos los modelos de miclustering implementan la interfaz:
  - fit(dataset) → self
  - predict(dataset) → Dict[str, int]
  - fit_predict(X, y=None) → Dict[str, int]
  - get_statistics() → Dict

Este fichero parametriza los mismos tests sobre los cuatro modelos usando
pytest.mark.parametrize, siguiendo el patrón "Strategy test" para evitar
duplicación.

Para modelos que requieren precomputed_matrix (MIDBSCAN, MIKMedoids),
se inyecta una matriz cercana-a-cero para que los tests sean rápidos.

Diseño consciente
-----------------
Se separa de los test_<modelo>.py individuales para mantener:
  - tests de contrato → aquí (qué hace la interfaz)
  - tests de comportamiento → en cada test_<modelo>.py (cómo lo hace)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.mikmeans import MIKMeans
from miclustering.models.mikmedoids import MIKMedoids
from miclustering.models.miknn import MIKnn
from miclustering.data.midata import MIData

from tests.models.conftest import _make_binary_dataset


#  helpers 

def _close_matrix(n: int) -> np.ndarray:
    m = np.full((n, n), 0.1)
    np.fill_diagonal(m, 0.0)
    return m


def _make_train_test(n_train: int = 10, n_test: int = 6) -> tuple[MIData, MIData]:
    train = _make_binary_dataset(n_pos=n_train // 2, n_neg=n_train // 2, seed=1)
    test = _make_binary_dataset(n_pos=n_test // 2, n_neg=n_test // 2, seed=2)
    return train, test


def _fit_model(model, train: MIData) -> None:
    """Entrena el modelo inyectando precomputed_matrix cuando corresponde."""
    n = train.get_num_bags()
    if isinstance(model, (MIDBSCAN, MIKMedoids)):
        model.fit(train, precomputed_matrix=_close_matrix(n))
    else:
        model.fit(train)


#  parametrize: un id de modelo → factory 

MODEL_FACTORIES = [
    pytest.param(
        lambda: MIDBSCAN(epsilon=999.0, min_pts=2, metric="hausdorff"),
        id="MIDBSCAN",
    ),
    pytest.param(
        lambda: MIKMeans(k=2, metric="hausdorff", random_state=0),
        id="MIKMeans",
    ),
    pytest.param(
        lambda: MIKMedoids(k=2, metric="hausdorff", random_state=0),
        id="MIKMedoids",
    ),
    pytest.param(
        lambda: MIKnn(k=1, metric="hausdorff"),
        id="MIKnn",
    ),
]


#  Contrato: estado inicial 

class TestModelInitialState:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_not_fitted_before_fit(self, factory):
        m = factory()
        assert not m.is_fitted

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_labels_empty_before_fit(self, factory):
        m = factory()
        assert m.labels == {}


#  Contrato: fit() 

class TestModelFitContract:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_fit_returns_self(self, factory):
        train, _ = _make_train_test()
        m = factory()
        result = _fit_model(m, train) or m  # _fit_model devuelve None, accedemos a m
        # Comprobamos directamente que fit devuelve self
        m2 = factory()
        n = train.get_num_bags()
        if isinstance(m2, (MIDBSCAN, MIKMedoids)):
            ret = m2.fit(train, precomputed_matrix=_close_matrix(n))
        else:
            ret = m2.fit(train)
        assert ret is m2

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_is_fitted_after_fit(self, factory):
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        assert m.is_fitted

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_labels_populated_after_fit(self, factory):
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        assert len(m.labels) == train.get_num_bags()

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_all_bag_ids_in_labels_after_fit(self, factory):
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        expected_ids = {bag.bag_id for bag in train.bags}
        assert set(m.labels.keys()) == expected_ids

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_fit_empty_dataset_raises_value_error(self, factory):
        m = factory()
        with pytest.raises(ValueError):
            m.fit(MIData([], "empty"))


#  Contrato: predict() 

class TestModelPredictContract:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_predict_before_fit_raises_runtime_error(self, factory):
        _, test = _make_train_test()
        m = factory()
        with pytest.raises(RuntimeError):
            m.predict(test)

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_predict_returns_dict(self, factory):
        train, test = _make_train_test()
        m = factory()
        _fit_model(m, train)
        result = m.predict(test)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_predict_covers_all_test_bag_ids(self, factory):
        train, test = _make_train_test()
        m = factory()
        _fit_model(m, train)
        result = m.predict(test)
        expected = {bag.bag_id for bag in test.bags}
        assert set(result.keys()) == expected

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_predict_values_are_integers(self, factory):
        train, test = _make_train_test()
        m = factory()
        _fit_model(m, train)
        result = m.predict(test)
        for v in result.values():
            assert isinstance(v, int)

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_predict_empty_test_raises_value_error(self, factory):
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        with pytest.raises(ValueError):
            m.predict(MIData([], "empty"))


#  Contrato: fit_predict() 

class TestModelFitPredictContract:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_fit_predict_without_y_returns_dict(self, factory):
        train, _ = _make_train_test()
        m = factory()
        result = m.fit_predict(train)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_fit_predict_without_y_covers_train_ids(self, factory):
        train, _ = _make_train_test()
        m = factory()
        result = m.fit_predict(train)
        expected = {bag.bag_id for bag in train.bags}
        assert set(result.keys()) == expected

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_fit_predict_with_y_covers_test_ids(self, factory):
        train, test = _make_train_test()
        m = factory()
        result = m.fit_predict(train, test)
        expected = {bag.bag_id for bag in test.bags}
        assert set(result.keys()) == expected


#  Contrato: get_statistics() 

class TestModelStatisticsContract:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_get_statistics_returns_dict(self, factory):
        m = factory()
        result = m.get_statistics()
        assert isinstance(result, dict)

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_get_statistics_not_fitted_has_status_key(self, factory):
        m = factory()
        stats = m.get_statistics()
        # Todos los modelos deben indicar de alguna forma que no están fitted
        assert "status" in stats or "fitted" in stats or "not_fitted" in str(stats).lower()

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_get_statistics_fitted_does_not_raise(self, factory):
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        stats = m.get_statistics()
        assert isinstance(stats, dict)


#  Contrato: labels property es copia 

class TestModelLabelsImmutability:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_labels_property_returns_copy(self, factory):
        """Mutar el dict devuelto no debe alterar el estado interno del modelo."""
        train, _ = _make_train_test()
        m = factory()
        _fit_model(m, train)
        copy = m.labels
        copy.clear()
        assert len(m.labels) == train.get_num_bags()


#  Contrato: representaciones 

class TestModelRepresentationContract:

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_repr_returns_non_empty_string(self, factory):
        m = factory()
        assert len(repr(m)) > 0

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_str_returns_non_empty_string(self, factory):
        m = factory()
        assert len(str(m)) > 0

    @pytest.mark.parametrize("factory", MODEL_FACTORIES)
    def test_repr_fitted_differs_from_unfitted(self, factory):
        """repr() debe cambiar después de fit()."""
        train, _ = _make_train_test()
        m_unfitted = factory()
        repr_before = repr(m_unfitted)
        _fit_model(m_unfitted, train)
        repr_after = repr(m_unfitted)
        assert repr_before != repr_after