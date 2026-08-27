"""
tests/models/test_cosmic.py

Tests unitarios para `miclustering.models.cosmic.COSMIC`.

Cubre:
  1. Construcción y validación de parámetros (epsilon, min_pts, epsilon_prime)
  2. Protocolo fit / extract_clusters / fit_predict
  3. Propiedades del estado interno (ordering, reachability_plot, core_distance)
  4. Extracción sucesiva de clústeres con varios epsilon_prime
  5. Soporte de precomputed_matrix
  6. Excepción NotImplementedError en predict (carácter transductivo)
  7. get_statistics / get_noise_points / get_cluster_members / get_cluster_sizes
  8. Representaciones __str__ / __repr__
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from miclustering.models.cosmic import COSMIC
from miclustering.data.midata import MIData
from tests.models.conftest import _make_binary_dataset, _schema


#  Helpers locales 

def _identity_matrix(n: int) -> np.ndarray:
    """Todos los puntos están a distancia 0."""
    return np.zeros((n, n))


def _distinct_matrix(n: int, val: float = 100.0) -> np.ndarray:
    """Todos los puntos están a distancia val (fuera de epsilon)."""
    m = np.full((n, n), val)
    np.fill_diagonal(m, 0.0)
    return m


#  1. Inicialización y Validación de Parámetros 

class TestCOSMICInit:
    def test_valid_init_defaults(self):
        c = COSMIC(epsilon=1.0, min_pts=3)
        assert c.epsilon == 1.0
        assert c.min_pts == 3
        assert c.epsilon_prime is None
        assert c.is_fitted is False
        assert c.labels == {}
        assert c.cluster_count == 0
        assert c.ordering == []
        assert c.reachability_plot == []

    def test_valid_init_with_eps_prime(self):
        c = COSMIC(epsilon=2.0, min_pts=2, epsilon_prime=1.0, metric="hausdorff_min")
        assert c.epsilon == 2.0
        assert c.min_pts == 2
        assert c.epsilon_prime == 1.0

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="'epsilon' debe ser > 0"):
            COSMIC(epsilon=0.0, min_pts=2)
        with pytest.raises(ValueError, match="'epsilon' debe ser > 0"):
            COSMIC(epsilon=-1.0, min_pts=2)

    def test_invalid_min_pts(self):
        with pytest.raises(ValueError, match="'min_pts' debe ser >= 1"):
            COSMIC(epsilon=1.0, min_pts=0)

    def test_invalid_epsilon_prime_greater_than_epsilon(self):
        with pytest.raises(ValueError, match="'epsilon_prime'.*no puede ser mayor que 'epsilon'"):
            COSMIC(epsilon=1.0, min_pts=2, epsilon_prime=2.0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Métrica 'invalid_metric' no reconocida"):
            COSMIC(epsilon=1.0, min_pts=2, metric="invalid_metric")


#  2. Fit y Ordenamiento de Densidad (Paso 1) 

class TestCOSMICFit:
    def test_fit_empty_dataset_raises(self):
        empty_data = MIData(bags=[], name="empty")
        c = COSMIC(epsilon=1.0, min_pts=2)
        with pytest.raises(ValueError, match="dataset de entrenamiento está vacío"):
            c.fit(empty_data)

    def test_fit_incompatible_matrix_shape_raises(self):
        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        with pytest.raises(ValueError, match="shape.*no coincide"):
            c.fit(dataset, precomputed_matrix=np.zeros((5, 5)))

    def test_fit_ordering_and_reachability(self):
        dataset = _make_binary_dataset(n_pos=4, n_neg=4)
        c = COSMIC(epsilon=1.0, min_pts=2)
        matrix = _identity_matrix(8)
        c.fit(dataset, precomputed_matrix=matrix)

        assert c.is_fitted is True
        assert len(c.ordering) == 8
        assert len(c.reachability_plot) == 8
        assert c.cluster_count == 1
        # Todos deben pertenecer al cluster 0
        assert all(v == 0 for v in c.labels.values())

    def test_fit_all_noise_when_isolated(self):
        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        matrix = _distinct_matrix(6, val=50.0)
        c.fit(dataset, precomputed_matrix=matrix)

        assert c.is_fitted is True
        assert c.cluster_count == 0
        assert all(v == COSMIC.NOISE_LABEL for v in c.labels.values())


#  3. Extracción Múltiple de Clústeres (Paso 2) 

class TestCOSMICExtractClusters:
    def test_extract_without_fit_raises(self):
        c = COSMIC(epsilon=1.0, min_pts=2)
        with pytest.raises(RuntimeError, match="No hay un ordenamiento calculado"):
            c.extract_clusters(0.5)

    def test_extract_with_eps_prime_greater_than_eps_raises(self):
        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        c.fit(dataset, precomputed_matrix=_identity_matrix(6))
        with pytest.raises(ValueError, match="epsilon_prime.*no puede ser mayor que epsilon"):
            c.extract_clusters(1.5)

    def test_extract_clusters_multiple_granularities(self):
        # 6 puntos: 3 en cluster A (dist=0.2), 3 en cluster B (dist=0.2), separación entre A y B = 0.8
        n = 6
        matrix = np.full((n, n), 0.8)
        # Cluster A: índices 0, 1, 2
        matrix[:3, :3] = 0.2
        # Cluster B: índices 3, 4, 5
        matrix[3:, 3:] = 0.2
        np.fill_diagonal(matrix, 0.0)

        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        c.fit(dataset, precomputed_matrix=matrix)

        # Con eps_prime = 1.0 (ambos grupos unidos si eps_prime >= 0.8)
        labels_coarse = c.extract_clusters(epsilon_prime=1.0)
        assert c.cluster_count == 1

        # Con eps_prime = 0.3 (grupos separados en 2 clusters)
        labels_fine = c.extract_clusters(epsilon_prime=0.3)
        assert c.cluster_count == 2


#  4. Transductive Behavior & fit_predict 

class TestCOSMICPredict:
    def test_predict_raises_not_implemented(self):
        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        c.fit(dataset, precomputed_matrix=_identity_matrix(6))

        with pytest.raises(NotImplementedError, match="COSMIC es transductivo"):
            c.predict(dataset)

    def test_fit_predict_returns_labels(self):
        dataset = _make_binary_dataset(n_pos=3, n_neg=3)
        c = COSMIC(epsilon=1.0, min_pts=2)
        preds = c.fit_predict(dataset)

        assert isinstance(preds, dict)
        assert len(preds) == 6
        assert preds == c.labels


#  5. Métodos de Consulta y Estadísticas 

class TestCOSMICStatistics:
    def test_get_statistics_not_fitted(self):
        c = COSMIC(epsilon=1.0, min_pts=2)
        assert c.get_statistics() == {"status": "not_fitted"}
        assert c.get_cluster_sizes() == {}
        assert c.get_noise_points() == []
        assert c.get_cluster_members(0) == []

    def test_get_statistics_fitted(self):
        dataset = _make_binary_dataset(n_pos=2, n_neg=2)
        c = COSMIC(epsilon=1.0, min_pts=2)
        c.fit(dataset, precomputed_matrix=_identity_matrix(4))

        stats = c.get_statistics()
        assert stats["total_bags"] == 4
        assert stats["num_clusters"] == 1
        assert stats["noise_points_count"] == 0
        assert stats["noise_percentage"] == 0.0
        assert stats["cluster_sizes"] == {0: 4}

    def test_repr_and_str(self):
        c = COSMIC(epsilon=1.0, min_pts=2)
        assert "unfitted" in repr(c)
        assert "Unfitted" in str(c)

        dataset = _make_binary_dataset(n_pos=2, n_neg=2)
        c.fit(dataset, precomputed_matrix=_identity_matrix(4))
        assert "fitted" in repr(c)
        assert "Fitted on 4 bags" in str(c)
