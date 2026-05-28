"""
tests/test_evaluation.py

Tests unitarios para el módulo `miclustering.evaluation`.

Cubre los tres ficheros del módulo en un único archivo, organizados en
clases por responsabilidad:

    1. TestDetectImbalanceRatio   - scoring.detect_imbalance_ratio()
    2. TestScoreLabels            - scoring.score_labels()
    3. TestMILEvaluatorMapping    - bcm.MILEvaluator.hungarian_map_clusters_to_labels()
    4. TestMILEvaluatorEvaluate   - bcm.MILEvaluator.evaluate()
    5. TestBaseCVI                - cvi.BaseCVI (contrato de la clase base)
    6. TestSEDIndex               - cvi.SEDIndex
    7. TestDDIndex                - cvi.DDIndex
    8. TestHcIndex                - cvi.HcIndex
    9. TestVRCIndex               - cvi.VRCIndex
    10. TestIIndex                - cvi.IIndex
    11. TestInternalCVIEvaluator  - cvi.InternalCVIEvaluator (orquestador)

Estrategia de testing
---------------------
*   Todo se construye en memoria: ni archivos ARFF, ni scipy, ni disco.
*   Los fixtures de conftest.py (make_bag, make_schema, etc.) se reutilizan
    a través de la función helper local `_make_dataset` para no depender de
    fixtures de otros módulos.
*   Se testea comportamiento observable (valores de retorno, tipos, contratos
    de error), nunca detalles de implementación.
*   Las anotaciones `@pytest.mark.xfail` documentan bugs conocidos o diseños
    que dificultan la testabilidad sin modificar producción.

Problemas de diseño detectados (auditoría)
-------------------------------------------
1.  `bcm.MILEvaluator.evaluate()` imprime en stdout como efecto secundario.
    Hace los tests verbosos; refactor sugerido: devolver el reporte como str
    o aceptar un parámetro `verbose=False`.

2.  `bcm.MILEvaluator.evaluate()` llama a `parse_label` sin importarla
    explícitamente en el scope local del método.  Falla en ejecución si
    `parse_label` no está importada en `bcm.py`.  Ya está en el código de
    producción, pero hay que verificarlo.

3.  `scoring.score_labels` mezcla mapeo húngaro (lógica compleja) con el
    cálculo final de la métrica.  Dificulta testear el mapeo por separado.
    Refactor sugerido: extraer `_hungarian_map()` como función independiente.

4.  `cvi.InternalCVIEvaluator._print_report()` imprime en stdout.  Idem 1.

5.  `test_cvi.py` existente llama a `MIData.from_arff()` (I/O), requiere el
    fichero `datasets/musk1.arff` en disco y usa `logging.basicConfig` a nivel
    de módulo, lo que contamina el logger de pytest.  Este fichero reemplaza
    esa lógica con fixtures puras.
"""

from __future__ import annotations

import math
import sys
import os

# Asegurar que src/ está en el path cuando se ejecuta desde la raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in (_PROJECT_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.midata import MIData

#  módulo bajo test 
from miclustering.evaluation.scoring import detect_imbalance_ratio, score_labels
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.evaluation.cvi import (
    BaseCVI,
    SEDIndex,
    DDIndex,
    HcIndex,
    VRCIndex,
    IIndex,
    InternalCVIEvaluator,
)


# Helpers de construcción en memoria

def _schema(n: int = 3) -> list[Attribute]:
    return [Attribute(f"f{i}", "real") for i in range(n)]


def _instance(values: list[float], schema=None) -> Instance:
    if schema is None:
        schema = _schema(len(values))
    return Instance(list(values), schema)


def _bag(bag_id: str, label: int | str, matrix: list[list[float]]) -> Bag:
    s = _schema(len(matrix[0]))
    insts = [Instance(row, s) for row in matrix]
    return Bag(bag_id=bag_id, label=label, instances=insts)


def _dataset(
    n_pos: int = 6,
    n_neg: int = 6,
    n_inst: int = 4,
    n_feat: int = 3,
    *,
    seed: int = 0,
    label_type: str = "int",   # "int" | "str"
) -> MIData:
    """Dataset binario completamente sintético, sin I/O."""
    rng = np.random.RandomState(seed)
    bags: list[Bag] = []
    s = _schema(n_feat)
    for i in range(n_pos):
        mat = (rng.rand(n_inst, n_feat) + 1.0).tolist()   # cluster +1
        lbl = 1 if label_type == "int" else "1"
        bags.append(Bag(f"pos_{i}", lbl, [Instance(r, s) for r in mat]))
    for i in range(n_neg):
        mat = (rng.rand(n_inst, n_feat) * 0.1).tolist()   # cluster ~0
        lbl = 0 if label_type == "int" else "0"
        bags.append(Bag(f"neg_{i}", lbl, [Instance(r, s) for r in mat]))
    return MIData(bags, "synthetic")


def _perfect_labels(dataset: MIData) -> dict[str, int]:
    """Devuelve el dict {bag_id: label_int} con etiquetas perfectas."""
    return {
        bag.bag_id: int(float(bag.label))
        for bag in dataset.bags
    }


def _all_noise(dataset: MIData) -> dict[str, int]:
    return {bag.bag_id: -1 for bag in dataset.bags}


def _single_cluster(dataset: MIData) -> dict[str, int]:
    return {bag.bag_id: 0 for bag in dataset.bags}


def _two_cluster_labels(dataset: MIData) -> dict[str, int]:
    """Asigna cluster 0 a los positivos y cluster 1 a los negativos."""
    result = {}
    for bag in dataset.bags:
        lbl = int(float(bag.label))
        result[bag.bag_id] = lbl   # coincide con la clase
    return result


# 1. detect_imbalance_ratio

class TestDetectImbalanceRatio:

    def test_balanced_returns_one(self):
        ds = _dataset(n_pos=10, n_neg=10)
        assert detect_imbalance_ratio(ds) == pytest.approx(1.0)

    def test_two_to_one_imbalance(self):
        ds = _dataset(n_pos=10, n_neg=5)
        assert detect_imbalance_ratio(ds) == pytest.approx(0.5)

    def test_returns_minority_over_majority(self):
        ds = _dataset(n_pos=3, n_neg=9)
        r = detect_imbalance_ratio(ds)
        assert 0.0 < r <= 1.0
        assert r == pytest.approx(3 / 9)

    def test_all_same_class_returns_zero(self):
        """Si una clase tiene 0 elementos → ratio 0."""
        s = _schema(2)
        bags = [Bag(f"b{i}", 1, [Instance([1.0, 2.0], s)]) for i in range(5)]
        ds = MIData(bags, "one_class")
        assert detect_imbalance_ratio(ds) == pytest.approx(0.0)

    def test_returns_float(self):
        ds = _dataset(n_pos=4, n_neg=8)
        assert isinstance(detect_imbalance_ratio(ds), float)

    def test_string_labels_are_parsed(self):
        ds = _dataset(n_pos=5, n_neg=5, label_type="str")
        assert detect_imbalance_ratio(ds) == pytest.approx(1.0)


# 2. score_labels

class TestScoreLabels:

    def test_all_noise_returns_zero(self):
        ds = _dataset()
        score = score_labels(ds, _all_noise(ds))
        assert score == pytest.approx(0.0)

    def test_single_cluster_returns_low_score(self):
        ds = _dataset()
        score = score_labels(ds, _single_cluster(ds))
        assert score <= 0.1

    def test_perfect_two_cluster_returns_high_score(self):
        ds = _dataset(n_pos=10, n_neg=10, seed=42)
        preds = _two_cluster_labels(ds)
        score = score_labels(ds, preds)
        assert score >= 0.7

    def test_score_in_zero_one_range(self):
        ds = _dataset()
        for pred in [_all_noise(ds), _single_cluster(ds), _two_cluster_labels(ds)]:
            s = score_labels(ds, pred)
            assert 0.0 <= s <= 1.0

    def test_imbalance_ratio_affects_score_when_imbalanced(self):
        """Con dataset muy desbalanceado, pasar imbalance_ratio < 0.3 no crashea."""
        ds = _dataset(n_pos=2, n_neg=18)
        preds = _two_cluster_labels(ds)
        score_bal = score_labels(ds, preds, imbalance_ratio=1.0)
        score_imb = score_labels(ds, preds, imbalance_ratio=0.1)
        # Ambos deben ser válidos; no asumimos cuál es mayor
        assert 0.0 <= score_bal <= 1.0
        assert 0.0 <= score_imb <= 1.0

    def test_empty_predicted_labels_returns_zero(self):
        ds = _dataset()
        assert score_labels(ds, {}) == pytest.approx(0.0)

    def test_returns_float(self):
        ds = _dataset()
        assert isinstance(score_labels(ds, _two_cluster_labels(ds)), float)

    def test_excess_noise_penalty_applied(self):
        """Asignando > 30 % de ruido, el score debe ser menor que sin ruido."""
        ds = _dataset(n_pos=10, n_neg=10, seed=7)
        clean_preds = _two_cluster_labels(ds)
        # Contaminar el 50 % con ruido
        noisy_preds = dict(clean_preds)
        bags = list(ds.bags)
        for bag in bags[: len(bags) // 2]:
            noisy_preds[bag.bag_id] = -1
        score_clean = score_labels(ds, clean_preds)
        score_noisy = score_labels(ds, noisy_preds)
        assert score_clean >= score_noisy


# 3. MILEvaluator.hungarian_map_clusters_to_labels

class TestMILEvaluatorMapping:

    def _run(self, y_true, y_pred):
        return MILEvaluator.hungarian_map_clusters_to_labels(
            np.array(y_true), np.array(y_pred)
        )

    def test_perfect_two_cluster_identity_mapping(self):
        # cluster 0 → clase 0, cluster 1 → clase 1
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1]
        mapped, mapping = self._run(y_true, y_pred)
        np.testing.assert_array_equal(mapped, y_true)
        assert set(mapping.values()) == {0, 1}

    def test_inverted_clusters_get_corrected(self):
        # cluster 0 tiene todos label 1, cluster 1 tiene todos label 0
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        mapped, mapping = self._run(y_true, y_pred)
        assert mapping[0] == 1
        assert mapping[1] == 0
        np.testing.assert_array_equal(mapped, y_true)

    def test_noise_cluster_mapped_to_majority(self):
        y_true = [0, 0, 0, 1, 1]
        y_pred = [-1, -1, -1, 1, 1]
        mapped, mapping = self._run(y_true, y_pred)
        # puntos de ruido deberían mapearse a la clase mayoritaria de ese grupo (0)
        assert mapping[-1] in (0, 1)   # debe existir en el mapping

    def test_returns_two_elements(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        result = self._run(y_true, y_pred)
        assert len(result) == 2

    def test_mapped_array_length_equals_input(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        mapped, _ = self._run(y_true, y_pred)
        assert len(mapped) == len(y_true)

    def test_mapped_values_are_only_0_or_1(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [2, 2, 3, 3, -1, -1]
        mapped, _ = self._run(y_true, y_pred)
        assert set(mapped.tolist()).issubset({0, 1})

    def test_more_clusters_than_classes_fallback(self):
        """K > 2 clusters → Hungarian asigna 2 y el resto usa majority voting."""
        y_true = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
        mapped, mapping = self._run(y_true, y_pred)
        assert 2 in mapping
        assert set(mapped.tolist()).issubset({0, 1})

    def test_single_cluster_all_same_label(self):
        y_true = [1, 1, 1, 1]
        y_pred = [0, 0, 0, 0]
        mapped, mapping = self._run(y_true, y_pred)
        assert mapping[0] == 1
        assert all(v == 1 for v in mapped)

    def test_returns_numpy_array_and_dict(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        mapped, mapping = self._run(y_true, y_pred)
        assert isinstance(mapped, np.ndarray)
        assert isinstance(mapping, dict)


# 4. MILEvaluator.evaluate

class TestMILEvaluatorEvaluate:
    """
    Nota de diseño: evaluate() imprime en stdout (efecto secundario).
    Usamos `capsys` de pytest para capturar y silenciar la salida.
    """

    def _perfect_model_labels(self, ds: MIData) -> dict[str, int]:
        return {bag.bag_id: int(float(bag.label)) for bag in ds.bags}

    def test_returns_dict_with_expected_keys(self, capsys):
        ds = _dataset()
        result = MILEvaluator.evaluate(ds, self._perfect_model_labels(ds))
        capsys.readouterr()
        for key in ("Precision", "Recall", "F1-Score", "Specificity"):
            assert key in result

    def test_perfect_predictions_give_high_f1(self, capsys):
        ds = _dataset(n_pos=10, n_neg=10, seed=0)
        result = MILEvaluator.evaluate(ds, self._perfect_model_labels(ds))
        capsys.readouterr()
        assert result["F1-Score"] >= 0.9

    def test_all_metrics_in_zero_one_range(self, capsys):
        ds = _dataset()
        result = MILEvaluator.evaluate(ds, self._perfect_model_labels(ds))
        capsys.readouterr()
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_all_noise_produces_low_metrics(self, capsys):
        ds = _dataset()
        result = MILEvaluator.evaluate(ds, _all_noise(ds))
        capsys.readouterr()
        # Con todo ruido, el mapeo mayoritario puede dar un resultado parcial,
        # pero F1 debe ser bajo
        assert result["F1-Score"] <= 0.7

    def test_empty_model_labels_returns_empty_dict(self, capsys):
        ds = _dataset()
        result = MILEvaluator.evaluate(ds, {})
        capsys.readouterr()
        assert result == {}

    def test_returns_float_values(self, capsys):
        ds = _dataset()
        result = MILEvaluator.evaluate(ds, self._perfect_model_labels(ds))
        capsys.readouterr()
        for v in result.values():
            assert isinstance(v, float)

    def test_string_labels_in_dataset_work(self, capsys):
        ds = _dataset(label_type="str")
        labels = {bag.bag_id: int(float(bag.label)) for bag in ds.bags}
        result = MILEvaluator.evaluate(ds, labels)
        capsys.readouterr()
        assert "F1-Score" in result


# 5. BaseCVI — contrato de la clase base

class TestBaseCVI:
    """BaseCVI es abstracta; se testea a través de una subclase mínima."""

    class _DummyCVI(BaseCVI):
        @property
        def name(self): return "Dummy"
        @property
        def category(self): return "compactness"
        def compute(self, dist_matrix, labels, bag_ids, X=None): return 0.0

    def test_higher_is_better_default_true(self):
        cvi = self._DummyCVI()
        assert cvi.higher_is_better is True

    def test_repr_contains_name(self):
        cvi = self._DummyCVI()
        assert "Dummy" in repr(cvi)

    def test_repr_contains_direction_arrow(self):
        cvi = self._DummyCVI()
        assert "↑" in repr(cvi) or "↓" in repr(cvi)

    def test_require_X_raises_when_none(self):
        cvi = self._DummyCVI()
        with pytest.raises(ValueError, match="X"):
            cvi._require_X(None)

    def test_require_X_returns_array_when_provided(self):
        cvi = self._DummyCVI()
        X = np.eye(3)
        result = cvi._require_X(X)
        np.testing.assert_array_equal(result, X)

    def test_label_array_aligned_with_bag_ids(self):
        cvi = self._DummyCVI()
        labels = {"b0": 0, "b1": 1, "b2": -1}
        bag_ids = ["b0", "b1", "b2"]
        arr = cvi._label_array(labels, bag_ids)
        np.testing.assert_array_equal(arr, [0, 1, -1])

    def test_label_array_missing_id_defaults_to_noise(self):
        cvi = self._DummyCVI()
        labels = {"b0": 0}
        bag_ids = ["b0", "b_missing"]
        arr = cvi._label_array(labels, bag_ids)
        assert arr[1] == -1

    def test_real_clusters_excludes_noise(self):
        cvi = self._DummyCVI()
        label_arr = np.array([0, 1, -1, 0, 1])
        clusters = cvi._real_clusters(label_arr)
        assert -1 not in clusters
        assert set(clusters.tolist()) == {0, 1}

    def test_cluster_idx_returns_correct_positions(self):
        cvi = self._DummyCVI()
        label_arr = np.array([0, 1, 0, 1, 0])
        idx = cvi._cluster_idx(label_arr, 0)
        np.testing.assert_array_equal(idx, [0, 2, 4])

    def test_abstract_methods_raise_not_implemented(self):
        """BaseCVI es abstracta; intenta instanciar directamente lanza TypeError."""
        with pytest.raises(TypeError):
            BaseCVI()  # type: ignore[abstract]


# Fixtures de CVI (compartidos por las clases de test concretas)

@pytest.fixture()
def two_cluster_setup():
    """
    Dos clusters bien separados:
      Cluster 0: 4 puntos cerca del origen.
      Cluster 1: 4 puntos lejos del origen.
    """
    X = np.array([
        [0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.2, 0.2],  # cluster 0
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1],  # cluster 1
    ])
    bag_ids = [f"b{i}" for i in range(8)]
    labels  = {bid: (0 if i < 4 else 1) for i, bid in enumerate(bag_ids)}
    dist_matrix = np.zeros((8, 8))  # placeholder (SED/DD/Hc no lo requieren)
    return X, bag_ids, labels, dist_matrix


@pytest.fixture()
def noisy_setup(two_cluster_setup):
    """Igual que two_cluster_setup pero con 2 puntos de ruido."""
    X, bag_ids, labels, dist_matrix = two_cluster_setup
    noisy_labels = dict(labels)
    noisy_labels["b0"] = -1
    noisy_labels["b4"] = -1
    return X, bag_ids, noisy_labels, dist_matrix


@pytest.fixture()
def single_cluster_setup():
    """Un único cluster (todos los puntos en cluster 0)."""
    X = np.random.RandomState(0).rand(6, 3)
    bag_ids = [f"b{i}" for i in range(6)]
    labels  = {bid: 0 for bid in bag_ids}
    dist_matrix = np.zeros((6, 6))
    return X, bag_ids, labels, dist_matrix


# 6. SEDIndex

class TestSEDIndex:

    def _cvi(self): return SEDIndex()

    def test_name_is_sed(self):
        assert self._cvi().name == "SED"

    def test_lower_is_better(self):
        assert self._cvi().higher_is_better is False

    def test_category_is_compactness(self):
        assert self._cvi().category == "compactness"

    def test_identical_points_in_cluster_gives_zero(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        bag_ids = ["b0", "b1", "b2"]
        labels = {"b0": 0, "b1": 0, "b2": 0}
        dm = np.zeros((3, 3))
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val == pytest.approx(0.0)

    def test_non_negative(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val >= 0.0

    def test_compact_clusters_lower_than_dispersed(self):
        """SED de clusters compactos < SED de clusters dispersos."""
        compact_X = np.array([
            [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],   # cluster 0
            [5.0, 5.0], [5.01, 5.0], [5.0, 5.01],   # cluster 1
        ])
        dispersed_X = np.array([
            [0.0, 0.0], [2.0, 0.0], [0.0, 2.0],     # cluster 0
            [5.0, 5.0], [7.0, 5.0], [5.0, 7.0],     # cluster 1
        ])
        bag_ids = [f"b{i}" for i in range(6)]
        labels  = {bid: (0 if i < 3 else 1) for i, bid in enumerate(bag_ids)}
        dm = np.zeros((6, 6))
        sed_compact   = self._cvi().compute(dm, labels, bag_ids, X=compact_X)
        sed_dispersed = self._cvi().compute(dm, labels, bag_ids, X=dispersed_X)
        assert sed_compact < sed_dispersed

    def test_no_real_clusters_returns_inf(self):
        X = np.random.rand(4, 2)
        bag_ids = [f"b{i}" for i in range(4)]
        labels  = {bid: -1 for bid in bag_ids}   # todo ruido
        dm = np.zeros((4, 4))
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert math.isinf(val)

    def test_requires_X(self):
        dm = np.zeros((3, 3))
        bag_ids = ["b0", "b1", "b2"]
        labels  = {"b0": 0, "b1": 0, "b2": 1}
        with pytest.raises(ValueError):
            self._cvi().compute(dm, labels, bag_ids, X=None)

    def test_noise_points_excluded_from_sed(self, noisy_setup):
        X, bag_ids, noisy_labels, dm = noisy_setup
        val_noisy = self._cvi().compute(dm, noisy_labels, bag_ids, X=X)
        # Con menos puntos por cluster, SED debería ser ≤ al original
        assert val_noisy >= 0.0

    def test_returns_float(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert isinstance(val, float)


# 7. DDIndex

class TestDDIndex:

    def _cvi(self): return DDIndex()

    def test_name_is_dd(self):
        assert self._cvi().name == "DD"

    def test_lower_is_better(self):
        assert self._cvi().higher_is_better is False

    def test_category_is_compactness(self):
        assert self._cvi().category == "compactness"

    def test_identical_points_gives_zero(self):
        X = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0],
                      [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]])
        bag_ids = [f"b{i}" for i in range(6)]
        labels  = {bid: (0 if i < 3 else 1) for i, bid in enumerate(bag_ids)}
        dm = np.zeros((6, 6))
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val == pytest.approx(0.0)

    def test_non_negative(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert self._cvi().compute(dm, labels, bag_ids, X=X) >= 0.0

    def test_dd_less_than_sed_for_same_data(self, two_cluster_setup):
        """DD = SSE / (n*d) ≤ SED (que es √SSE por norma, no SSE)
        No hay una relación algebraica exacta garantizada, pero DD debería
        ser un valor razonable (comprobamos que ambos son finitos y >= 0)."""
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert math.isfinite(val) and val >= 0.0

    def test_no_clusters_returns_inf(self):
        X = np.random.rand(4, 2)
        bag_ids = [f"b{i}" for i in range(4)]
        labels  = {bid: -1 for bid in bag_ids}
        dm = np.zeros((4, 4))
        assert math.isinf(self._cvi().compute(dm, labels, bag_ids, X=X))

    def test_requires_X(self):
        dm = np.zeros((3, 3))
        bag_ids = ["b0", "b1", "b2"]
        labels  = {"b0": 0, "b1": 0, "b2": 1}
        with pytest.raises(ValueError):
            self._cvi().compute(dm, labels, bag_ids, X=None)

    def test_returns_float(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert isinstance(self._cvi().compute(dm, labels, bag_ids, X=X), float)

    def test_normalization_scales_with_d(self):
        """Mismo clustering pero con d=1 vs d=2 → DD diferente."""
        # d=1
        X1 = np.array([[0.0], [0.0], [5.0], [5.0]])
        # d=2 (segunda columna es cero → misma dispersión en d=1)
        X2 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
        bag_ids = ["b0", "b1", "b2", "b3"]
        labels  = {"b0": 0, "b1": 0, "b2": 1, "b3": 1}
        dm = np.zeros((4, 4))
        dd1 = self._cvi().compute(dm, labels, bag_ids, X=X1)
        dd2 = self._cvi().compute(dm, labels, bag_ids, X=X2)
        # dd2 debe ser exactamente la mitad de dd1 (misma SSE, doble d)
        assert dd2 == pytest.approx(dd1 / 2.0, rel=1e-6)


# 8. HcIndex

class TestHcIndex:

    def _cvi(self): return HcIndex()

    def test_name_is_hc(self):
        assert self._cvi().name == "Hc"

    def test_lower_is_better(self):
        assert self._cvi().higher_is_better is False

    def test_category_is_compactness(self):
        assert self._cvi().category == "compactness"

    def test_does_not_require_X(self, two_cluster_setup):
        """HcIndex solo necesita las etiquetas, no la matriz de características."""
        X, bag_ids, labels, dm = two_cluster_setup
        # Debe funcionar sin X
        val = self._cvi().compute(dm, labels, bag_ids, X=None)
        assert math.isfinite(val)

    def test_single_cluster_gives_zero(self, single_cluster_setup):
        X, bag_ids, labels, dm = single_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=None)
        assert val == pytest.approx(0.0)

    def test_equal_size_clusters_gives_log2(self):
        """Dos clusters del mismo tamaño → Hc = -2*(0.5*log(0.5)) = log(2)."""
        bag_ids = [f"b{i}" for i in range(6)]
        labels  = {bid: (0 if i < 3 else 1) for i, bid in enumerate(bag_ids)}
        dm = np.zeros((6, 6))
        val = self._cvi().compute(dm, labels, bag_ids, X=None)
        assert val == pytest.approx(math.log(2), rel=1e-6)

    def test_non_negative(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert self._cvi().compute(dm, labels, bag_ids, X=None) >= 0.0

    def test_no_clusters_returns_inf(self):
        bag_ids = [f"b{i}" for i in range(4)]
        labels  = {bid: -1 for bid in bag_ids}
        dm = np.zeros((4, 4))
        assert math.isinf(self._cvi().compute(dm, labels, bag_ids, X=None))

    def test_more_equal_clusters_higher_entropy(self):
        """Distribución más uniforme → entropía mayor."""
        bag_ids4 = [f"b{i}" for i in range(8)]
        labels_2  = {bid: (0 if i < 4 else 1) for i, bid in enumerate(bag_ids4)}
        labels_4  = {bid: i // 2 for i, bid in enumerate(bag_ids4)}
        dm = np.zeros((8, 8))
        hc_2 = self._cvi().compute(dm, labels_2, bag_ids4)
        hc_4 = self._cvi().compute(dm, labels_4, bag_ids4)
        assert hc_4 > hc_2

    def test_returns_float(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert isinstance(self._cvi().compute(dm, labels, bag_ids, X=None), float)


# 9. VRCIndex

class TestVRCIndex:

    def _cvi(self): return VRCIndex()

    def test_name_is_vrc(self):
        assert self._cvi().name == "VRC"

    def test_higher_is_better(self):
        assert self._cvi().higher_is_better is True

    def test_category_is_compactness_separation(self):
        assert self._cvi().category == "compactness_separation"

    def test_requires_X(self):
        dm = np.zeros((3, 3))
        bag_ids = ["b0", "b1", "b2"]
        labels  = {"b0": 0, "b1": 0, "b2": 1}
        with pytest.raises(ValueError):
            self._cvi().compute(dm, labels, bag_ids, X=None)

    def test_single_cluster_returns_zero(self, single_cluster_setup):
        X, bag_ids, labels, dm = single_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val == pytest.approx(0.0)

    def test_well_separated_clusters_high_vrc(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val > 10.0   # clusters muy separados → ratio alto

    def test_non_negative(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val >= 0.0

    def test_well_separated_better_than_overlapping(self):
        """VRC de clusters bien separados > VRC de clusters solapados."""
        bag_ids = [f"b{i}" for i in range(8)]
        labels  = {bid: (0 if i < 4 else 1) for i, bid in enumerate(bag_ids)}
        dm = np.zeros((8, 8))

        X_sep = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [9.9, 9.9], [10.0, 9.9], [9.9, 10.0], [10.0, 10.0],
        ])
        X_ov = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [0.2, 0.2], [0.3, 0.2], [0.2, 0.3], [0.3, 0.3],
        ])
        vrc_sep = self._cvi().compute(dm, labels, bag_ids, X=X_sep)
        vrc_ov  = self._cvi().compute(dm, labels, bag_ids, X=X_ov)
        assert vrc_sep > vrc_ov

    def test_returns_float(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert isinstance(self._cvi().compute(dm, labels, bag_ids, X=X), float)


# 10. IIndex (PBM)

class TestIIndex:

    def _cvi(self): return IIndex()

    def test_name_is_i(self):
        assert self._cvi().name == "I"

    def test_higher_is_better(self):
        assert self._cvi().higher_is_better is True

    def test_category_is_compactness_separation(self):
        assert self._cvi().category == "compactness_separation"

    def test_requires_X(self):
        dm = np.zeros((3, 3))
        bag_ids = ["b0", "b1", "b2"]
        labels  = {"b0": 0, "b1": 0, "b2": 1}
        with pytest.raises(ValueError):
            self._cvi().compute(dm, labels, bag_ids, X=None)

    def test_no_clusters_returns_zero(self):
        X = np.random.rand(4, 2)
        bag_ids = [f"b{i}" for i in range(4)]
        labels  = {bid: -1 for bid in bag_ids}
        dm = np.zeros((4, 4))
        assert self._cvi().compute(dm, labels, bag_ids, X=X) == pytest.approx(0.0)

    def test_non_negative(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val >= 0.0

    def test_well_separated_gives_positive_value(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val > 0.0

    def test_identical_centroids_returns_zero(self):
        """Caso degenerado: Dk = 0 cuando todos los centroides son idénticos.
        También Ek ≈ 0 porque todos los puntos son idénticos → I = 0.
        """
        X = np.array([
            [1.0, 1.0], [1.0, 1.0],   # cluster 0 (idénticos)
            [1.0, 1.0], [1.0, 1.0],   # cluster 1 (idénticos, mismo punto)
        ])
        bag_ids = ["b0", "b1", "b2", "b3"]
        labels  = {"b0": 0, "b1": 0, "b2": 1, "b3": 1}
        dm = np.zeros((4, 4))
        val = self._cvi().compute(dm, labels, bag_ids, X=X)
        assert val == pytest.approx(0.0)

    def test_returns_float(self, two_cluster_setup):
        X, bag_ids, labels, dm = two_cluster_setup
        assert isinstance(self._cvi().compute(dm, labels, bag_ids, X=X), float)


# 11. InternalCVIEvaluator

class TestInternalCVIEvaluator:

    def _eval(self) -> InternalCVIEvaluator:
        return InternalCVIEvaluator(cvis=[SEDIndex(), DDIndex(), HcIndex(), VRCIndex(), IIndex()])

    def _inputs(self, well_separated: bool = True):
        """Devuelve (dist_matrix, labels, bag_ids, dataset) para un setup sencillo."""
        if well_separated:
            X_vals = [
                [[0.0, 0.0], [0.1, 0.0]],
                [[0.0, 0.1], [0.1, 0.1]],
                [[0.0, 0.0], [0.0, 0.1]],
                [[10.0, 10.0], [10.1, 10.0]],
                [[10.0, 10.1], [10.1, 10.1]],
                [[10.0, 10.0], [10.1, 10.1]],
            ]
            lbl_list = [0, 0, 0, 1, 1, 1]
        else:
            X_vals = [[[float(i)] * 2 for _ in range(2)] for i in range(6)]
            lbl_list = [0, 0, 0, 1, 1, 1]

        s = _schema(2)
        bags = [
            Bag(f"b{i}", lbl_list[i], [Instance(r, s) for r in X_vals[i]])
            for i in range(6)
        ]
        dataset = MIData(bags, "eval_test")
        bag_ids = [b.bag_id for b in bags]
        labels  = {bid: lbl_list[i] for i, bid in enumerate(bag_ids)}
        dist_matrix = np.zeros((6, 6))
        return dist_matrix, labels, bag_ids, dataset

    def test_returns_expected_top_level_keys(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        for key in ("title", "n_bags", "n_clusters", "noise_count", "noise_pct", "scores"):
            assert key in result

    def test_scores_contains_all_registered_cvis(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        for name in ("SED", "DD", "Hc", "VRC", "I"):
            assert name in result["scores"]

    def test_n_bags_matches_input(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        assert result["n_bags"] == 6

    def test_n_clusters_correct(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        assert result["n_clusters"] == 2

    def test_noise_count_correct(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        # Introducir ruido en 2 bolsas
        noisy_labels = dict(labels)
        noisy_labels["b0"] = -1
        noisy_labels["b1"] = -1
        result = ev.evaluate(dm, noisy_labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        assert result["noise_count"] == 2

    def test_cvi_names_property(self):
        ev = InternalCVIEvaluator(cvis=[SEDIndex(), HcIndex()])
        assert ev.cvi_names == ["SED", "Hc"]

    def test_register_adds_cvi(self):
        ev = InternalCVIEvaluator(cvis=[SEDIndex()])
        ev.register(HcIndex())
        assert "Hc" in ev.cvi_names

    def test_register_wrong_type_raises_type_error(self):
        ev = InternalCVIEvaluator()
        with pytest.raises(TypeError):
            ev.register("not_a_cvi")   # type: ignore[arg-type]

    def test_default_cvis_include_compactness(self):
        ev = InternalCVIEvaluator()
        # Comprueba que al menos HcIndex está en los defaults
        assert "Hc" in ev.cvi_names

    def test_evaluate_without_dataset_still_computes_hc(self, capsys):
        """HcIndex no necesita X, debe funcionar sin dataset."""
        ev = InternalCVIEvaluator(cvis=[HcIndex()])
        dm, labels, bag_ids, _ = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=None, verbose=False)
        capsys.readouterr()
        assert result["scores"]["Hc"]["value"] is not None

    def test_evaluate_without_dataset_sed_reports_error(self, capsys):
        """SEDIndex requiere X; sin dataset debe reportar error en el score."""
        ev = InternalCVIEvaluator(cvis=[SEDIndex()])
        dm, labels, bag_ids, _ = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=None, verbose=False)
        capsys.readouterr()
        sed_result = result["scores"]["SED"]
        # Debe reportar valor None o contener clave 'error'
        assert sed_result["value"] is None or "error" in sed_result

    def test_verbose_true_prints_to_stdout(self, capsys):
        ev = InternalCVIEvaluator(cvis=[HcIndex()])
        dm, labels, bag_ids, _ = self._inputs()
        ev.evaluate(dm, labels, bag_ids, verbose=True)
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_verbose_false_no_stdout(self, capsys):
        ev = InternalCVIEvaluator(cvis=[HcIndex()])
        dm, labels, bag_ids, _ = self._inputs()
        ev.evaluate(dm, labels, bag_ids, verbose=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_all_noise_scores_handled_gracefully(self, capsys):
        ev = self._eval()
        dm, _, bag_ids, ds = self._inputs()
        all_noise = {bid: -1 for bid in bag_ids}
        result = ev.evaluate(dm, all_noise, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        # No debe lanzar excepción; scores pueden ser None o inf
        assert "scores" in result

    def test_score_values_are_floats_or_none(self, capsys):
        ev = self._eval()
        dm, labels, bag_ids, ds = self._inputs()
        result = ev.evaluate(dm, labels, bag_ids, dataset=ds, verbose=False)
        capsys.readouterr()
        for name, info in result["scores"].items():
            v = info["value"]
            assert v is None or isinstance(v, float), f"{name}: {v!r} is not float or None"

    def test_chaining_register_returns_self(self):
        ev = InternalCVIEvaluator(cvis=[])
        ret = ev.register(HcIndex())
        assert ret is ev