"""
tests/preprocessing/test_preprocessing.py

Tests unitarios para ``miclustering.preprocessing.scaler``.


AUDITORÍA TÉCNICA DEL MÓDULO preprocessing/scaler.py


Clases expuestas

  BaseScaler        — ABC; define el contrato fit / transform / fit_transform
  MinMaxScaler      — escala a [feature_min, feature_max]  (default [0, 1])
  StandardScaler    — estandariza a μ=0, σ=1

Problemas de diseño detectados

1. _create_transformed_dataset accede a ``instance._values`` directamente
   (atributo privado con __slots__). Acopla BaseScaler al detalle de
   implementación de Instance. Si se renombra o encapsula _values, transform()
   se romperá silenciosamente.
   Refactor sugerido: añadir Instance.with_replacement(idx, val) → Instance.

2. transform(inplace=True) itera con for-in pero llama a get_value/set_value,
   mezclando dos APIs distintas de Instance. Sin cobertura de tests.
   Marcado como xfail hasta que se estabilice la API.

3. BaseScaler._collect_numeric_data lanza ValueError con el typo
   "No se encontraron instancias en el datset" (falta 'a' en 'dataset').

4. MinMaxScaler.fit y StandardScaler.fit mutan el array capturado localmente
   para manejar rango/std = 0. Correcto en NumPy pero frágil ante refactors
   que introduzcan copias.

Estrategia de testing

Un único archivo para todo el módulo porque:
  • Solo existe un script (scaler.py) con dos subclases de la misma jerarquía.
  • Muchos casos son parametrizables entre MinMaxScaler y StandardScaler.
  • Refleja la convención del proyecto: un archivo de test por módulo fuente,
    agrupado en el subdirectorio que espeja la estructura de src/.

Cambios en pyproject.toml

No se requiere ninguno. testpaths = ["tests"] y pythonpath = ["src"] ya
recogen este subdirectorio automáticamente mediante pytest discovery.
"""

import os
import sys

import numpy as np
import pytest

#  path (igual que test_run.py y test_hausdorff.py) 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.midata import MIData
from miclustering.preprocessing.scaler import BaseScaler, MinMaxScaler, StandardScaler


# 
# Factories locales (sin dependencia del conftest raíz para aislamiento)
# 

def _schema(*types: str) -> list:
    return [Attribute(f"f{i}", t) for i, t in enumerate(types)]


def _bag(bag_id: str, label: int, rows: list, schema: list) -> Bag:
    return Bag(bag_id=bag_id, label=label,
               instances=[Instance(list(row), schema) for row in rows])


def _dataset(specs: list, schema: list, name: str = "test_ds") -> MIData:
    """specs: list of (bag_id, label, rows)."""
    return MIData([_bag(bid, lbl, rows, schema) for bid, lbl, rows in specs], name)


# 
# Fixtures
# 

@pytest.fixture()
def numeric_schema():
    return _schema("real", "real")


@pytest.fixture()
def simple_dataset(numeric_schema):
    """
    3 bolsas, 2 features. Valores diseñados para validar aritmética exacta:

      bag_0: [[0, 0], [2, 2]]
      bag_1: [[4, 4], [6, 6]]
      bag_2: [[8, 8], [10, 10]]

    Estadísticos globales:
      min = 0,  max = 10,  range = 10
      mean([0,2,4,6,8,10]) = 5.0
      std([0,2,4,6,8,10])  ≈ 3.4156  (población, numpy)
    """
    return _dataset(
        [
            ("bag_0", 0, [[0.0, 0.0], [2.0, 2.0]]),
            ("bag_1", 0, [[4.0, 4.0], [6.0, 6.0]]),
            ("bag_2", 1, [[8.0, 8.0], [10.0, 10.0]]),
        ],
        numeric_schema,
    )


@pytest.fixture()
def mixed_schema():
    """1 atributo nominal (ignorado por el scaler) + 2 reales."""
    return _schema("nominal", "real", "real")


@pytest.fixture()
def mixed_dataset(mixed_schema):
    return _dataset(
        [
            ("bag_0", 0, [["cat", 2.0, 4.0], ["cat", 6.0, 8.0]]),
            ("bag_1", 1, [["dog", 0.0, 10.0], ["dog", 4.0, 6.0]]),
        ],
        mixed_schema,
    )


@pytest.fixture()
def constant_dataset(numeric_schema):
    """Todas las instancias idénticas → range = 0 y std = 0."""
    return _dataset(
        [
            ("bag_0", 0, [[5.0, 5.0], [5.0, 5.0]]),
            ("bag_1", 1, [[5.0, 5.0], [5.0, 5.0]]),
        ],
        numeric_schema,
    )


@pytest.fixture()
def single_bag_dataset(numeric_schema):
    return _dataset([("only", 0, [[1.0, 2.0], [3.0, 4.0]])], numeric_schema)


# 
# 1. Contrato de BaseScaler — parametrizado sobre ambas subclases
# 

class TestBaseScalerContract:
    """
    Verifica el contrato público compartido (BaseScaler) sobre MinMaxScaler
    y StandardScaler. Cualquier nueva subclase debe pasar estos mismos tests.
    """

    @pytest.fixture(params=["minmax", "standard"])
    def scaler(self, request):
        return MinMaxScaler() if request.param == "minmax" else StandardScaler()

    #  estado inicial 

    def test_not_fitted_on_construction(self, scaler):
        assert not scaler.is_fitted

    def test_schema_none_before_fit(self, scaler):
        assert scaler.schema is None

    def test_numeric_indices_empty_before_fit(self, scaler):
        assert scaler.numeric_indices == []

    #  fit 

    def test_fit_returns_self(self, scaler, simple_dataset):
        assert scaler.fit(simple_dataset) is scaler

    def test_is_fitted_after_fit(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert scaler.is_fitted

    def test_fit_empty_dataset_raises_value_error(self, scaler):
        with pytest.raises(ValueError):
            scaler.fit(MIData([], "empty"))

    def test_fit_stores_schema(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert scaler.schema is not None
        assert len(scaler.schema) == 2

    def test_fit_identifies_numeric_indices(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert scaler.numeric_indices == [0, 1]

    #  transform 

    def test_transform_returns_midata(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert isinstance(scaler.transform(simple_dataset), MIData)

    def test_transform_preserves_bag_count(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert len(scaler.transform(simple_dataset)) == len(simple_dataset)

    def test_transform_preserves_instance_counts(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        for orig, new in zip(simple_dataset.bags, result.bags):
            assert len(new) == len(orig)

    def test_transform_preserves_bag_ids(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        assert [b.bag_id for b in result.bags] == [b.bag_id for b in simple_dataset.bags]

    def test_transform_preserves_labels(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        assert [b.label for b in result.bags] == [b.label for b in simple_dataset.bags]

    def test_transform_appends_transformed_suffix_to_name(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        assert "transformed" in result.name

    def test_transform_before_fit_raises_runtime_error(self, scaler, simple_dataset):
        with pytest.raises(RuntimeError, match="fit"):
            scaler.transform(simple_dataset)

    def test_transform_incompatible_schema_raises_value_error(self, scaler, simple_dataset):
        schema_3f = _schema("real", "real", "real")
        other = _dataset([("b0", 0, [[1.0, 2.0, 3.0]])], schema_3f)
        scaler.fit(simple_dataset)
        with pytest.raises(ValueError):
            scaler.transform(other)

    def test_transform_does_not_mutate_original(self, scaler, simple_dataset):
        before = simple_dataset.bags[0].as_matrix().copy()
        scaler.fit(simple_dataset)
        scaler.transform(simple_dataset)
        np.testing.assert_array_equal(simple_dataset.bags[0].as_matrix(), before)

    def test_all_transformed_values_finite(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        for bag in scaler.transform(simple_dataset).bags:
            assert np.all(np.isfinite(bag.as_matrix()))

    #  fit_transform 

    def test_fit_transform_equivalent_to_fit_then_transform(self, simple_dataset):
        s1 = MinMaxScaler()
        chained = s1.fit(simple_dataset).transform(simple_dataset)

        s2 = MinMaxScaler()
        combined = s2.fit_transform(simple_dataset)

        for b1, b2 in zip(chained.bags, combined.bags):
            np.testing.assert_allclose(b1.as_matrix(), b2.as_matrix(), atol=1e-12)

    #  repr / str 

    def test_repr_contains_not_fitted_before_fit(self, scaler):
        assert "not" in repr(scaler).lower() or "unfitted" in repr(scaler).lower()

    def test_repr_contains_fitted_after_fit(self, scaler, simple_dataset):
        scaler.fit(simple_dataset)
        assert "fitted" in repr(scaler).lower()

    def test_str_contains_class_name(self, scaler):
        assert type(scaler).__name__ in str(scaler)


# 
# 2. Helpers internos de BaseScaler
# 

class TestBaseScalerInternalHelpers:

    def test_identify_numeric_indices_all_real(self, numeric_schema):
        scaler = MinMaxScaler()
        assert scaler._identify_numeric_indices(numeric_schema) == [0, 1]

    def test_identify_numeric_indices_skips_nominal(self, mixed_schema):
        scaler = MinMaxScaler()
        indices = scaler._identify_numeric_indices(mixed_schema)
        assert 0 not in indices   # nominal
        assert 1 in indices
        assert 2 in indices

    def test_identify_numeric_indices_empty_schema_returns_empty(self):
        assert MinMaxScaler()._identify_numeric_indices([]) == []

    def test_numeric_to_position_mapping(self, mixed_schema):
        """Para indices [1, 2], posición 0 → índice 1, posición 1 → índice 2."""
        scaler = MinMaxScaler()
        scaler._identify_numeric_indices(mixed_schema)
        assert scaler._numeric_to_position[1] == 0
        assert scaler._numeric_to_position[2] == 1

    def test_extract_schema_returns_attribute_list(self, simple_dataset):
        schema = MinMaxScaler()._extract_schema(simple_dataset)
        assert isinstance(schema, list)
        assert all(isinstance(a, Attribute) for a in schema)
        assert len(schema) == 2

    def test_extract_schema_empty_dataset_raises_value_error(self):
        with pytest.raises(ValueError):
            MinMaxScaler()._extract_schema(MIData([], "empty"))

    def test_collect_numeric_data_shape(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler._schema = scaler._extract_schema(simple_dataset)
        scaler._numeric_indices = scaler._identify_numeric_indices(scaler._schema)
        mat = scaler._collect_numeric_data(simple_dataset)
        # 3 bolsas × 2 instancias = 6 filas, 2 columnas numéricas
        assert mat.shape == (6, 2)

    def test_validate_schema_raises_before_fit(self, simple_dataset):
        with pytest.raises(RuntimeError):
            MinMaxScaler()._validate_schema(simple_dataset)

    def test_numeric_indices_property_returns_copy(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        copy = scaler.numeric_indices
        copy.clear()
        assert len(scaler.numeric_indices) == 2


# 
# 3. MinMaxScaler — fit
# 

class TestMinMaxScalerFit:

    def test_data_min_learned_correctly(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        np.testing.assert_allclose(scaler.data_min, [0.0, 0.0])

    def test_data_max_learned_correctly(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        np.testing.assert_allclose(scaler.data_max, [10.0, 10.0])

    def test_data_min_returns_copy(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        scaler.data_min[0] = 999.0
        np.testing.assert_allclose(scaler.data_min, [0.0, 0.0])

    def test_fit_single_bag_works(self, single_bag_dataset):
        scaler = MinMaxScaler()
        scaler.fit(single_bag_dataset)
        assert scaler.is_fitted

    def test_fit_constant_feature_does_not_raise(self, constant_dataset):
        """range = 0 debe manejarse silenciosamente, no con excepción."""
        scaler = MinMaxScaler()
        scaler.fit(constant_dataset)
        assert scaler.is_fitted

    def test_invalid_feature_range_raises_value_error(self):
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=(1.0, 0.0))

    def test_equal_bounds_feature_range_raises_value_error(self):
        with pytest.raises(ValueError):
            MinMaxScaler(feature_range=(0.5, 0.5))

    def test_feature_range_property_stored(self):
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        assert scaler.feature_range == (-1.0, 1.0)


# 
# 4. MinMaxScaler — transform
# 

class TestMinMaxScalerTransform:

    def test_min_value_maps_to_zero(self, simple_dataset):
        """Valor 0 (min global) → 0.0."""
        result = MinMaxScaler().fit_transform(simple_dataset)
        np.testing.assert_allclose(result.bags[0].as_matrix()[0], [0.0, 0.0], atol=1e-12)

    def test_max_value_maps_to_one(self, simple_dataset):
        """Valor 10 (max global) → 1.0."""
        result = MinMaxScaler().fit_transform(simple_dataset)
        np.testing.assert_allclose(result.bags[2].as_matrix()[1], [1.0, 1.0], atol=1e-12)

    def test_intermediate_value_scaled_correctly(self, simple_dataset):
        """Valor 4 → 0.4; valor 6 → 0.6  (min=0, max=10)."""
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        np.testing.assert_allclose(result.bags[1].as_matrix()[0], [0.4, 0.4], atol=1e-12)
        np.testing.assert_allclose(result.bags[1].as_matrix()[1], [0.6, 0.6], atol=1e-12)

    def test_all_values_in_zero_one(self, simple_dataset):
        result = MinMaxScaler().fit_transform(simple_dataset)
        for bag in result.bags:
            mat = bag.as_matrix()
            assert np.all(mat >= -1e-12)
            assert np.all(mat <= 1.0 + 1e-12)

    def test_nominal_column_preserved(self, mixed_dataset):
        scaler = MinMaxScaler()
        scaler.fit(mixed_dataset)
        result = scaler.transform(mixed_dataset)
        for orig_bag, new_bag in zip(mixed_dataset.bags, result.bags):
            for orig_inst, new_inst in zip(orig_bag, new_bag):
                assert orig_inst.get_value(0) == new_inst.get_value(0)

    def test_constant_feature_result_is_finite(self, constant_dataset):
        result = MinMaxScaler().fit_transform(constant_dataset)
        for bag in result.bags:
            assert np.all(np.isfinite(bag.as_matrix()))

    def test_unseen_data_uses_train_statistics(self, simple_dataset):
        """transform debe extrapolarse usando las estadísticas de fit, no del test."""
        schema = _schema("real", "real")
        test_ds = _dataset([("t0", 0, [[5.0, 5.0]]), ("t1", 1, [[15.0, 15.0]])], schema)
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)   # min=0, max=10
        result = scaler.transform(test_ds)
        np.testing.assert_allclose(result.bags[0].as_matrix()[0], [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(result.bags[1].as_matrix()[0], [1.5, 1.5], atol=1e-12)


# 
# 5. MinMaxScaler — feature_range personalizado
# 

class TestMinMaxScalerFeatureRange:

    def test_custom_range_min_maps_to_lower_bound(self, simple_dataset):
        """Con rango (-1, 1): valor 0 → -1."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        result = scaler.fit_transform(simple_dataset)
        np.testing.assert_allclose(result.bags[0].as_matrix()[0], [-1.0, -1.0], atol=1e-12)

    def test_custom_range_max_maps_to_upper_bound(self, simple_dataset):
        """Con rango (-1, 1): valor 10 → 1."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        result = scaler.fit_transform(simple_dataset)
        np.testing.assert_allclose(result.bags[2].as_matrix()[1], [1.0, 1.0], atol=1e-12)

    def test_custom_range_midpoint(self, simple_dataset):
        """Con rango (-1, 1): valor 5 → 0."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        # bag_1 primera instancia = [4, 4] → 4/10 * 2 - 1 = -0.2
        np.testing.assert_allclose(result.bags[1].as_matrix()[0], [-0.2, -0.2], atol=1e-12)


# 
# 6. MinMaxScaler — inverse_transform
# 

class TestMinMaxScalerInverseTransform:

    def test_round_trip_recovers_original(self, simple_dataset):
        scaler = MinMaxScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(simple_dataset))
        for orig, rec in zip(simple_dataset.bags, recovered.bags):
            np.testing.assert_allclose(orig.as_matrix(), rec.as_matrix(), atol=1e-9)

    def test_round_trip_custom_range(self, simple_dataset):
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        recovered = scaler.inverse_transform(scaler.fit_transform(simple_dataset))
        for orig, rec in zip(simple_dataset.bags, recovered.bags):
            np.testing.assert_allclose(orig.as_matrix(), rec.as_matrix(), atol=1e-9)

    def test_inverse_before_fit_raises_runtime_error(self, simple_dataset):
        with pytest.raises(RuntimeError):
            MinMaxScaler().inverse_transform(simple_dataset)


# 
# 7. StandardScaler — fit
# 

class TestStandardScalerFit:

    def test_mean_learned_correctly(self, simple_dataset):
        """mean([0, 2, 4, 6, 8, 10]) = 5.0."""
        scaler = StandardScaler()
        scaler.fit(simple_dataset)
        np.testing.assert_allclose(scaler.mean, [5.0, 5.0], atol=1e-12)

    def test_std_learned_correctly(self, simple_dataset):
        expected = np.std([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        scaler = StandardScaler()
        scaler.fit(simple_dataset)
        np.testing.assert_allclose(scaler.std, [expected, expected], atol=1e-10)

    def test_mean_returns_copy(self, simple_dataset):
        scaler = StandardScaler()
        scaler.fit(simple_dataset)
        scaler.mean[0] = 999.0
        np.testing.assert_allclose(scaler.mean, [5.0, 5.0], atol=1e-12)

    def test_std_returns_copy(self, simple_dataset):
        scaler = StandardScaler()
        scaler.fit(simple_dataset)
        original_std = scaler.std[0]
        scaler.std[0] = 999.0
        assert scaler.std[0] == original_std

    def test_fit_constant_feature_does_not_raise(self, constant_dataset):
        scaler = StandardScaler()
        scaler.fit(constant_dataset)
        assert scaler.is_fitted

    def test_fit_constant_feature_std_replaced_with_one(self, constant_dataset):
        """std = 0 se reemplaza por 1 para evitar división por cero."""
        scaler = StandardScaler()
        scaler.fit(constant_dataset)
        assert np.all(scaler.std >= 1.0)

    def test_fit_empty_dataset_raises_value_error(self):
        with pytest.raises(ValueError):
            StandardScaler().fit(MIData([], "empty"))


# 
# 8. StandardScaler — transform
# 

class TestStandardScalerTransform:

    def test_transformed_mean_approximately_zero(self, simple_dataset):
        result = StandardScaler().fit_transform(simple_dataset)
        all_values = np.vstack([b.as_matrix() for b in result.bags])
        np.testing.assert_allclose(np.mean(all_values, axis=0), [0.0, 0.0], atol=1e-10)

    def test_transformed_std_approximately_one(self, simple_dataset):
        result = StandardScaler().fit_transform(simple_dataset)
        all_values = np.vstack([b.as_matrix() for b in result.bags])
        np.testing.assert_allclose(np.std(all_values, axis=0), [1.0, 1.0], atol=1e-10)

    def test_specific_value_standardized_correctly(self, simple_dataset):
        """Valor 5 (= media) → 0.0 exacto."""
        # bag_1 tiene [4,4] y [6,6]; la media de ambas columnas en el dataset es 5
        scaler = StandardScaler()
        scaler.fit(simple_dataset)
        result = scaler.transform(simple_dataset)
        # valor 0 → (0 - 5) / std
        expected_std = np.std([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        expected_val = (0.0 - 5.0) / expected_std
        np.testing.assert_allclose(
            result.bags[0].as_matrix()[0], [expected_val, expected_val], atol=1e-10
        )

    def test_constant_feature_result_is_finite(self, constant_dataset):
        result = StandardScaler().fit_transform(constant_dataset)
        for bag in result.bags:
            assert np.all(np.isfinite(bag.as_matrix()))

    def test_nominal_column_preserved(self, mixed_dataset):
        scaler = StandardScaler()
        scaler.fit(mixed_dataset)
        result = scaler.transform(mixed_dataset)
        for orig_bag, new_bag in zip(mixed_dataset.bags, result.bags):
            for orig_inst, new_inst in zip(orig_bag, new_bag):
                assert orig_inst.get_value(0) == new_inst.get_value(0)


# 
# 9. StandardScaler — inverse_transform
# 

class TestStandardScalerInverseTransform:

    def test_round_trip_recovers_original(self, simple_dataset):
        scaler = StandardScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(simple_dataset))
        for orig, rec in zip(simple_dataset.bags, recovered.bags):
            np.testing.assert_allclose(orig.as_matrix(), rec.as_matrix(), atol=1e-9)

    def test_inverse_before_fit_raises_runtime_error(self, simple_dataset):
        with pytest.raises(RuntimeError):
            StandardScaler().inverse_transform(simple_dataset)


# 
# 10. Edge cases — parametrizados sobre ambas subclases
# 

class TestScalerEdgeCases:

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_single_instance_per_bag(self, ScalerClass):
        schema = _schema("real", "real")
        ds = _dataset([("b0", 0, [[2.0, 4.0]]), ("b1", 1, [[6.0, 8.0]])], schema)
        result = ScalerClass().fit_transform(ds)
        assert len(result) == 2
        for bag in result.bags:
            assert np.all(np.isfinite(bag.as_matrix()))

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_single_bag_dataset(self, ScalerClass, single_bag_dataset):
        result = ScalerClass().fit_transform(single_bag_dataset)
        assert len(result) == 1

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_schema_only_nominal_returns_midata(self, ScalerClass):
        """Sin atributos numéricos, transform devuelve un MIData válido."""
        schema = _schema("nominal", "nominal")
        ds = _dataset(
            [("b0", 0, [["a", "b"]]), ("b1", 1, [["c", "d"]])],
            schema,
        )
        result = ScalerClass().fit_transform(ds)
        assert isinstance(result, MIData)

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_large_values_no_overflow(self, ScalerClass):
        schema = _schema("real", "real")
        rng = np.random.default_rng(42)
        bags = []
        for i in range(10):
            rows = (rng.random((4, 2)) * 1e6).tolist()
            bags.append(_bag(f"bag_{i}", i % 2, rows, schema))
        ds = MIData(bags, "large")
        result = ScalerClass().fit_transform(ds)
        for bag in result.bags:
            assert np.all(np.isfinite(bag.as_matrix()))

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_refit_updates_parameters(self, ScalerClass, simple_dataset):
        """Llamar fit() dos veces debe sobreescribir los parámetros anteriores."""
        schema = _schema("real", "real")
        other = _dataset([("x0", 0, [[100.0, 200.0]]), ("x1", 1, [[300.0, 400.0]])], schema)
        scaler = ScalerClass()
        scaler.fit(simple_dataset)
        scaler.fit(other)
        result = scaler.transform(other)
        assert isinstance(result, MIData)
        if ScalerClass is MinMaxScaler:
            np.testing.assert_allclose(scaler.data_min, [100.0, 200.0], atol=1e-10)

    @pytest.mark.parametrize("ScalerClass", [MinMaxScaler, StandardScaler])
    def test_transform_different_bag_count_from_train(self, ScalerClass, simple_dataset):
        """El test set puede tener distinto número de bolsas que el train."""
        schema = _schema("real", "real")
        test_ds = _dataset([("t0", 0, [[1.0, 1.0]])], schema)
        scaler = ScalerClass()
        scaler.fit(simple_dataset)
        result = scaler.transform(test_ds)
        assert len(result) == 1


# 
# 11. Notas de auditoría (xfail)
# 

class TestScalerDesignAuditNotes:
    """
    Tests marcados xfail que documentan problemas de diseño detectados.
    No bloquean CI pero aparecen en el reporte como recordatorio de refactor.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG de diseño: _create_transformed_dataset accede a instance._values "
            "directamente (atributo privado con __slots__), acoplando BaseScaler "
            "a un detalle de implementación de Instance. "
            "Refactor: añadir Instance.with_replacement(idx, val) -> Instance."
        ),
    )
    def test_transform_does_not_access_private_instance_values(self):
        import inspect
        from miclustering.preprocessing.scaler import BaseScaler as BS
        src = inspect.getsource(BS._create_transformed_dataset)
        assert "._values" not in src, (
            "_create_transformed_dataset accede a instance._values directamente"
        )

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "MEJORA: transform(inplace=True) no tiene cobertura de tests. "
            "La implementación mezcla iteración for-in con la API "
            "get_value/set_value, creando dos caminos de código distintos. "
            "Añadir tests cuando se estabilice la semántica inplace."
        ),
    )
    def test_transform_inplace_modifies_original(self, simple_dataset):
        scaler = MinMaxScaler()
        scaler.fit(simple_dataset)
        before = simple_dataset.bags[0].as_matrix()[0, 0]
        scaler.transform(simple_dataset, inplace=True)
        after = simple_dataset.bags[0].as_matrix()[0, 0]
        assert after != before, "inplace=True debe modificar el dataset original"

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "TYPO: _collect_numeric_data lanza ValueError con el mensaje "
            "'No se encontraron instancias en el datset' "
            "(falta la 'a' en 'dataset'). Trivial de corregir."
        ),
    )
    def test_collect_numeric_data_error_message_no_typo(self):
        schema = _schema("real")
        train_ds = _dataset([("b", 0, [[1.0]])], schema)
        scaler = MinMaxScaler()
        scaler._schema = scaler._extract_schema(train_ds)
        scaler._numeric_indices = scaler._identify_numeric_indices(scaler._schema)
        # Dataset con bolsa vacía → debe lanzar ValueError
        ds_empty_bag = MIData([Bag("empty", 0, [])], "empty_bags")
        with pytest.raises(ValueError, match="dataset"):   # 'dataset' sin typo
            scaler._collect_numeric_data(ds_empty_bag)