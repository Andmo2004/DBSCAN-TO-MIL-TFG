"""
tests/data/test_bag.py

Tests unitarios para la clase `Bag`.

Bag es el contenedor central del dominio MIL: agrupa instancias bajo un
identificador y una etiqueta. Los tests cubren:
  - construcción y propiedades
  - protocolo de iteración / contenedor / secuencia
  - add_instance y mutación controlada
  - as_matrix (contrato NumPy crítico para el cálculo de distancias)
  - igualdad estructural
  - manejo de bolsas vacías (edge-case ubicuo en DBSCAN)
  - rechazo de instancias con tipo incorrecto
"""

import numpy as np
import pytest

from miclustering.data.bag import Bag
from miclustering.data.instance import Instance

from tests.conftest import make_bag, make_instance, make_schema


# Construcción y propiedades

class TestBagConstruction:

    def test_bag_stores_id_label_and_instances(self, basic_bag):
        assert basic_bag.bag_id == "bag_A"
        assert basic_bag.label == 1
        assert len(basic_bag) == 3

    def test_default_instances_is_empty_list(self):
        bag = Bag(bag_id="b", label=0)
        assert len(bag) == 0

    def test_label_setter_updates_value(self, basic_bag):
        basic_bag.label = 0
        assert basic_bag.label == 0

    def test_bag_id_accepts_string(self):
        bag = Bag(bag_id="my_bag", label=1)
        assert bag.bag_id == "my_bag"

    def test_bag_id_accepts_integer(self):
        bag = Bag(bag_id=42, label=0)
        assert bag.bag_id == 42

    def test_bag_id_accepts_bytes(self):
        bag = Bag(bag_id=b"bag_bytes", label=0)
        assert bag.bag_id == b"bag_bytes"

    def test_non_instance_elements_raise_type_error(self, schema_3f):
        with pytest.raises(TypeError):
            Bag(bag_id="bad", label=0, instances=["not_an_instance"])

    def test_mixed_list_raises_type_error(self, schema_3f):
        good = Instance([1.0, 2.0, 3.0], schema_3f)
        with pytest.raises(TypeError):
            Bag(bag_id="bad", label=0, instances=[good, "string"])


# Protocolo de contenedor y secuencia

class TestBagContainerProtocol:

    def test_len_returns_instance_count(self, basic_bag):
        assert len(basic_bag) == 3

    def test_len_empty_bag_is_zero(self, empty_bag):
        assert len(empty_bag) == 0

    def test_iteration_yields_all_instances(self, basic_bag):
        instances_visited = list(basic_bag)
        assert len(instances_visited) == 3

    def test_getitem_returns_correct_instance(self, basic_bag):
        first = basic_bag[0]
        assert first.get_value(0) == 1.0

    def test_getitem_last_element(self, basic_bag):
        last = basic_bag[2]
        assert last.get_value(0) == 7.0

    def test_getitem_out_of_bounds_raises_index_error(self, basic_bag):
        with pytest.raises(IndexError):
            _ = basic_bag[99]

    def test_get_instance_equivalent_to_getitem(self, basic_bag):
        assert basic_bag.get_instance(1) == basic_bag[1]

    def test_get_instance_negative_index_raises_index_error(self, basic_bag):
        with pytest.raises(IndexError):
            basic_bag.get_instance(-1)

    def test_contains_returns_true_for_member(self, basic_bag):
        inst = basic_bag[0]
        assert inst in basic_bag

    def test_contains_returns_false_for_non_member(self, basic_bag, schema_3f):
        stranger = Instance([99.0, 99.0, 99.0], schema_3f)
        assert stranger not in basic_bag

    def test_get_num_instances_matches_len(self, basic_bag):
        assert basic_bag.get_num_instances() == len(basic_bag)


# Mutación: add_instance

class TestBagAddInstance:

    def test_add_instance_increases_length(self, empty_bag, schema_3f):
        inst = Instance([1.0, 2.0, 3.0], schema_3f)
        empty_bag.add_instance(inst)
        assert len(empty_bag) == 1

    def test_add_instance_appended_at_end(self, basic_bag, schema_3f):
        new_inst = Instance([10.0, 20.0, 30.0], schema_3f)
        basic_bag.add_instance(new_inst)
        assert basic_bag[-1 + len(basic_bag)] == new_inst  # last element
        assert len(basic_bag) == 4

    def test_add_multiple_instances(self, empty_bag, schema_3f):
        for i in range(5):
            empty_bag.add_instance(Instance([float(i)] * 3, schema_3f))
        assert len(empty_bag) == 5

    def test_instances_property_returns_copy(self, basic_bag):
        """Mutating the returned list must not affect the bag."""
        copy = basic_bag.instances
        copy.clear()
        assert len(basic_bag) == 3  # original unchanged


# as_matrix — contrato crítico para distancias

class TestBagAsMatrix:

    def test_shape_is_n_instances_by_n_features(self, basic_bag):
        M = basic_bag.as_matrix()
        assert M.shape == (3, 3)

    def test_values_match_instances(self, basic_bag):
        M = basic_bag.as_matrix()
        np.testing.assert_array_equal(M[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(M[1], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(M[2], [7.0, 8.0, 9.0])

    def test_dtype_is_float64(self, basic_bag):
        assert basic_bag.as_matrix().dtype == np.float64

    def test_empty_bag_returns_empty_array(self, empty_bag):
        M = empty_bag.as_matrix()
        assert M.size == 0

    def test_singleton_bag_shape_is_1_by_n(self, singleton_bag):
        M = singleton_bag.as_matrix()
        assert M.shape == (1, 3)

    def test_matrix_is_not_a_view_of_internal_state(self, basic_bag):
        """as_matrix should return a fresh array, not expose internal data."""
        M = basic_bag.as_matrix()
        M[0, 0] = 999.0
        # Original instance value must remain unchanged
        assert basic_bag[0].get_value(0) == 1.0

    def test_column_consistency_across_rows(self):
        """All rows must have the same number of columns (no ragged arrays)."""
        schema = make_schema(4)
        instances = [Instance([float(i)] * 4, schema) for i in range(5)]
        bag = Bag("b", 0, instances)
        M = bag.as_matrix()
        assert M.shape == (5, 4)


# Igualdad estructural

class TestBagEquality:

    def test_equal_bags_with_same_content(self, schema_3f):
        inst = Instance([1.0, 2.0, 3.0], schema_3f)
        b1 = Bag("id", 0, [inst])
        b2 = Bag("id", 0, [inst])
        assert b1 == b2

    def test_different_id_not_equal(self, schema_3f):
        inst = Instance([1.0, 2.0, 3.0], schema_3f)
        assert Bag("id_a", 0, [inst]) != Bag("id_b", 0, [inst])

    def test_different_label_not_equal(self, schema_3f):
        inst = Instance([1.0, 2.0, 3.0], schema_3f)
        assert Bag("id", 0, [inst]) != Bag("id", 1, [inst])

    def test_inequality_with_non_bag(self, basic_bag):
        assert basic_bag != "not_a_bag"
        assert basic_bag != 42
        assert basic_bag != None  # noqa: E711

# Representación

class TestBagRepresentation:

    def test_str_contains_bag_id(self, basic_bag):
        assert "bag_A" in str(basic_bag)

    def test_repr_contains_id_label_and_count(self, basic_bag):
        r = repr(basic_bag)
        assert "bag_A" in r
        assert "1" in r   # label
        assert "3" in r   # instance count

    def test_str_empty_bag(self, empty_bag):
        assert "empty" in str(empty_bag)
        assert "0" in str(empty_bag)

# make_bag helper consistency

class TestMakeBagHelper:
    """Validate the test helper itself — it underpins all other fixtures."""

    def test_make_bag_default_values(self):
        bag = make_bag()
        assert bag.bag_id == "bag_0"
        assert bag.label == 0
        assert len(bag) == 3

    def test_make_bag_custom_matrix(self):
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        bag = make_bag(values_matrix=matrix, n_features=2)
        M = bag.as_matrix()
        np.testing.assert_array_equal(M, matrix)