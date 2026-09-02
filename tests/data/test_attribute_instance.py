"""
tests/data/test_attribute_instance.py

Tests unitarios para `Attribute` e `Instance`.

Estrategia: un fichero por par de clases atómicas (Attribute e Instance están
íntimamente relacionadas: Instance depende de un schema de Attributes).
Ambas son value-objects sin estado mutable relevante → tests centrados en
construcción, propiedades y comportamiento observable.
"""

import pytest

from miclustering.data.attribute import Attribute
from miclustering.data.instance import Instance
from miclustering.data.bag import Bag

from tests.conftest import make_schema

# Attribute

class TestAttribute:
    """Attribute es un descriptor inmutable de columna."""

    #  Construcción 

    def test_basic_construction_stores_name_and_type(self):
        attr = Attribute("feature_x", "real")
        assert attr.name == "feature_x"
        assert attr.type == "real"

    def test_optional_fields_default_to_none(self):
        attr = Attribute("x", "real")
        assert attr.values is None
        assert attr.data_format is None
        assert attr.val_range is None

    def test_nominal_attribute_stores_values_list(self):
        attr = Attribute("label", "nominal", values=["neg", "pos"])
        assert attr.values == ["neg", "pos"]

    def test_date_attribute_stores_data_format(self):
        attr = Attribute("ts", "date", data_format="yyyy-MM-dd")
        assert attr.data_format == "yyyy-MM-dd"

    def test_range_attribute_stores_val_range(self):
        attr = Attribute("score", "real", val_range=(0.0, 1.0))
        assert attr.val_range == (0.0, 1.0)

    #  Representación 

    def test_repr_contains_name_and_type(self):
        attr = Attribute("my_feat", "integer")
        r = repr(attr)
        assert "my_feat" in r
        assert "integer" in r

    #  Inmutabilidad vía __slots__

    def test_attribute_uses_slots_no_dict(self):
        attr = Attribute("x", "real")
        assert not hasattr(attr, "__dict__")

    def test_cannot_add_arbitrary_attributes(self):
        attr = Attribute("x", "real")
        with pytest.raises(AttributeError):
            attr.new_field = "oops"  # type: ignore[attr-defined]

    #  Tipos soportados (smoke) 

    @pytest.mark.parametrize("attr_type", ["real", "integer", "string", "nominal", "date"])
    def test_all_supported_types_construct_without_error(self, attr_type):
        attr = Attribute("col", attr_type)
        assert attr.type == attr_type


# Instance

class TestInstance:
    """Instance almacena un vector de valores alineado con un schema."""

    #  Construcción 

    def test_construction_stores_values_and_schema(self, schema_3f, basic_instance):
        assert basic_instance.values == [1.0, 2.0, 3.0]
        assert basic_instance.schema is schema_3f

    def test_default_weight_is_one(self, basic_instance):
        assert basic_instance.weight == 1.0

    def test_custom_weight_is_stored(self, schema_3f):
        inst = Instance([1.0, 2.0], make_schema(2), weight=0.5)
        assert inst.weight == 0.5

    def test_weight_setter_updates_value(self, basic_instance):
        basic_instance.weight = 2.0
        assert basic_instance.weight == 2.0

    #  Acceso a valores 

    def test_get_value_returns_correct_element(self, basic_instance):
        assert basic_instance.get_value(0) == 1.0
        assert basic_instance.get_value(2) == 3.0

    def test_set_value_updates_element(self, basic_instance):
        basic_instance.set_value(1, 99.0)
        assert basic_instance.get_value(1) == 99.0

    def test_set_value_out_of_bounds_raises_index_error(self, basic_instance):
        with pytest.raises(IndexError):
            basic_instance.set_value(10, 0.0)

    def test_set_value_negative_index_raises_index_error(self, basic_instance):
        with pytest.raises(IndexError):
            basic_instance.set_value(-1, 0.0)

    #  num_attributes 

    def test_num_attributes_matches_schema_length(self, basic_instance, schema_3f):
        assert basic_instance.num_attributes() == len(schema_3f)

    def test_num_attributes_single_feature(self):
        schema = make_schema(1)
        inst = Instance([42.0], schema)
        assert inst.num_attributes() == 1

    #  Igualdad 

    def test_equality_same_values_and_schema(self, schema_3f):
        i1 = Instance([1.0, 2.0, 3.0], schema_3f)
        i2 = Instance([1.0, 2.0, 3.0], schema_3f)
        assert i1 == i2

    def test_inequality_different_values(self, schema_3f):
        i1 = Instance([1.0, 2.0, 3.0], schema_3f)
        i2 = Instance([1.0, 2.0, 9.9], schema_3f)
        assert i1 != i2

    def test_inequality_with_non_instance(self, basic_instance):
        assert basic_instance != "not an instance"
        assert basic_instance != 42

    #  Representación 

    def test_repr_contains_values(self, basic_instance):
        r = repr(basic_instance)
        assert "1.0" in r

    #  Validación de tipo vía set_value 

    def test_set_integer_value_on_real_attribute_accepted(self):
        schema = [Attribute("x", "real")]
        inst = Instance([1.0], schema)
        inst.set_value(0, 5)  # int is accepted for 'real'
        assert inst.get_value(0) == 5

    def test_set_string_value_on_string_attribute_accepted(self):
        schema = [Attribute("s", "string")]
        inst = Instance(["hello"], schema)
        inst.set_value(0, "world")
        assert inst.get_value(0) == "world"

    def test_set_nominal_value_not_in_vocabulary_raises_type_error(self):
        schema = [Attribute("cat", "nominal", values=["a", "b"])]
        inst = Instance(["a"], schema)
        with pytest.raises(TypeError):
            inst.set_value(0, "c")  # 'c' not in vocabulary

    def test_set_valid_nominal_value_accepted(self):
        schema = [Attribute("cat", "nominal", values=["a", "b"])]
        inst = Instance(["a"], schema)
        inst.set_value(0, "b")
        assert inst.get_value(0) == "b"

    def test_set_nominal_value_when_values_is_none_accepted(self):
        schema = [Attribute("cat", "nominal", values=None)]
        inst = Instance(["a"], schema)
        inst.set_value(0, "arbitrary_string")
        assert inst.get_value(0) == "arbitrary_string"

    def test_set_value_invalidates_parent_bag_cache(self):
        schema = [Attribute("x", "real"), Attribute("y", "real")]
        inst1 = Instance([1.0, 2.0], schema)
        inst2 = Instance([3.0, 4.0], schema)
        bag = Bag("test_bag", 0, [inst1, inst2])

        # Generate cache
        mat_before = bag.as_matrix()
        assert mat_before[0, 0] == 1.0

        # Mutate instance via set_value
        inst1.set_value(0, 99.0)

        # Cache must be invalidated, new matrix must reflect update
        mat_after = bag.as_matrix()
        assert mat_after[0, 0] == 99.0

    def test_orphaned_instance_set_value_does_not_crash(self):
        schema = [Attribute("x", "real")]
        inst = Instance([1.0], schema)
        # No parent bag attached
        inst.set_value(0, 42.0)
        assert inst.get_value(0) == 42.0

    #  __slots__ 

    def test_instance_uses_slots_no_dict(self, basic_instance):
        assert not hasattr(basic_instance, "__dict__")