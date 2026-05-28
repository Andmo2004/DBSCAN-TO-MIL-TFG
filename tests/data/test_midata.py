"""
tests/data/test_midata.py

Tests unitarios para `MIData`.

MIData es el dataset container del pipeline MIL. Sus responsabilidades son:
  1. Almacenar y exponer bolsas de forma segura (sin mutar el estado interno).
  2. Dividir el dataset en train/test de forma reproducible (split_data).
  3. Proveer queries convenientes (positive/negative bags, labels).
  4. Comportarse como una secuencia iterable (len, iter, contains, getitem).

Los tests validan **comportamiento observable**, no detalles de implementación.
"""

import pytest
import random

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag

from tests.conftest import make_bag, make_dataset, make_schema
from miclustering.data.instance import Instance


# Fixtures locales

@pytest.fixture()
def labelled_dataset() -> MIData:
    """Dataset de 4 bolsas con etiquetas int 0 / 1."""
    schema = make_schema(2)
    bags = [
        Bag("b0", 0, [Instance([1.0, 0.0], schema)]),
        Bag("b1", 1, [Instance([0.0, 1.0], schema)]),
        Bag("b2", 0, [Instance([2.0, 0.0], schema)]),
        Bag("b3", 1, [Instance([0.0, 2.0], schema)]),
    ]
    return MIData(bags, "labelled")


@pytest.fixture()
def string_label_dataset() -> MIData:
    """Dataset con etiquetas como strings (típico en ARFF)."""
    schema = make_schema(2)
    bags = [
        Bag("b0", "positive", [Instance([1.0, 0.0], schema)]),
        Bag("b1", "negative", [Instance([0.0, 1.0], schema)]),
        Bag("b2", "1", [Instance([2.0, 0.0], schema)]),
    ]
    return MIData(bags, "string_labels")


# Construcción y propiedades básicas

class TestMIDataConstruction:

    def test_stores_name(self, small_dataset):
        assert small_dataset.name == "synthetic"

    def test_get_num_bags_matches_input(self, small_dataset):
        assert small_dataset.get_num_bags() == 6

    def test_len_matches_get_num_bags(self, small_dataset):
        assert len(small_dataset) == small_dataset.get_num_bags()

    def test_empty_dataset_has_zero_bags(self):
        ds = MIData([], "empty")
        assert len(ds) == 0
        assert ds.get_num_bags() == 0

    def test_repr_contains_name_and_bag_count(self, small_dataset):
        r = repr(small_dataset)
        assert "synthetic" in r
        assert "6" in r

    def test_str_contains_name(self, small_dataset):
        assert "synthetic" in str(small_dataset)


# Protocolo de contenedor/secuencia

class TestMIDataContainerProtocol:

    def test_iteration_yields_all_bags(self, small_dataset):
        bags = list(small_dataset)
        assert len(bags) == 6

    def test_getitem_returns_correct_bag(self, small_dataset):
        bag = small_dataset[0]
        assert bag.bag_id == "bag_0"

    def test_getitem_last_bag(self, small_dataset):
        bag = small_dataset[5]
        assert bag.bag_id == "bag_5"

    def test_getitem_out_of_bounds_raises_index_error(self, small_dataset):
        with pytest.raises(IndexError):
            _ = small_dataset[999]

    def test_get_bag_equivalent_to_getitem(self, small_dataset):
        assert small_dataset.get_bag(2) == small_dataset[2]

    def test_get_bag_negative_index_raises_index_error(self, small_dataset):
        with pytest.raises(IndexError):
            small_dataset.get_bag(-1)

    def test_contains_returns_true_for_member_bag(self, small_dataset):
        bag = small_dataset[0]
        assert bag in small_dataset

    def test_contains_returns_false_for_non_member(self, small_dataset):
        stranger = make_bag(bag_id="stranger_999")
        assert stranger not in small_dataset

    def test_bags_property_returns_copy(self, small_dataset):
        """Mutating the returned list must not affect MIData's internal state."""
        copy = small_dataset.bags
        copy.clear()
        assert len(small_dataset) == 6  # original unchanged

    def test_iteration_order_is_stable(self, small_dataset):
        ids_first = [b.bag_id for b in small_dataset]
        ids_second = [b.bag_id for b in small_dataset]
        assert ids_first == ids_second


# get_labels

class TestMIDataGetLabels:

    def test_returns_list_of_correct_length(self, labelled_dataset):
        labels = labelled_dataset.get_labels()
        assert len(labels) == 4

    def test_labels_order_matches_bags_order(self, labelled_dataset):
        labels = labelled_dataset.get_labels()
        assert labels == [0, 1, 0, 1]

    def test_string_labels_returned_as_strings(self, string_label_dataset):
        labels = string_label_dataset.get_labels()
        assert labels[0] == "positive"
        assert labels[1] == "negative"

    def test_empty_dataset_returns_empty_list(self):
        ds = MIData([], "empty")
        assert ds.get_labels() == []


# get_positive_bags / get_negative_bags

class TestMIDataLabelQueries:

    def test_positive_bags_with_int_labels(self, labelled_dataset):
        pos = labelled_dataset.get_positive_bags()
        assert len(pos) == 2
        assert all(b.label == 1 for b in pos)

    def test_negative_bags_with_int_labels(self, labelled_dataset):
        neg = labelled_dataset.get_negative_bags()
        assert len(neg) == 2
        assert all(b.label == 0 for b in neg)

    def test_positive_bags_with_string_label(self):
        schema = make_schema(1)
        bags = [
            Bag("p", "positive", [Instance([1.0], schema)]),
            Bag("n", "negative", [Instance([0.0], schema)]),
        ]
        ds = MIData(bags, "strings")
        assert len(ds.get_positive_bags()) == 1
        assert len(ds.get_negative_bags()) == 1

    def test_positive_bags_integer_one_string(self):
        schema = make_schema(1)
        bags = [Bag("a", "1", [Instance([1.0], schema)])]
        ds = MIData(bags, "x")
        # "1" is not in the positive_labels set {"1", 1, "positive", "pos", True}
        # String "1" IS in the set → should be positive
        assert len(ds.get_positive_bags()) == 1

    def test_counts_sum_to_total_when_all_labelled(self, labelled_dataset):
        pos = labelled_dataset.get_positive_bags()
        neg = labelled_dataset.get_negative_bags()
        assert len(pos) + len(neg) == len(labelled_dataset)

    def test_positive_bags_empty_dataset(self):
        ds = MIData([], "empty")
        assert ds.get_positive_bags() == []

    def test_negative_bags_empty_dataset(self):
        ds = MIData([], "empty")
        assert ds.get_negative_bags() == []


# split_data — contrato de reproducibilidad y proporciones

class TestMIDataSplitData:

    def test_split_returns_two_midata_objects(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(70)
        assert isinstance(train, MIData)
        assert isinstance(test, MIData)

    def test_split_total_equals_original(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(70)
        assert len(train) + len(test) == len(binary_dataset_10)

    def test_split_70_30_approximate_proportions(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(70, seed=0)
        assert len(train) == 7
        assert len(test) == 3

    def test_split_50_50(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(50, seed=0)
        assert len(train) == 5
        assert len(test) == 5

    def test_split_no_overlap(self, binary_dataset_10):
        """No bag should appear in both train and test."""
        train, test = binary_dataset_10.split_data(70, seed=42)
        train_ids = {b.bag_id for b in train}
        test_ids = {b.bag_id for b in test}
        assert train_ids.isdisjoint(test_ids)

    def test_split_covers_all_bags(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(70, seed=42)
        all_ids = {b.bag_id for b in binary_dataset_10}
        split_ids = {b.bag_id for b in train} | {b.bag_id for b in test}
        assert all_ids == split_ids

    def test_split_reproducible_with_same_seed(self, binary_dataset_10):
        train_a, _ = binary_dataset_10.split_data(70, seed=7)
        train_b, _ = binary_dataset_10.split_data(70, seed=7)
        assert [b.bag_id for b in train_a] == [b.bag_id for b in train_b]

    def test_split_different_seeds_produce_different_partitions(self, binary_dataset_10):
        train_a, _ = binary_dataset_10.split_data(70, seed=1)
        train_b, _ = binary_dataset_10.split_data(70, seed=2)
        # Very unlikely (but not impossible) to be identical for 10 bags
        ids_a = [b.bag_id for b in train_a]
        ids_b = [b.bag_id for b in train_b]
        assert ids_a != ids_b

    def test_split_name_suffix_applied(self, binary_dataset_10):
        train, test = binary_dataset_10.split_data(70)
        assert train.name.endswith("_train")
        assert test.name.endswith("_test")

    def test_split_does_not_mutate_original(self, binary_dataset_10):
        original_len = len(binary_dataset_10)
        binary_dataset_10.split_data(70, seed=0)
        assert len(binary_dataset_10) == original_len

    @pytest.mark.parametrize("pct", [10, 30, 50, 80, 90])
    def test_split_various_percentages_total_is_preserved(self, pct):
        ds = make_dataset(n_bags=20, seed=0)
        train, test = ds.split_data(pct, seed=0)
        assert len(train) + len(test) == 20

    def test_split_100_percent_empty_test(self):
        ds = make_dataset(n_bags=10, seed=0)
        train, test = ds.split_data(100, seed=0)
        assert len(train) == 10
        assert len(test) == 0

    def test_split_zero_percent_empty_train(self):
        ds = make_dataset(n_bags=10, seed=0)
        train, test = ds.split_data(0, seed=0)
        assert len(train) == 0
        assert len(test) == 10


# Igualdad estructural

class TestMIDataEquality:

    def test_equal_datasets_with_same_content(self):
        bags = [make_bag(bag_id=f"b{i}", label=i % 2) for i in range(3)]
        d1 = MIData(bags, "ds")
        d2 = MIData(bags, "ds")
        assert d1 == d2

    def test_different_name_not_equal(self):
        bags = [make_bag()]
        assert MIData(bags, "A") != MIData(bags, "B")

    def test_different_bags_not_equal(self):
        bags_a = [make_bag(bag_id="x")]
        bags_b = [make_bag(bag_id="y")]
        assert MIData(bags_a, "ds") != MIData(bags_b, "ds")

    def test_inequality_with_non_midata(self, small_dataset):
        assert small_dataset != "not_a_dataset"
        assert small_dataset != []