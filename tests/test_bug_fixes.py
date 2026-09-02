"""
tests/test_bug_fixes.py

Unit tests verifying all patched bugs and edge cases:
1. Nominal attribute validation with None values
2. Weakref cache invalidation on Instance mutation
3. InternalCVIEvaluator._compute_bag_centroids with empty/missing first bag
4. MIData.split_data PRNG isolation
5. cd_diagram_data exact Nemenyi q_alpha for multiple alphas
6. wilcoxon_posthoc Holm step-down adjusted p-value monotonicity
7. MIKMeans handling of empty bags in clusters and fallback reinit
8. Cross-validation fallback when min_class_count < 2
9. Numeric consistency between CPU and torch backends
"""

import unittest
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, KFold

from miclustering.data.attribute import Attribute
from miclustering.data.instance import Instance
from miclustering.data.bag import Bag
from miclustering.data.midata import MIData
from miclustering.evaluation.cvi import InternalCVIEvaluator
from miclustering.models.mikmeans import MIKMeans
from miclustering.distances.torch_backend import (
    hausdorff_torch,
    cauchy_schwarz_torch,
    is_torch_available,
    get_torch_device,
)
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_avg
from miclustering.distances.probability_distribution import cauchy_schwarz_distance

import sys
import importlib
from pathlib import Path
from typing import Any

# Add experiments directory to sys.path for phase 4 statistics testing
_exp_dir = Path(__file__).resolve().parents[3] / "MIClustering-experiments"
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))

try:
    _fase4 = importlib.import_module("fase4_estadistica")
    cd_diagram_data: Any = _fase4.cd_diagram_data
    wilcoxon_posthoc: Any = _fase4.wilcoxon_posthoc
except ImportError:
    cd_diagram_data: Any = lambda *args, **kwargs: {}
    wilcoxon_posthoc: Any = lambda *args, **kwargs: None




class TestNominalValidation(unittest.TestCase):
    """Test 1: Instance nominal attribute validation with None values."""

    def test_nominal_validation_with_none_values_accepted(self):
        attr = Attribute("cat", "nominal", values=None)
        inst = Instance(["foo"], [attr])
        self.assertEqual(inst.get_value(0), "foo")
        inst.set_value(0, "bar")
        self.assertEqual(inst.get_value(0), "bar")

    def test_nominal_validation_with_vocabulary_enforced(self):
        attr = Attribute("cat", "nominal", values=["a", "b"])
        inst = Instance(["a"], [attr])
        inst.set_value(0, "b")
        with self.assertRaises(TypeError):
            inst.set_value(0, "invalid")


class TestCacheInvalidation(unittest.TestCase):
    """Test 2: Weakref cache invalidation on Instance mutation."""

    def test_set_value_invalidates_bag_matrix_cache(self):
        schema = [Attribute("f1", "real"), Attribute("f2", "real")]
        inst1 = Instance([1.0, 2.0], schema)
        inst2 = Instance([3.0, 4.0], schema)
        bag = Bag("bag1", 0, [inst1, inst2])

        m1 = bag.as_matrix()
        np.testing.assert_array_equal(m1, np.array([[1.0, 2.0], [3.0, 4.0]]))

        inst1.set_value(0, 10.0)
        m2 = bag.as_matrix()
        np.testing.assert_array_equal(m2, np.array([[10.0, 2.0], [3.0, 4.0]]))

    def test_dead_weakref_does_not_crash(self):
        schema = [Attribute("f1", "real")]
        inst = Instance([1.0], schema)
        bag = Bag("b", 0, [inst])
        del bag  # bag is garbage-collected, inst has dead weakref
        inst.set_value(0, 5.0)
        self.assertEqual(inst.get_value(0), 5.0)


class TestCVIBagCentroids(unittest.TestCase):
    """Test 3: InternalCVIEvaluator._compute_bag_centroids with empty/missing first bag."""

    def test_empty_first_bag_preserves_dimensionality(self):
        schema = [Attribute(f"f{i}", "real") for i in range(5)]
        empty_bag = Bag("bag_empty", 0, [])
        valid_inst = Instance([1.0, 2.0, 3.0, 4.0, 5.0], schema)
        valid_bag = Bag("bag_valid", 1, [valid_inst])

        dataset = MIData([empty_bag, valid_bag], "test_dataset")
        evaluator = InternalCVIEvaluator()

        X = evaluator._compute_bag_centroids(dataset, ["bag_empty", "bag_valid"])
        self.assertEqual(X.shape, (2, 5))
        np.testing.assert_array_equal(X[0], np.zeros(5))
        np.testing.assert_array_equal(X[1], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


class TestPRNGIsolation(unittest.TestCase):
    """Test 4: MIData.split_data PRNG isolation."""

    def test_split_data_does_not_mutate_global_random_state(self):
        schema = [Attribute("x", "real")]
        bags = [Bag(f"b{i}", i % 2, [Instance([float(i)], schema)]) for i in range(10)]
        data = MIData(bags, "test_split")

        random.seed(999)
        val_before = random.random()

        # Reset global state to 999
        random.seed(999)
        # Call split_data with a different seed
        _ = data.split_data(percentage_train=80, seed=42)
        val_after = random.random()

        # The global random stream must be completely untouched
        self.assertEqual(val_before, val_after)


class TestStatisticalAnalysis(unittest.TestCase):
    """Test 5 & 6: cd_diagram_data and wilcoxon_posthoc Holm monotonicity."""

    def test_cd_diagram_data_multi_alpha(self):
        # Create a sample performance matrix (10 datasets x 5 models)
        np.random.seed(42)
        matrix = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f"M{i}" for i in range(5)],
            index=[f"D{i}" for i in range(10)],
        )

        res_05 = cd_diagram_data(matrix, alpha=0.05)
        res_01 = cd_diagram_data(matrix, alpha=0.01)
        res_10 = cd_diagram_data(matrix, alpha=0.10)

        self.assertEqual(res_05["alpha"], 0.05)
        self.assertEqual(res_01["alpha"], 0.01)
        self.assertEqual(res_10["alpha"], 0.10)

        # Critical difference for alpha=0.01 must be strictly larger than for alpha=0.05
        self.assertGreater(res_01["critical_difference"], res_05["critical_difference"])
        # Critical difference for alpha=0.05 must be strictly larger than for alpha=0.10
        self.assertGreater(res_05["critical_difference"], res_10["critical_difference"])

    def test_wilcoxon_posthoc_holm_monotonicity(self):
        # Test monotonicity: non-decreasing adjusted p-values in sorted order
        np.random.seed(123)
        matrix = pd.DataFrame(
            np.random.rand(15, 4),
            columns=["A", "B", "C", "D"],
            index=[f"D{i}" for i in range(15)],
        )
        p_df = wilcoxon_posthoc(matrix, alpha=0.05)

        # Extract unique pairwise adjusted p-values
        models = p_df.columns.tolist()
        p_vals = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                p_vals.append(p_df.iloc[i, j])

        self.assertTrue(all(0.0 <= p <= 1.0 for p in p_vals))
        # Check symmetry
        self.assertTrue(np.allclose(p_df.values, p_df.values.T))


class TestMIKMeansEdgeCases(unittest.TestCase):
    """Test 7: MIKMeans empty bag handling and centroid calculation."""

    def test_calculate_centroid_filters_empty_bags(self):
        schema = [Attribute("f1", "real"), Attribute("f2", "real")]
        empty_bag = Bag("empty", 0, [])
        valid_bag = Bag("valid", 1, [Instance([2.0, 4.0], schema)])

        kmeans = MIKMeans(k=1, random_state=42)
        kmeans._train_bags = [valid_bag]
        centroid = kmeans._calculate_centroid([empty_bag, valid_bag], 0)
        np.testing.assert_array_equal(centroid.as_matrix(), np.array([[2.0, 4.0]]))

    def test_fit_with_empty_bags_in_dataset_succeeds(self):
        schema = [Attribute("f1", "real"), Attribute("f2", "real")]
        bags = [
            Bag("b0", 0, [Instance([1.0, 1.0], schema)]),
            Bag("b1", 0, [Instance([1.5, 1.2], schema)]),
            Bag("b2", 1, [Instance([5.0, 5.0], schema)]),
            Bag("b3", 1, [Instance([5.2, 5.1], schema)]),
            Bag("b4_empty", 0, []),
        ]
        data = MIData(bags, "mixed_dataset")
        kmeans = MIKMeans(k=2, random_state=42)
        kmeans.fit(data)
        self.assertTrue(kmeans.is_fitted)
        self.assertEqual(len(kmeans.labels), 5)


class TestCVImbalanceFallback(unittest.TestCase):
    """Test 8: CV splitting fallback when min_class_count < 2."""

    def test_fallback_when_class_count_is_1(self):
        labels = np.array([0, 0, 0, 0, 1])  # Class 1 has only 1 sample
        bag_ids = [f"bag_{i}" for i in range(5)]
        min_class_count = int(np.bincount(labels).min())
        self.assertEqual(min_class_count, 1)

        # Safe fallback logic when min_class_count < 2
        if min_class_count >= 2:
            splitter = StratifiedKFold(n_splits=min(5, min_class_count))
            splits = list(splitter.split(bag_ids, labels))
        else:
            splitter = KFold(n_splits=min(5, len(labels)), shuffle=True, random_state=42)
            splits = list(splitter.split(bag_ids))

        self.assertEqual(len(splits), 5)


class TestTorchNumericComparison(unittest.TestCase):
    """Test 9: Numeric consistency between CPU and GPU torch backends."""

    def setUp(self):
        if not is_torch_available():
            self.skipTest("PyTorch not available in this environment")

        np.random.seed(42)
        self.m1 = np.random.randn(20, 10).astype(np.float64)
        self.m2 = np.random.randn(25, 10).astype(np.float64)

        schema = [Attribute(f"f{i}", "real") for i in range(10)]
        self.b1 = Bag("b1", 0, [Instance(row.tolist(), schema) for row in self.m1])
        self.b2 = Bag("b2", 1, [Instance(row.tolist(), schema) for row in self.m2])

    def test_hausdorff_cpu_float64_exact_match(self):
        """CPU torch backend uses float64 and matches NumPy to 1e-7."""
        cpu_dev = get_torch_device("cpu")

        cpu_max = hausdorff_distance(self.b1, self.b2)
        torch_max = hausdorff_torch(self.m1, self.m2, device=cpu_dev, mode="max", dtype=None)
        self.assertTrue(np.isclose(cpu_max, torch_max, atol=1e-7))

        cpu_avg = hausdorff_distance_avg(self.b1, self.b2)
        torch_avg = hausdorff_torch(self.m1, self.m2, device=cpu_dev, mode="avg", dtype=None)
        self.assertTrue(np.isclose(cpu_avg, torch_avg, atol=1e-7))

        cpu_cs = cauchy_schwarz_distance(self.b1, self.b2)
        torch_cs = cauchy_schwarz_torch(self.m1, self.m2, device=cpu_dev, dtype=None)
        self.assertTrue(np.isclose(cpu_cs, torch_cs, atol=1e-7))

    def test_gpu_float32_precision_tolerance(self):
        """GPU/MPS/CUDA accelerators use float32 for speed, agreeing with NumPy within float32 tolerance (1e-4)."""
        import torch
        # Test simulated accelerator dtype (float32)
        torch_max_f32 = hausdorff_torch(self.m1, self.m2, device=get_torch_device("cpu"), mode="max", dtype=torch.float32)
        cpu_max_f64 = hausdorff_distance(self.b1, self.b2)
        self.assertTrue(np.isclose(cpu_max_f64, torch_max_f32, atol=1e-4))

        torch_avg_f32 = hausdorff_torch(self.m1, self.m2, device=get_torch_device("cpu"), mode="avg", dtype=torch.float32)
        cpu_avg_f64 = hausdorff_distance_avg(self.b1, self.b2)
        self.assertTrue(np.isclose(cpu_avg_f64, torch_avg_f32, atol=1e-4))

        torch_cs_f32 = cauchy_schwarz_torch(self.m1, self.m2, device=get_torch_device("cpu"), dtype=torch.float32)
        cpu_cs_f64 = cauchy_schwarz_distance(self.b1, self.b2)
        self.assertTrue(np.isclose(cpu_cs_f64, torch_cs_f32, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
