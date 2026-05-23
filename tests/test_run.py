"""
tests/test_run_module.py

Tests unitarios para miclustering.run.
No requieren ficheros ARFF en disco: usan datasets sintéticos.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Permitir importar miclustering desde src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.midata import MIData
from miclustering.run import RunConfig, run_json, run_pipeline

def _make_instance(values, label_str=None):
    schema = [Attribute(f"f{i}", "real") for i in range(len(values))]
    return Instance(list(values), schema)


def _make_bag(bag_id, label, n_instances=5, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    instances = [_make_instance(rng.rand(n_features)) for _ in range(n_instances)]
    return Bag(bag_id=bag_id, label=str(label), instances=instances)


def _make_dataset(n_pos=10, n_neg=10, n_features=4, prefix="bag"):
    bags = []
    for i in range(n_pos):
        bags.append(_make_bag(f"{prefix}_pos_{i}", label=1, n_features=n_features, seed=i))
    for i in range(n_neg):
        bags.append(_make_bag(f"{prefix}_neg_{i}", label=0, n_features=n_features, seed=i + 100))
    return MIData(bags, "synthetic")

class TestRunConfig:
    def test_minimal_config(self):
        cfg = RunConfig.from_dict({"dataset": "musk1"})
        assert cfg.dataset == "musk1"
        assert cfg.algorithm == "midbscan"          # default
        assert cfg.distance_metric == "hausdorff"   # default
        assert cfg.scaler == "MinMaxScaler"          # default

    def test_full_config_aliases(self):
        """Acepta claves en español (alias)."""
        cfg = RunConfig.from_dict({
            "dataset": "musk1",
            "medida_de_distancia": "hausdorff_avg",
            "metodo_de_escalado": "StandardScaler",
            "semilla": 7,
            "algoritmo": "miknn",
            "hiperparametros": {"k": 3},
            "optimizar_optuna": False,
        })
        assert cfg.distance_metric == "hausdorff_avg"
        assert cfg.scaler == "StandardScaler"
        assert cfg.seed == 7
        assert cfg.algorithm == "miknn"
        assert cfg.hyperparams == {"k": 3}

    def test_missing_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset"):
            RunConfig.from_dict({"algoritmo": "miknn"})

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Algoritmo"):
            RunConfig.from_dict({"dataset": "x", "algorithm": "svm"})

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="distancia"):
            RunConfig.from_dict({"dataset": "x", "distance_metric": "cosine"})

    def test_none_scaler(self):
        cfg = RunConfig.from_dict({"dataset": "x", "scaler": None})
        assert cfg.scaler is None

    def test_none_scaler_from_string(self):
        cfg = RunConfig.from_dict({"dataset": "x", "scaler": "none"})
        assert cfg.scaler is None

    def test_str_representation(self):
        cfg = RunConfig.from_dict({"dataset": "musk1"})
        s = str(cfg)
        assert "musk1" in s
        assert "midbscan" in s

class TestRunPipeline:
    @pytest.fixture
    def datasets(self):
        train = _make_dataset(n_pos=8, n_neg=8, prefix="train")
        test  = _make_dataset(n_pos=4, n_neg=4, prefix="test")
        return train, test

    def test_midbscan_returns_metrics(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "midbscan",
            "hiperparametros": {"epsilon": 5.0, "min_pts": 2},
        })
        result = run_pipeline(train, test, cfg)
        metrics = result["metrics"]
        for key in ("Accuracy", "Precision", "Recall", "F1-Score", "Specificity"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_miknn_returns_metrics(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "miknn",
            "hiperparametros": {"k": 3},
        })
        result = run_pipeline(train, test, cfg)
        assert 0.0 <= result["metrics"]["F1-Score"] <= 1.0
        # KNN no genera mapping de clusters
        assert result["mapping"] == {}

    def test_mikmeans_returns_metrics(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "mikmeans",
            "hiperparametros": {"k": 2},
        })
        result = run_pipeline(train, test, cfg)
        assert "F1-Score" in result["metrics"]

    def test_mikmedoids_returns_metrics(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "mikmedoids",
            "hiperparametros": {"k": 2},
        })
        result = run_pipeline(train, test, cfg)
        assert "F1-Score" in result["metrics"]

    def test_no_scaler(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "miknn",
            "scaler": None,
            "hiperparametros": {"k": 1},
        })
        result = run_pipeline(train, test, cfg)
        assert "metrics" in result

    def test_standard_scaler(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "miknn",
            "scaler": "StandardScaler",
            "hiperparametros": {"k": 1},
        })
        result = run_pipeline(train, test, cfg)
        assert "metrics" in result

    def test_result_has_hyperparams(self, datasets):
        train, test = datasets
        cfg = RunConfig.from_dict({
            "dataset": "synthetic",
            "algorithm": "miknn",
            "hiperparametros": {"k": 5},
        })
        result = run_pipeline(train, test, cfg)
        assert result["hyperparams"]["k"] == 5

    def test_different_distances(self, datasets):
        train, test = datasets
        for metric in ("hausdorff", "hausdorff_avg", "cauchy_schwarz"):
            cfg = RunConfig.from_dict({
                "dataset": "synthetic",
                "algorithm": "miknn",
                "distance_metric": metric,
                "hiperparametros": {"k": 1},
            })
            result = run_pipeline(train, test, cfg)
            assert "F1-Score" in result["metrics"], f"Fallo con métrica {metric}"


# ── Tests de run_json ─────────────────────────────────────────────────────────


class TestRunJson:
    def _write_config(self, tmpdir, **kwargs):
        path = os.path.join(tmpdir, "config.json")
        payload = {
            "dataset": "synthetic",
            "algoritmo": "miknn",
            "hiperparametros": {"k": 1},
            **kwargs,
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def _make_datasets(self):
        train = _make_dataset(n_pos=6, n_neg=6, prefix="train")
        test  = _make_dataset(n_pos=3, n_neg=3, prefix="test")
        return train, test

    def test_run_json_with_inmemory_data(self, tmp_path):
        train, test = self._make_datasets()
        cfg_path = self._write_config(str(tmp_path))
        result = run_json(cfg_path, train_data=train, test_data=test, verbose=False)
        assert "metrics" in result
        assert "F1-Score" in result["metrics"]

    def test_run_json_saves_output(self, tmp_path):
        train, test = self._make_datasets()
        cfg_path = self._write_config(str(tmp_path))
        out_dir  = str(tmp_path / "results")
        result   = run_json(
            cfg_path, train_data=train, test_data=test,
            output_dir=out_dir, verbose=False
        )
        assert result["output_file"] is not None
        assert os.path.isfile(result["output_file"])
        # El JSON guardado debe ser válido
        with open(result["output_file"]) as fh:
            saved = json.load(fh)
        assert "metrics" in saved

    def test_run_json_config_reflected_in_output(self, tmp_path):
        train, test = self._make_datasets()
        cfg_path = self._write_config(str(tmp_path))
        result   = run_json(cfg_path, train_data=train, test_data=test, verbose=False)
        # El dict original del JSON debe estar en la respuesta
        assert result["config"]["dataset"] == "synthetic"

    def test_run_json_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_json(str(tmp_path / "nonexistent.json"))

    def test_run_json_only_one_dataset_raises(self, tmp_path):
        train, _ = self._make_datasets()
        cfg_path = self._write_config(str(tmp_path))
        with pytest.raises(ValueError, match="AMBOS"):
            run_json(cfg_path, train_data=train, verbose=False)

    def test_run_json_invalid_config_raises(self, tmp_path):
        path = os.path.join(str(tmp_path), "bad.json")
        with open(path, "w") as f:
            json.dump({"algoritmo": "midbscan"}, f)  # falta 'dataset'
        with pytest.raises(ValueError, match="dataset"):
            run_json(path, verbose=False)

    def test_example_json_format(self, tmp_path):
        """Verifica que el formato del example.json original funciona."""
        example = {
            "dataset": "synthetic",
            "medida_de_distancia": "hausdorff",
            "metodo_de_escalado": "MinMaxScaler",
            "semilla": 42,
            "metrica_de_rendimiento_a_optimizar": "F1-Score",
            "algoritmo": "miknn",
            "hiperparametros": {"k": 3},
            "optimizar_optuna": False,
            "optuna_trials": 30,
        }
        path = os.path.join(str(tmp_path), "example.json")
        with open(path, "w") as f:
            json.dump(example, f)
        train, test = self._make_datasets()
        result = run_json(path, train_data=train, test_data=test, verbose=False)
        assert result["metrics"]["F1-Score"] >= 0.0