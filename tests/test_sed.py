"""
tests/test_sed.py

Test rápido para verificar la implementación de SEDIndex.
Usa los parámetros óptimos ya conocidos de datasets_config.
No pasa por MILEvaluator ni por InternalCVIEvaluator — llama a
SEDIndex.compute() directamente para aislar la lógica del índice.
"""

import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np

from data.midata import MIData
from models.midbscan import MIDBSCAN
from preprocessing.scaler import MinMaxScaler, StandardScaler
from distances.hausdorff import hausdorff_distance
from distances.cauchy_schwarz import cauchy_schwarz_distance
from distances.distance_matrix import compute_distance_matrix
from evaluation.internal_cvi import SEDIndex, InternalCVIEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test_sed")

# ── Parámetros del dataset a probar ──────────────────────────────────────────
# Cambia este dict para probar otro dataset

DATASET = {
    "dataset_name": "musk1",
    "best_scaler":  MinMaxScaler,
    "best_distance":"hausdorff",
    "best_eps":     2.1673,
    "best_min_pts": 2,
}

DATASETS_DIR = "datasets"

# ── Helpers ───────────────────────────────────────────────────────────────────

METRIC_MAP = {
    "hausdorff":     hausdorff_distance,
    "cauchy_schwarz": cauchy_schwarz_distance,
}

# ── Test ──────────────────────────────────────────────────────────────────────

def test_sed(config: dict):
    name       = config["dataset_name"]
    scaler_cls = config["best_scaler"]
    metric     = config["best_distance"]
    eps        = config["best_eps"]
    min_pts    = config["best_min_pts"]

    logger.info(f"Dataset      : {name}")
    logger.info(f"Scaler       : {scaler_cls.__name__}")
    logger.info(f"Métrica      : {metric}")
    logger.info(f"eps={eps}  min_pts={min_pts}")

    # 1. Cargar y escalar
    path    = os.path.join(DATASETS_DIR, f"{name}.arff")
    dataset = MIData.from_arff(path)
    train, _ = dataset.split_data(percentage_train=70, seed=42)

    scaler  = scaler_cls()
    train_scaled = scaler.fit_transform(train)
    logger.info(f"Bolsas train : {train_scaled.get_num_bags()}")

    # 2. Entrenar MIDBSCAN
    model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
    model.fit(train_scaled)

    stats = model.get_statistics()
    logger.info(f"Clusters     : {stats['num_clusters']}")
    logger.info(f"Ruido        : {stats['noise_points_count']} "
                f"({stats['noise_percentage']:.1f}%)")

    # 3. Preparar inputs para SEDIndex
    labels  = model.labels                              # {bag_id: cluster_id}
    bag_ids = [bag.bag_id for bag in train_scaled.bags]

    # X: centroides de bolsas (N × n_features)
    X = np.array([
        np.mean(bag.as_matrix(), axis=0)
        for bag in train_scaled.bags
    ])
    logger.info(f"X shape      : {X.shape}")

    # dist_matrix: solo si la necesitamos para la firma — SED no la usa
    # pero la pasamos igualmente para respetar la interfaz
    dist_matrix = np.zeros((len(bag_ids), len(bag_ids)))  # placeholder

    # 4. Calcular SED directamente
    sed = SEDIndex()
    value = sed.compute(dist_matrix, labels, bag_ids, X=X)

    print(f"\n{'='*45}")
    print(f"  SED — {name}")
    print(f"{'='*45}")
    print(f"  Clusters : {stats['num_clusters']}")
    print(f"  Ruido    : {stats['noise_percentage']:.1f}%")
    print(f"  SED      : {value:.6f}  (↓ mejor)")
    print(f"{'='*45}\n")

    # 5. Sanity checks básicos
    assert value >= 0.0, "SED debe ser >= 0"
    assert not np.isnan(value), "SED no debe ser NaN"
    assert not np.isinf(value), "SED no debe ser inf (hay clusters reales)"

    logger.info("Todos los sanity checks pasaron.")
    return value


if __name__ == "__main__":
    test_sed(DATASET)