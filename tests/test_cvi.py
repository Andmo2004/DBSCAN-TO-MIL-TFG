"""
tests/test_cvi.py
 
Test rápido para verificar la implementación de SEDIndex, DDIndex y HcIndex.
Usa los parámetros óptimos ya conocidos de datasets_config.
Llama a cada CVI directamente sin pasar por InternalCVIEvaluator.
"""

import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from miclustering.data.midata import MIData 
from miclustering.models.midbscan import MIDBSCAN 
from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler 
from miclustering.distances.hausdorff import hausdorff_distance 
from miclustering.distances.cauchy_schwarz import cauchy_schwarz_distance 
from miclustering.distances.distance_matrix import compute_distance_matrix 
from miclustering.evaluation.cvi import SEDIndex, HcIndex, DDIndex 

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test_cvi")

# ── Parámetros del dataset a probar ──────────────────────────────────────────
# Cambia este dict para probar otro dataset

DATASET = {
    "dataset_name": "musk1",
    "best_scaler":  MinMaxScaler,
    "best_distance":"hausdorff",
    "best_eps":     2.1673,
    "best_min_pts": 2,
}

DATASETS_DIR = os.path.join(project_root, "datasets")
 
# CVIs a testear: (instancia, necesita_X, criterio)
CVIS = [
    (SEDIndex(), True,  "↓ mejor"),
    (DDIndex(),  True,  "↓ mejor"),
    (HcIndex(),  False, "↓ mejor"),
]

def sanity_checks(name: str, value: float):
    assert not np.isnan(value),  f"{name}: no debe ser NaN"
    assert not np.isinf(value),  f"{name}: no debe ser inf (hay clusters reales)"
    assert value >= 0.0,         f"{name}: debe ser >= 0"

# ── Test ──────────────────────────────────────────────────────────────────────
 
def test_cvis(config: dict):
    name       = config["dataset_name"]
    scaler_cls = config["best_scaler"]
    metric     = config["best_distance"]
    eps        = config["best_eps"]
    min_pts    = config["best_min_pts"]
 
    logger.info(f"Dataset  : {name}")
    logger.info(f"Scaler   : {scaler_cls.__name__}")
    logger.info(f"Métrica  : {metric}  eps={eps}  min_pts={min_pts}")
 
    # 1. Cargar y escalar
    path         = os.path.join(DATASETS_DIR, f"{name}.arff")
    dataset      = MIData.from_arff(path)
    train, _     = dataset.split_data(percentage_train=70, seed=42)
    train_scaled = scaler_cls().fit_transform(train)
    logger.info(f"Bolsas train : {train_scaled.get_num_bags()}")
 
    # 2. Entrenar MIDBSCAN
    model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
    model.fit(train_scaled)
 
    stats = model.get_statistics()
    logger.info(f"Clusters : {stats['num_clusters']}  "
                f"Ruido: {stats['noise_percentage']:.1f}%")
 
    if stats["num_clusters"] == 0:
        logger.warning("Sin clusters reales — no se pueden calcular CVIs.")
        return
 
    # 3. Inputs comunes
    labels  = model.labels
    bag_ids = [bag.bag_id for bag in train_scaled.bags]
    dm      = np.zeros((len(bag_ids), len(bag_ids)))  # placeholder (SED/DD/Hc no lo usan)
    X       = np.array([
        np.mean(bag.as_matrix(), axis=0)
        for bag in train_scaled.bags
    ])
    logger.info(f"X shape  : {X.shape}")
 
    # 4. Calcular y verificar cada CVI
    results = {}
    for cvi, needs_X, criterio in CVIS:
        x_arg = X if needs_X else None
        value = cvi.compute(dm, labels, bag_ids, X=x_arg)
        sanity_checks(cvi.name, value)
        results[cvi.name] = (value, criterio)
 
    # 5. Reporte
    print(f"\n{'='*45}")
    print(f"  CVIs internos — {name}")
    print(f"{'='*45}")
    print(f"  Clusters : {stats['num_clusters']}")
    print(f"  Ruido    : {stats['noise_percentage']:.1f}%")
    print(f"  {'─'*41}")
    for cvi_name, (value, criterio) in results.items():
        print(f"  {cvi_name:<6} : {value:>12.6f}  ({criterio})")
    print(f"{'='*45}\n")
 
    logger.info("Todos los sanity checks pasaron.")
    return results


if __name__ == "__main__":
    test_cvis(DATASET)