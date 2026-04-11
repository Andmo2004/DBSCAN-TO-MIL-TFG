"""
tests/test_full_evaluation.py

Evaluación completa para un dataset: CVIs internos + métricas externas.

CVIs internos (sin ground-truth):
  Grupo 1 — Solo Compactibilidad:
    SED, DD, Hc
  Grupo 2 — Compactibilidad + Separación:
    VRC, I

Métricas externas (con ground-truth):
    Precision, Recall, F1-Score, Specificity
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
from evaluation.cvi import SEDIndex, DDIndex, HcIndex, VRCIndex, IIndex
from evaluation.bcm import MILEvaluator

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test_full_evaluation")

# ── Dataset a evaluar ─────────────────────────────────────────────────────────

DATASET = {
    "dataset_name": "musk1",
    "arff_name":    "musk1",
    "best_scaler":  MinMaxScaler,
    "best_distance":"hausdorff",
    "best_eps":     2.1673,
    "best_min_pts": 2,
}

DATASETS_DIR = "datasets"

METRIC_FUNC = {
    "hausdorff":      hausdorff_distance,
    "cauchy_schwarz": cauchy_schwarz_distance,
}

# CVIs internos: (instancia, necesita_X)
INTERNAL_CVIS = [
    (SEDIndex(), True),
    (DDIndex(),  True),
    (HcIndex(),  False),
    (VRCIndex(), True),
    (IIndex(),   True),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_X(dataset: MIData, bag_ids: list) -> np.ndarray:
    """Centroides de bolsas (N × n_features) alineados con bag_ids."""
    bag_index = {bag.bag_id: bag for bag in dataset.bags}
    return np.array([
        np.mean(bag_index[bid].as_matrix(), axis=0)
        for bid in bag_ids
    ])

# ── Evaluación ────────────────────────────────────────────────────────────────

def run_evaluation(config: dict):
    name      = config["dataset_name"]
    arff_name = config["arff_name"]
    scaler_cls= config["best_scaler"]
    metric    = config["best_distance"]
    eps       = config["best_eps"]
    min_pts   = config["best_min_pts"]

    # 1. Cargar y dividir
    path    = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
    dataset = MIData.from_arff(path)
    train, test = dataset.split_data(percentage_train=70, seed=42)

    # 2. Escalar
    scaler       = scaler_cls()
    train_scaled = scaler.fit_transform(train)
    test_scaled  = scaler.transform(test)

    # 3. Entrenar y predecir
    model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
    model.fit(train_scaled)
    test_labels = model.predict(test_scaled)

    stats = model.get_statistics()

    # 4. CVIs internos (sobre train)
    bag_ids = [bag.bag_id for bag in train_scaled.bags]
    dm      = np.zeros((len(bag_ids), len(bag_ids)))  # SED/DD/Hc/VRC/I no usan dist_matrix
    X       = compute_X(train_scaled, bag_ids)

    internal_results = {}
    if stats["num_clusters"] == 0:
        for cvi, _ in INTERNAL_CVIS:
            internal_results[cvi.name] = None
    else:
        for cvi, needs_X in INTERNAL_CVIS:
            try:
                val = cvi.compute(dm, model.labels, bag_ids, X=X if needs_X else None)
                internal_results[cvi.name] = float(val)
            except Exception as exc:
                logger.warning(f"[{cvi.name}] Error: {exc}")
                internal_results[cvi.name] = None

    # 5. Métricas externas (sobre test)
    external_results = MILEvaluator.evaluate(
        test_scaled,
        test_labels,
        title=f"{name}",
        # silenciamos el reporte interno de MILEvaluator, lo imprimimos nosotros
    )

    # 6. Reporte unificado
    _print_report(
        name, scaler_cls.__name__, metric, eps, min_pts,
        stats, internal_results, external_results
    )

    return internal_results, external_results


def _print_report(
    name, scaler_name, metric, eps, min_pts,
    stats, internal: dict, external: dict
):
    W = 52

    print(f"\n{'═'*W}")
    print(f"  EVALUACIÓN COMPLETA — {name}")
    print(f"{'═'*W}")
    print(f"  Scaler   : {scaler_name}")
    print(f"  Métrica  : {metric}")
    print(f"  eps      : {eps}   min_pts: {min_pts}")
    print(f"  Clusters : {stats['num_clusters']}")
    print(f"  Ruido    : {stats['noise_percentage']:.1f}%")

    # ── CVIs internos ─────────────────────────────────────────────────────────
    groups = [
        ("Solo Compactibilidad   (↓ mejor)", ["SED", "DD", "Hc"]),
        ("Compact. + Separación  (↑ mejor)", ["VRC", "I"]),
    ]

    for group_label, cvi_names in groups:
        print(f"\n  ── {group_label}")
        for cvi_name in cvi_names:
            val = internal.get(cvi_name)
            if val is None:
                val_str = "        N/A"
            elif abs(val) >= 1e15:
                val_str = "          ∞"
            else:
                val_str = f"{val:>11.4f}"
            print(f"    {cvi_name:<6} {val_str}")

    # ── Métricas externas ─────────────────────────────────────────────────────
    print(f"\n  ── Métricas Externas    (↑ mejor)")
    ext_keys = ["Precision", "Recall", "F1-Score", "Specificity"]
    for key in ext_keys:
        val = external.get(key)
        val_str = f"{val:>11.4f}" if val is not None else "        N/A"
        print(f"    {key:<12} {val_str}")

    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    run_evaluation(DATASET)