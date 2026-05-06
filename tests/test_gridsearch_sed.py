"""
tests/test_cvi_grid.py

Grid search de CVIs internos sobre todos los datasets.

Ejes del grid por cada dataset:
  - Scaler  : MinMaxScaler, StandardScaler
  - Métrica : hausdorff, cauchy_schwarz
  - Eps     : N_EPS valores en [EPS_LO·best_eps, EPS_HI·best_eps]

Por cada combinación se entrena MIDBSCAN con best_min_pts y se calcula
SED directamente desde internal_cvi (sin pasar por MILEvaluator).

Casos especiales registrados en el CSV:
  - 0 clusters (todo ruido) → SED = None, status = "no_clusters"
  - excepción inesperada    → SED = None, status = "error"

Salida: CSV en results/cvi_grid_<timestamp>.csv
"""

import os
import sys
import csv
import logging
from datetime import datetime
from itertools import product
from typing import Any, Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np

from data.midata import MIData
from models.midbscan import MIDBSCAN
from preprocessing.scaler import MinMaxScaler, StandardScaler
from distances.hausdorff import hausdorff_distance
from distances.cauchy_schwarz import cauchy_schwarz_distance
from evaluation.cvi import SEDIndex

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("cvi_grid")

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR

# ── Parámetros del grid ───────────────────────────────────────────────────────

SCALERS = [MinMaxScaler, StandardScaler]
METRICS = ["hausdorff", "cauchy_schwarz"]
N_EPS   = 5     # valores de eps por combinación
EPS_LO  = 0.5  # factor inferior sobre best_eps
EPS_HI  = 1.5  # factor superior sobre best_eps

# ── Helpers ───────────────────────────────────────────────────────────────────

def eps_range(best_eps: float) -> List[float]:
    """N_EPS valores en [EPS_LO·best_eps, EPS_HI·best_eps]."""
    return list(np.linspace(best_eps * EPS_LO, best_eps * EPS_HI, N_EPS).round(6))


def compute_X(dataset: MIData, bag_ids: List[str]) -> np.ndarray:
    """Centroides de bolsas (N × n_features) alineados con bag_ids."""
    bag_index = {bag.bag_id: bag for bag in dataset.bags}
    rows = []
    for bid in bag_ids:
        bag = bag_index.get(bid)
        rows.append(
            np.mean(bag.as_matrix(), axis=0) if bag and len(bag) > 0
            else np.zeros(1)
        )
    return np.array(rows)


# ── Lógica principal ──────────────────────────────────────────────────────────

def run_grid(config: dict) -> List[Dict[str, Any]]:
    """Grid completo para un dataset. Devuelve lista de filas."""
    name      = config["dataset_name"]
    arff_name = config["arff_name"]
    min_pts   = config["best_min_pts"]
    best_eps  = config["best_eps"]

    print(f"  {name} ...", end="", flush=True)

    path     = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
    dataset  = MIData.from_arff(path)
    train, _ = dataset.split_data(percentage_train=70, seed=42)

    sed_index = SEDIndex()
    rows: List[Dict[str, Any]] = []

    for scaler_cls, metric in product(SCALERS, METRICS):
        scaler       = scaler_cls()
        train_scaled = scaler.fit_transform(train)

        bag_ids = [bag.bag_id for bag in train_scaled.bags]
        X       = compute_X(train_scaled, bag_ids)
        dm      = np.zeros((len(bag_ids), len(bag_ids)))  # SED no usa dist_matrix

        for eps in eps_range(best_eps):
            row: Dict[str, Any] = {
                "dataset":   name,
                "scaler":    scaler_cls.__name__,
                "metric":    metric,
                "eps":       round(eps, 8),
                "min_pts":   min_pts,
                "is_best":   (
                    scaler_cls is config["best_scaler"]
                    and metric == config["best_distance"]
                    and abs(eps - best_eps) < 1e-6
                ),
                "clusters":  0,
                "noise_pct": 0.0,
                "SED":       None,
                "status":    "ok",
                "error":     "",
            }

            try:
                model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
                model.fit(train_scaled)

                stats            = model.get_statistics()
                row["clusters"]  = stats["num_clusters"]
                row["noise_pct"] = round(stats["noise_percentage"], 1)

                # Sin clusters reales no tiene sentido calcular SED
                if stats["num_clusters"] == 0:
                    row["status"] = "no_clusters"
                else:
                    row["SED"] = round(
                        float(sed_index.compute(dm, model.labels, bag_ids, X=X)), 6
                    )

            except Exception as exc:
                row["status"] = "error"
                row["error"]  = str(exc)
                logger.warning(f"[{name}] {scaler_cls.__name__}/{metric}/eps={eps:.6f} → {exc}")

            rows.append(row)

    ok          = sum(1 for r in rows if r["status"] == "ok")
    no_clusters = sum(1 for r in rows if r["status"] == "no_clusters")
    errors      = sum(1 for r in rows if r["status"] == "error")
    print(f" ok={ok}  no_clusters={no_clusters}  errors={errors}")

    return rows


def save_csv(all_rows: List[Dict[str, Any]]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%d%m%Y%H%M")
    out_path = os.path.join(RESULTS_DIR, f"cvi_grid_{ts}.csv")

    fieldnames = [
        "dataset", "scaler", "metric", "eps", "min_pts",
        "is_best", "clusters", "noise_pct", "SED",
        "status", "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total = len(DATASETS_CONFIG) * len(SCALERS) * len(METRICS) * N_EPS
    print(f"\nGrid search CVIs internos — SED")
    print(f"Datasets : {len(DATASETS_CONFIG)}")
    print(f"Scalers  : {[s.__name__ for s in SCALERS]}")
    print(f"Métricas : {METRICS}")
    print(f"Eps      : {N_EPS} valores en [×{EPS_LO}, ×{EPS_HI}] del best_eps")
    print(f"Total    : {total} runs\n")

    all_rows: List[Dict[str, Any]] = []
    for config in DATASETS_CONFIG:
        all_rows.extend(run_grid(config))

    csv_path = save_csv(all_rows)

    # Resumen final
    total_ok  = sum(1 for r in all_rows if r["status"] == "ok")
    total_nc  = sum(1 for r in all_rows if r["status"] == "no_clusters")
    total_err = sum(1 for r in all_rows if r["status"] == "error")
    print(f"\nTotal  ok={total_ok}  no_clusters={total_nc}  errors={total_err}")
    print(f"CSV guardado → {csv_path}")


if __name__ == "__main__":
    main()