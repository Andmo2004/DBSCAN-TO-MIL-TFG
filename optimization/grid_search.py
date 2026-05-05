"""
optimization/grid_search.py

Búsqueda en grid de (eps, min_pts) para MIDBSCAN.
Selecciona la combinación con mejor F1 en el propio conjunto de entrenamiento
usando validación interna (etiquetas reales de train).
"""

import logging
import numpy as np
from itertools import product
from typing import Callable, List, Tuple, Dict, Any, Optional

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.bag import Bag
from data.midata import MIData
from distances.distance_matrix import compute_distance_matrix
from optimization.knn_dist_eps import optimize_eps, _adaptive_eps_cap_percentile
from evaluation.scoring import detect_imbalance_ratio, score_labels

logger = logging.getLogger(__name__)


def grid_search_dbscan(
    dataset: MIData,
    distance_func: Callable[[Bag, Bag], float],
    metric_name: str = "hausdorff",
    min_pts_values: Optional[List[int]] = None,
    n_eps_values: int = 12,
    save_plots: bool = False,
) -> Tuple[float, int]:
    """
    Busca la combinación óptima de (eps, min_pts) para MIDBSCAN evaluando
    directamente sobre el conjunto de entrenamiento con etiquetas ground-truth.

    Workflow:
      1. Calcula la matriz de distancias una sola vez.
      2. Para cada min_pts candidato, estima un eps base con knn_dist_eps.
      3. Alrededor de ese eps base, prueba n_eps_values variaciones.
      4. Selecciona la combinación con mayor score interno.
      5. Selecciona la combinación con mayor score:
        - F1 binario si dataset equilibrado
        - F1 macro   si dataset desbalanceado (ratio < 0.3)


    :param dataset:         MIData ya escalado (train).
    :param distance_func:   Función (Bag, Bag) → float.
    :param metric_name:     Nombre de la métrica (para logging).
    :param min_pts_values:  Lista de valores min_pts a explorar.
                            Si None, usa heurística: [2, 3, ln(n), 2*ln(n)].
    :param n_eps_values:    Número de valores eps a explorar alrededor del óptimo.
    :param save_plots:      Pasar a optimize_eps.
    :returns:               Tupla (best_eps, best_min_pts).
    """
    from models.midbscan import MIDBSCAN

    n_bags = dataset.get_num_bags()

    # ── Detección de desbalanceo (una sola vez) ───────────────────────────────
    imbalance_ratio = detect_imbalance_ratio(dataset)
    logger.info(
        f"Ratio de desbalanceo: {imbalance_ratio:.3f} → "
        f"{'DESBALANCEADO (F1 macro)' if imbalance_ratio < 0.3 else 'Equilibrado (F1 binario)'}"
    )

    # ── Valores de min_pts a explorar ─────────────────────────────────────────
    if min_pts_values is None:
        ln_n = max(2, int(np.log(n_bags)))
        candidates = sorted(set([2, 3, ln_n, min(ln_n * 2, n_bags // 4)]))
        min_pts_values = [v for v in candidates if 2 <= v <= n_bags // 2]
    logger.info(f"Grid search: min_pts candidatos = {min_pts_values}")

    # ── Matriz de distancias (se calcula UNA sola vez) ────────────────────────
    logger.info("Calculando matriz de distancias para grid search...")
    bags = dataset.bags
    dist_matrix = compute_distance_matrix(bags, distance_func, metric_name)

    # Límites de eps desde las distancias del dataset
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    upper = upper[upper > 0]
    cap_pct = _adaptive_eps_cap_percentile(n_bags)
    eps_global_cap: float = float(np.percentile(upper, cap_pct))
    eps_global_min: float = float(np.percentile(upper, 3))
    best_eps: float       = float(np.percentile(upper, 15))

    best_score    = -1.0
    best_min_pts  = min_pts_values[0]

    results_log: List[Dict[str, Any]] = []

    for min_pts in min_pts_values:
        # Eps base para este min_pts usando knn
        eps_base_raw = optimize_eps(
            dataset=dataset,
            distance_func=distance_func,
            min_pts=min_pts,
            try_range=False,      # Solo k = min_pts-1 para rapidez
            save_plots=save_plots,
        )
        # Convertir explícitamente a float puro de Python
        eps_base = float(eps_base_raw) if hasattr(eps_base_raw, 'item') else float(eps_base_raw)
        logger.info(f"min_pts={min_pts} → eps_base={eps_base:.6f}")

        # Rango de eps alrededor del base: [0.5*base, 1.5*base] con n_eps_values pasos
        eps_lo = max(eps_global_min, eps_base * 0.5)
        eps_hi = min(eps_global_cap, eps_base * 1.5)
        if eps_lo >= eps_hi:
            eps_candidates = [eps_base]
        else:
            eps_candidates = [float(e.item()) if hasattr(e, 'item') else float(e) 
                            for e in np.linspace(eps_lo, eps_hi, n_eps_values)]

        for eps in eps_candidates:
            try:
                model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric_name)

                # inyección directa
                model._distance_matrix = dist_matrix

                # Entrenamos el modelo normalmente
                model.fit(dataset)

                score = score_labels(dataset, model.labels, imbalance_ratio=imbalance_ratio)

                n_clusters = model.cluster_count
                noise_pct  = model.get_statistics().get("noise_percentage", 0)

                logger.info(
                    f"  eps={eps:.4f} min_pts={min_pts} → "
                    f"clusters={n_clusters} noise={noise_pct:.1f}% score={score:.4f}"
                )

                results_log.append({
                    "eps": eps, "min_pts": min_pts,
                    "score": score, "clusters": n_clusters, "noise_pct": noise_pct,
                })

                if score > best_score:
                    best_score   = score
                    best_eps     = eps
                    best_min_pts = min_pts

            except Exception as e:
                logger.warning(f"  eps={eps:.4f} min_pts={min_pts} → Error: {e}")

    logger.info(
        f"Grid search completado → best_eps={best_eps:.6f}, "
        f"best_min_pts={best_min_pts}, best_score={best_score:.4f}"
    )
    return best_eps, best_min_pts