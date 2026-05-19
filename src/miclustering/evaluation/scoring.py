import logging
import numpy as np
from typing import Dict

from miclustering.data.midata import MIData
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def detect_imbalance_ratio(dataset: MIData) -> float:
    """
    Calcula el ratio de desbalanceo: n_minority / n_majority.

    Retorna un valor en (0, 1]:
        1.0 → clases perfectamente equilibradas
        0.1 → clase minoritaria 10x más pequeña que la mayoritaria
    """
    labels = []
    for bag in dataset.bags:
        lv = int(float(bag.label)) if isinstance(bag.label, (str, float)) else int(bag.label)
        labels.append(lv)

    arr = np.array(labels)
    counts = np.bincount(arr.astype(int), minlength=2)
    if counts.min() == 0:
       return 0.0

    return float(counts.min()) / float(counts.max())


def score_labels(
    dataset: MIData,
    predicted_labels: Dict[str, int],
    imbalance_ratio: float = 1.0,
) -> float:
    """
    Calcula un score combinado para una configuración DBSCAN dada sobre
    el conjunto de entrenamiento.

    Criterios (en orden de importancia):
      1. Penalización por casos degenerados:
           - 0 clusters reales (todo ruido) → score = 0
           - 1 cluster → score muy bajo (no hay separación)
      2. Maximizar F1 (usando Hungarian mapping contra las etiquetas reales).
      3. Desempate: penalizar exceso de ruido y exceso de fragmentación.

    :param dataset:          MIData con etiquetas ground-truth en bag.label.
    :param predicted_labels: Dict {bag_id: cluster_id} producido por MIDBSCAN.
    :param imbalance_ratio:  Ratio de desbalanceo para decidir métrica (macro/binary).
    :returns:                Score en [0, 1] (mayor es mejor).
    """
    y_true, y_pred = [], []
    for bag in dataset.bags:
        if bag.bag_id in predicted_labels:
            label_val = int(float(bag.label)) \
                if isinstance(bag.label, (str, float)) else int(bag.label)
            y_true.append(label_val)
            y_pred.append(predicted_labels[bag.bag_id])

    if not y_true:
        return 0.0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    real_clusters = np.unique(y_pred[y_pred >= 0])
    n_clusters = len(real_clusters)

    # ── Casos degenerados ────────────────────────────────────────────────────
    if n_clusters == 0:
        return 0.0
    if n_clusters == 1:
        # Un único cluster puede ser correcto en casos raros, pero penalizamos
        return 0.05

    # ── Hungarian mapping - F1 ───────────────────────────────────────────────
    classes = np.array([0, 1])
    cost = np.zeros((n_clusters, 2), dtype=int)
    for i, c in enumerate(real_clusters):
        mask = y_pred == c
        for j, cls in enumerate(classes):
            cost[i, j] = np.sum(y_true[mask] != cls)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    assigned = set()
    for r, c in zip(row_ind, col_ind):
        mapping[int(real_clusters[r])] = int(classes[c])
        assigned.add(int(real_clusters[r]))

    # Fallback majority voting para clusters no asignados
    for c in real_clusters:
        cid = int(c)
        if cid not in assigned:
            mask = y_pred == c
            counts = np.bincount(y_true[mask].astype(int), minlength=2)
            mapping[cid] = int(np.argmax(counts))

    # Ruido → clase mayoritaria global
    noise_mask = y_pred < 0
    if noise_mask.any():
        counts = np.bincount(y_true[noise_mask].astype(int), minlength=2)
        mapping[-1] = int(np.argmax(counts))

    y_mapped = np.array([mapping.get(p, 0) for p in y_pred])
    
    # ── Métrica de score adaptativa al desbalanceo ───────────────────────────
    if imbalance_ratio < 0.3:
        base_f1 = float(f1_score(y_true, y_mapped, average='macro', zero_division=0))
        logger.debug(
            f"Dataset desbalanceado (ratio={imbalance_ratio:.2f}) "
            f"- F1 macro={base_f1:.4f}"
        )
    else:
        base_f1 = float(f1_score(y_true, y_mapped, average='binary', zero_division=0))

    # ── Penalización por ruido excesivo y fragmentación ───────────────────────
    noise_pct = float(np.sum(y_pred < 0)) / len(y_pred)
    # Penalizar si ruido > 30%
    noise_penalty = max(0.0, noise_pct - 0.30) * 0.5

    # Penalizar clusters > 10 (fragmentación)
    frag_penalty = max(0, n_clusters - 10) * 0.02

    score = float(base_f1) - noise_penalty - frag_penalty
    return float(max(0.0, score))
