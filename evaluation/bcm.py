import logging
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple

from data.midata import MIData

logger = logging.getLogger(__name__)

class MILEvaluator:
    """
    Clase para evaluar resultados de algoritmos MIL.
    Transforma resultados de clustering en métricas de clasificación binaria (BCM).
    """
    @staticmethod
    def hungarian_map_clusters_to_labels(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Refs:
            - Kuhn, H. W. (1955). The Hungarian method for the assignment problem. 
              Naval Research Logistics Quarterly, 2(1:2), 83:97.
              https://doi.org/10.1002/nav.3800020109

            - Munkres, J. (1957). Algorithms for the assignment and transportation problems. 
              Journal of the Society for Industrial and Applied Mathematics, 5(1), 32:38.
              (scipy.optimize.linear_sum_assignment)

                    "
                    El algoritmo también se conoce como Kuhn Munkres o Munkres assignment algorithm. 
                        La complejidad del algoritmo original era O(n⁴), pero Edmonds, Karp y Tomizawa 
                        lo modificaron independientemente para alcanzar O(n³).
                    "

        Asigna etiquetas a clusters usando el algoritmo Húngaro (asignación óptima global).
        
        El majority voting local falla cuando todos los clusters tienen mayoría de la misma
        clase: todos se mapean a 0 o todos a 1. El algoritmo Húngaro maximiza el acuerdo
        global entre clusters y clases.

        Para problemas binarios (clases 0/1) con K clusters:
          - Construimos una matriz de coste (K x 2) donde coste[k, c] = instancias
            del cluster k que NO son de la clase c.
          - linear_sum_assignment minimiza el coste total → maximiza el acuerdo.
          - Si hay más clusters que clases, los clusters sin asignación única se resuelven
            por majority voting local como fallback.

        :returns: (Tuple[np.ndarray, Dict[int, int]])
            y_pred_mapped: Array con las predicciones traducidas a 0/1.
            mapping: Diccionario {cluster_id: clase_asignada}.
        """
        clusters = np.unique(y_pred)
        # Separamos ruido (-1) para tratarlo aparte
        noise_clusters = clusters[clusters < 0]
        real_clusters  = clusters[clusters >= 0]

        # ── Paso 1: Construir matriz de coste para clusters reales ────────────
        classes = np.array([0, 1])
        n_clusters = len(real_clusters)
        n_classes  = len(classes)

        # cost[i, j] = nº de instancias del cluster i que NO son de la clase j
        cost_matrix = np.zeros((n_clusters, n_classes), dtype=int)
        for i, cluster in enumerate(real_clusters):
            mask = y_pred == cluster
            true_labels_in_cluster = y_true[mask]
            for j, cls in enumerate(classes):
                # Coste = cuantas instancias NO coinciden
                cost_matrix[i, j] = np.sum(true_labels_in_cluster != cls)

        # ── Paso 2: Algoritmo Húngaro ─────────────────────────────────────────
        # linear_sum_assignment trabaja sobre matrices cuadradas o rectangulares.
        # Si hay más clusters que clases (K > 2), el solver solo asignará min(K,2)
        # clusters de forma óptima; el resto se asigna por majority voting local.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping: Dict[int, int] = {}
        assigned_clusters = set()

        for r, c in zip(row_ind, col_ind):
            cluster_id  = int(real_clusters[r])
            class_label = int(classes[c])
            mapping[cluster_id] = class_label
            assigned_clusters.add(cluster_id)

        # ── Paso 3: Fallback majority voting para clusters no asignados ───────
        for cluster in real_clusters:
            cid = int(cluster)
            if cid not in assigned_clusters:
                mask = y_pred == cluster
                true_labels_in_cluster = y_true[mask]
                if len(true_labels_in_cluster) > 0:
                    counts = np.bincount(true_labels_in_cluster.astype(int), minlength=2)
                    mapping[cid] = int(np.argmax(counts))
                else:
                    mapping[cid] = 0

        # ── Paso 4: Ruido → clase mayoritaria global (no 0 por defecto) ───────
        for cluster in noise_clusters:
            cid = int(cluster)
            mask = y_pred == cluster
            true_labels_in_cluster = y_true[mask]
            if len(true_labels_in_cluster) > 0:
                counts = np.bincount(true_labels_in_cluster.astype(int), minlength=2)
                mapping[cid] = int(np.argmax(counts))
            else:
                mapping[cid] = 0

        # ── Paso 5: Construir array de predicciones mapeadas ──────────────────
        y_pred_mapped = np.zeros_like(y_pred)
        for cluster_id, class_label in mapping.items():
            y_pred_mapped[y_pred == cluster_id] = class_label

        return y_pred_mapped, mapping

    @staticmethod
    def map_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Asigna a cada clúster la etiqueta real más frecuente (Majority Voting).
        Esto permite evaluar un algoritmo no supervisado como si fuera supervisado.
        
        :returns: (Tuple[np.ndarray, Dict[int, int]])
            y_pred_mapped: Array con las predicciones traducidas a 0/1.
            mapping: Diccionario {cluster_id: clase_asignada}.
        """
        y_pred_mapped = np.zeros_like(y_pred)
        mapping = {}
        
        # Obtenemos los clústeres únicos (incluyendo ruido -1 si existe)
        clusters = np.unique(y_pred)
        
        for cluster in clusters:
            # Buscamos índices donde aparece este cluster
            indices = np.where(y_pred == cluster)[0]
            
            # Obtenemos las etiquetas reales de esos puntos
            true_labels_in_cluster = y_true[indices]
            
            if len(true_labels_in_cluster) > 0:
                # Encontramos la moda (la etiqueta más común en este cluster)
                # bincount cuenta ocurrencias de enteros no negativos.
                # Como las etiquetas son 0 o 1, funciona perfecto.
                counts = np.bincount(true_labels_in_cluster.astype(int))
                most_frequent_label = np.argmax(counts)
                
                mapping[cluster] = most_frequent_label
                y_pred_mapped[indices] = most_frequent_label
            else:
                # Caso raro: cluster vacío (no debería pasar)
                mapping[cluster] = 0 
                
        return y_pred_mapped, mapping

    @staticmethod
    def evaluate(dataset: MIData, model_labels: Dict[str, int], title: str = "Evaluación") -> Dict[str, float]:
        """
        Calcula Precision, Recall, F1 y Specificity.
        """
        # Alineamos etiquetas (Ground Truth vs Predicciones)
        y_true = []
        y_pred_raw = [] # Etiquetas del cluster (0, 1, 2, -1...)
        
        # Solo evaluamos bolsas que existen en el resultado
        for bag in dataset.bags:
            if bag.bag_id in model_labels:
                # Convertimos la etiqueta de la bolsa a int (por si viene como string '1.0')
                label_val = int(float(bag.label)) if isinstance(bag.label, (str, float)) else int(bag.label)
                y_true.append(label_val)
                y_pred_raw.append(model_labels[bag.bag_id])
        
        if not y_true:
            logger.warning("No hay etiquetas para evaluar.")
            return {}

        y_true = np.array(y_true)
        y_pred_raw = np.array(y_pred_raw)

        # Mapeo Mágico: Convertir Clusters -> Clases (0/1)
        # y_pred_mapped, mapping = MILEvaluator.map_clusters_to_labels(y_true, y_pred_raw)
        y_pred_mapped, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true, y_pred_raw)

        # Cálculo de Métricas
        # Positive label = 1
        precision = metrics.precision_score(y_true, y_pred_mapped, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred_mapped, zero_division=0) # Sensitivity
        f1 = metrics.f1_score(y_true, y_pred_mapped, zero_division=0)
        
        # Specificity no tiene función directa en sklearn, se calcula via matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_mapped, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Specificity": specificity
        }

        print(f"\n{'='*60}")
        print(f"REPORTE DE CLASIFICACIÓN: {title}")
        print(f"{'='*60}")
        
        print("\nMétricas:")
        print(f"{'Métrica':<15} | {'Valor':<10}")
        print("-" * 30)
        for k, v in results.items():
            print(f"{k:<15} | {v:.4f}")
            
        print("\nMatriz de Confusión (Mapeada):")
        print(f"TN: {tn:<4} FP: {fp:<4}")
        print(f"FN: {fn:<4} TP: {tp:<4}")

        """ 
        print("\nMapeo de Clusters (Interpretación):")
        print("-" * 30)
        for cluster_id, class_label in mapping.items():
            label_name = "Positive (1)" if class_label == 1 else "Negative (0)"
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Ruido (-1)"
            print(f"{cluster_name:<12} -> {label_name}")
        """

        print("\nMapeo de Clusters - Clases (Húngaro + Fallback):")
        print("-" * 35)
        for cluster_id, class_label in sorted(mapping.items()):
            label_name   = "Positive (1)" if class_label == 1 else "Negative (0)"
            cluster_name = f"Cluster {cluster_id}" if cluster_id >= 0 else f"Ruido ({cluster_id})"
            print(f"  {cluster_name:<14} - {label_name}")
        
        return results