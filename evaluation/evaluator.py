from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from data.midata import MIData
import logging

logger = logging.getLogger(__name__)

class MILEvaluator:
    """
    Clase para evaluar resultados de algoritmos MIL.
    Transforma resultados de clustering en métricas de clasificación binaria.
    """

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
        y_pred_mapped, mapping = MILEvaluator.map_clusters_to_labels(y_true, y_pred_raw)

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
        
        print("\nMapeo de Clusters (Interpretación):")
        print("-" * 30)
        for cluster_id, class_label in mapping.items():
            label_name = "Positive (1)" if class_label == 1 else "Negative (0)"
            cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Ruido (-1)"
            print(f"{cluster_name:<12} -> {label_name}")
            
        return results