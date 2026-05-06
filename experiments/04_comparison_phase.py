"""
Fase 4 — Clasificación Final y Comparativa (Baseline con MIKnn)
Propósito: Responder la pregunta central del TFG: ¿es MIDBSCAN competitivo frente a un clasificador supervisado de distancias en problemas MIL?
"""

import os
import sys
import csv
import logging
from datetime import datetime
from typing import Dict

# Configurar PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data.midata import MIData
from models.midbscan import MIDBSCAN
from models.miknn import MIKnn
from evaluation.bcm import MILEvaluator
from distances.matrix_cache import global_persistent_cache
from distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

DISTANCES = {
    "hausdorff": hausdorff_distance,
    "hausdorff_min": hausdorff_distance_min,
    "hausdorff_avg": hausdorff_distance_avg,
    "cauchy_schwarz": cauchy_schwarz_distance,
    "earth_movers": earth_movers_distance,
    "mahalanobis": mahalanobis_distance
}

from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Datasets representativos para matriz de confusión
REP_DATASETS = ["musk1", "BirdsHammonds", "Thioredoxin"]

def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Real 0", "Real 1"])
    plt.title(f"Confusion Matrix - {model_name}\nDataset: {dataset_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{model_name}_{dataset_name}.png"))
    plt.close()

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "F1-Macro": f1_macro,
        "Specificity": spec,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def main():
    print("="*80)
    print("INICIANDO FASE 4: CLASIFICACIÓN FINAL Y COMPARATIVA (BASELINE MIKnn)")
    print("="*80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_dir = os.path.join(RESULTS_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    all_results = []
    
    for config in DATASETS_CONFIG:
        name = config["dataset_name"]
        arff_name = config["arff_name"]
        scaler_cls = config["best_scaler"]
        metric = config["best_distance"]
        best_eps = config["best_eps"]
        min_pts = config["best_min_pts"]
        
        print(f"\nProcesando dataset: {name}")
        path = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        
        if not os.path.exists(path):
            print(f"  [!] Archivo no encontrado: {path}")
            continue
            
        dataset = MIData.from_arff(path)
        # Partición: 70% train / 30% test
        train_data, test_data = dataset.split_data(percentage_train=70, seed=42)
        
        scaler = scaler_cls()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        y_true_train = np.array([int(float(bag.label)) for bag in train_scaled.bags])
        y_true_test = np.array([int(float(bag.label)) for bag in test_scaled.bags])
        
        # ---------------------------------------------------------
        # MIDBSCAN
        # ---------------------------------------------------------
        dbscan = MIDBSCAN(epsilon=best_eps, min_pts=min_pts, metric=metric)
        scaler_name_str = "MinMaxScaler" if "MinMaxScaler" in str(scaler_cls) else "StandardScaler"
        dist_matrix = global_persistent_cache.get(
            dataset_name=name,
            split="train",
            scaler_name=scaler_name_str,
            metric_name=metric,
            bags=train_scaled.bags,
            metric_func=DISTANCES[metric]
        )
        dbscan._distance_matrix = dist_matrix
        dbscan.fit(train_scaled)
        
        # Predicción en test (cluster labels crudas, incluyendo -1)
        dbscan_pred_dict = dbscan.predict(test_scaled)
        y_pred_raw_test = np.array([dbscan_pred_dict[bag.bag_id] for bag in test_scaled.bags])
        
        # Mapeo Húngaro usando los clusters encontrados en train
        y_pred_train_raw = np.array([dbscan.labels.get(bag.bag_id, dbscan.NOISE_LABEL) for bag in train_scaled.bags])
        _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true_train, y_pred_train_raw)
        
        # Aplicamos el mapeo a test
        y_pred_dbscan_test = np.zeros_like(y_pred_raw_test)
        for i, cluster in enumerate(y_pred_raw_test):
            if cluster in mapping:
                y_pred_dbscan_test[i] = mapping[cluster]
            else:
                y_pred_dbscan_test[i] = 0 # Fallback 0 si es un clúster no visto (o ruido sin mapear)
                
        dbscan_eval = evaluate_predictions(y_true_test, y_pred_dbscan_test)
        
        stats = dbscan.get_statistics()
        clusters_count = stats.get("num_clusters", 0)
        noise_pct = stats.get("noise_percentage", 100.0)
        
        # ---------------------------------------------------------
        # MIKnn
        # ---------------------------------------------------------
        # Búsqueda del mejor k sobre una partición de validación extraída de train
        train_sub, val_sub = train_scaled.split_data(percentage_train=80, seed=42)
        y_true_val = np.array([int(float(bag.label)) for bag in val_sub.bags])
        
        best_k = 1
        best_f1_knn = -1
        
        for k in [1, 3, 5]:
            # Usar k min_bag para evitar error si k > N
            k_eff = min(k, train_sub.get_num_bags())
            if k_eff < 1: 
                k_eff = 1
            
            knn = MIKnn(k=k_eff, metric=metric)
            knn.fit(train_sub)
            val_preds = knn.predict(val_sub)
            y_pred_val = np.array([val_preds.get(bag.bag_id, 0) for bag in val_sub.bags])
            
            f1_val = metrics.f1_score(y_true_val, y_pred_val, zero_division=0)
            if f1_val > best_f1_knn:
                best_f1_knn = f1_val
                best_k = k_eff
                
        # Entrenar modelo MIKnn final con el mejor k y todo el train
        best_knn = MIKnn(k=best_k, metric=metric)
        best_knn.fit(train_scaled)
        knn_pred_dict = best_knn.predict(test_scaled)
        y_pred_knn_test = np.array([knn_pred_dict.get(bag.bag_id, 0) for bag in test_scaled.bags])
        
        knn_eval = evaluate_predictions(y_true_test, y_pred_knn_test)
        
        delta_f1 = dbscan_eval["F1-Score"] - knn_eval["F1-Score"]
        
        res = {
            "Dataset": name,
            "F1-DBSCAN": round(dbscan_eval["F1-Score"], 4),
            "F1-KNN (best k)": round(knn_eval["F1-Score"], 4),
            "Δ F1": round(delta_f1, 4),
            "Clusters": clusters_count,
            "Noise%": round(noise_pct, 1),
            "k*": best_k,
            "DBSCAN-Acc": round(dbscan_eval["Accuracy"], 4),
            "KNN-Acc": round(knn_eval["Accuracy"], 4),
            "DBSCAN-Prec": round(dbscan_eval["Precision"], 4),
            "KNN-Prec": round(knn_eval["Precision"], 4),
            "DBSCAN-Rec": round(dbscan_eval["Recall"], 4),
            "KNN-Rec": round(knn_eval["Recall"], 4),
            "DBSCAN-Spec": round(dbscan_eval["Specificity"], 4),
            "KNN-Spec": round(knn_eval["Specificity"], 4),
            "DBSCAN-F1-Macro": round(dbscan_eval["F1-Macro"], 4),
            "KNN-F1-Macro": round(knn_eval["F1-Macro"], 4)
        }
        all_results.append(res)
        
        print(f"  [+] DBSCAN F1: {dbscan_eval['F1-Score']:.4f} | KNN F1 (k={best_k}): {knn_eval['F1-Score']:.4f} | Δ F1: {delta_f1:+.4f}")
        
        # Generar matriz de confusión solo para los representativos
        if name in REP_DATASETS:
            plot_confusion_matrix(y_true_test, y_pred_dbscan_test, "MIDBSCAN", name, cm_dir)
            plot_confusion_matrix(y_true_test, y_pred_knn_test, "MIKnn", name, cm_dir)
            print(f"  [+] Matrices de confusión generadas en {cm_dir}")

    # Guardar CSV final
    ts = datetime.now().strftime("%d%m%Y%H%M")
    csv_path = os.path.join(RESULTS_DIR, f"full_eval_{ts}.csv")
    
    fieldnames = [
        "Dataset", "F1-DBSCAN", "F1-KNN (best k)", "Δ F1", "Clusters", "Noise%", "k*",
        "DBSCAN-Acc", "KNN-Acc", "DBSCAN-Prec", "KNN-Prec", "DBSCAN-Rec", "KNN-Rec",
        "DBSCAN-Spec", "KNN-Spec", "DBSCAN-F1-Macro", "KNN-F1-Macro"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k) for k in fieldnames})
            
    print(f"\n[+] Resultados completos guardados en: {csv_path}")
    print("\n" + "="*80)
    print("FASE 4 COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()
