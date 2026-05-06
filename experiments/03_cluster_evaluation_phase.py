"""
Phase 3: Evaluación de la Calidad del Clustering (CVIs Internos)
Propósito: Evaluar si los clústeres detectados por MIDBSCAN son geométricamente sólidos,
independientemente de las etiquetas reales.
"""

import os
import sys
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List

# Configurar PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
from scipy.stats import spearmanr

from data.midata import MIData
from models.midbscan import MIDBSCAN
from evaluation.cvi import InternalCVIEvaluator, SEDIndex, DDIndex, HcIndex, VRCIndex, IIndex
from config.settings import DATASETS_CONFIG, DATASETS_DIR, RESULTS_DIR

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# F1-scores hardcodeados de los mejores resultados
F1_SCORES = {
    "musk1": 0.7674,
    "musk2": 0.7191,
    "ImageElephant": 0.6667,
    "BirdsChestnut": 0.8086,
    "BirdsHammonds": 0.9913,
    "Harddrive1": 0.9734,
    "mutagenesis_atoms": 0.8586,
    "mutagenesis_chains": 0.8889,
    "Newsgroups1": 0.7919,
    "simple_dummy": 0.889,
    "Thioredoxin": 0.6223,
}

def evaluate_model(
    config_name: str,
    eps: float,
    min_pts: int,
    metric: str,
    train_scaled: MIData,
    evaluator: InternalCVIEvaluator
) -> Dict[str, Any]:
    
    bag_ids = [bag.bag_id for bag in train_scaled.bags]
    n = len(bag_ids)
    
    # MIDBSCAN
    model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
    try:
        model.fit(train_scaled)
        stats = model.get_statistics()
        num_clusters = stats["num_clusters"]
        noise_pct = stats["noise_percentage"]
    except Exception as e:
        logger.warning(f"Error entrenando modelo {config_name}: {e}")
        return {
            "Configuración": config_name,
            "Clusters": 0,
            "Noise%": 100.0,
            "SED": None, "DD": None, "Hc": None, "VRC": None, "I": None
        }

    # Dummy dist_matrix porque estos CVIs usan X (centroides)
    dist_matrix = np.zeros((n, n))

    results = evaluator.evaluate(
        dist_matrix=dist_matrix,
        labels=model.labels,
        bag_ids=bag_ids,
        dataset=train_scaled,
        verbose=False
    )

    scores = results["scores"]
    return {
        "Configuración": config_name,
        "Clusters": num_clusters,
        "Noise%": round(noise_pct, 1),
        "SED": scores.get("SED", {}).get("value"),
        "DD": scores.get("DD", {}).get("value"),
        "Hc": scores.get("Hc", {}).get("value"),
        "VRC": scores.get("VRC", {}).get("value"),
        "I": scores.get("I", {}).get("value"),
    }

def main():
    print("="*80)
    print("INICIANDO FASE 3: EVALUACIÓN DE CALIDAD DE CLUSTERING (CVIs INTERNOS)")
    print("="*80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    evaluator = InternalCVIEvaluator(cvis=[SEDIndex(), DDIndex(), HcIndex(), VRCIndex(), IIndex()])
    
    all_results: List[Dict[str, Any]] = []
    
    # Para la correlación de Spearman
    vrc_values_opt = []
    i_values_opt = []
    f1_values_opt = []

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
        train, _ = dataset.split_data(percentage_train=70, seed=42)
        
        scaler = scaler_cls()
        train_scaled = scaler.fit_transform(train)
        
        # Evaluar configuraciones
        configs_to_run = [
            ("Óptima", best_eps),
            ("eps x 2", best_eps * 2.0),
            ("eps x 0.5", best_eps * 0.5)
        ]
        
        for conf_name, eps_val in configs_to_run:
            res = evaluate_model(conf_name, eps_val, min_pts, metric, train_scaled, evaluator)
            res["Dataset"] = name
            all_results.append(res)
            
            # Guardar VRC, I y F1 solo de la óptima para correlación
            if conf_name == "Óptima":
                vrc = res["VRC"]
                i_val = res["I"]
                f1 = F1_SCORES.get(name, None)
                
                if vrc is not None and i_val is not None and f1 is not None:
                    # Filter out inf/nan
                    if not (np.isinf(vrc) or np.isnan(vrc) or np.isinf(i_val) or np.isnan(i_val)):
                        vrc_values_opt.append(vrc)
                        i_values_opt.append(i_val)
                        f1_values_opt.append(f1)
                        
        print("  [+] Evaluaciones completadas.")

    # Guardar CSV
    ts = datetime.now().strftime("%d%m%Y%H%M")
    csv_path = os.path.join(RESULTS_DIR, f"cvi_comparative_{ts}.csv")
    
    fieldnames = ["Dataset", "Configuración", "Clusters", "Noise%", "SED", "DD", "Hc", "VRC", "I"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            # Reordenar claves para CSV
            writer.writerow({k: row.get(k) for k in fieldnames})
            
    print(f"\n[+] Resultados guardados en: {csv_path}")

    # Análisis de Correlación
    print("\n" + "-"*40)
    print("ANÁLISIS DE CORRELACIÓN (Spearman)")
    print("-"*40)
    
    if len(vrc_values_opt) > 2:
        corr_vrc, p_vrc = spearmanr(vrc_values_opt, f1_values_opt)
        corr_i, p_i = spearmanr(i_values_opt, f1_values_opt)
        
        print(f"VRC vs F1-score : r = {corr_vrc:.4f} (p-valor = {p_vrc:.4f})")
        print(f"I vs F1-score   : r = {corr_i:.4f} (p-valor = {p_i:.4f})")
    else:
        print("[!] No hay suficientes datos válidos para calcular la correlación.")

    print("\n" + "="*80)
    print("FASE 3 COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()
