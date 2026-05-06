"""
Phase 1 - Caracterización del Problema (EDA)
Genera estadísticas sobre las distribuciones intra/inter clase, y visualizaciones
de las distancias (heatmaps) y tamaños de las bolsas (boxplots).
"""

import os
import sys
import csv
import logging
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.midata import MIData
from preprocessing.scaler import MinMaxScaler
from distances.distance_matrix import compute_distance_matrix
from distances.hausdorff import hausdorff_distance
from distances.probability_distribution import cauchy_schwarz_distance

from visualization.heatmap import plot_distance_heatmap
from visualization.boxplots import plot_instances_per_bag_boxplot

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Usaremos la misma configuración de datasets que en test_full_eval y best_params
DATASETS_CONFIG = [
    {"dataset_name": "musk1",              "arff_name": "musk1"},
    {"dataset_name": "musk2",              "arff_name": "musk2"},
    {"dataset_name": "ImageElephant",      "arff_name": "ImageElephant"},
    {"dataset_name": "BirdsChestnut",      "arff_name": "BirdsChestnut-backedChickadee"},
    {"dataset_name": "BirdsHammonds",      "arff_name": "BirdsHammondsFlycatcher"},
    {"dataset_name": "Harddrive1",         "arff_name": "Harddrive1"},
    {"dataset_name": "mutagenesis_atoms",  "arff_name": "mutagenesis3_atoms"},
    {"dataset_name": "mutagenesis_chains", "arff_name": "mutagenesis3_chains"},
    {"dataset_name": "Newsgroups1",        "arff_name": "Newsgroups1"},
    {"dataset_name": "Thioredoxin",        "arff_name": "Thioredoxin"},
]

def analyze_distances(dist_matrix: np.ndarray, sorted_bags: list) -> tuple:
    """
    Dada una matriz de distancias y las bolsas ordenadas correspondientes, 
    extrae métricas de separabilidad intra e inter clase.
    """
    n = len(sorted_bags)
    labels = np.array([int(float(b.label)) for b in sorted_bags])
    
    intra_distances = []
    inter_distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_distances.append(dist_matrix[i, j])
            else:
                inter_distances.append(dist_matrix[i, j])
                
    mean_intra = np.mean(intra_distances) if intra_distances else 0.0
    mean_inter = np.mean(inter_distances) if inter_distances else 0.0
    
    std_intra = np.std(intra_distances) if intra_distances else 0.0
    cv_intra = std_intra / mean_intra if mean_intra > 0 else 0.0
    
    sep_ratio = mean_inter / mean_intra if mean_intra > 0 else 0.0
    
    return mean_intra, mean_inter, sep_ratio, cv_intra

def run_phase_1():
    os.makedirs("results/eda", exist_ok=True)
    os.makedirs("results/output_heatmaps", exist_ok=True)
    
    summary_data = []
    
    print(f"\n{'='*70}")
    print(f"  INICIANDO PHASE 1: EDA Y CARACTERIZACIÓN DEL PROBLEMA")
    print(f"{'='*70}")
    
    scaler = MinMaxScaler()
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        print(f"\n► Analizando Dataset: {dataset_name}...")
        
        path = os.path.join("datasets", f"{arff_name}.arff")
        if not os.path.exists(path):
            print(f"  [!] No se encontró el archivo: {path}. Omitiendo.")
            continue
            
        # Cargar y dividir
        dataset_full = MIData.from_arff(path)
        train_data, _ = dataset_full.split_data(percentage_train=70, seed=42)
        
        # 1. Escalar (MinMaxScaler como base estandarizada para el EDA)
        train_scaled = scaler.fit_transform(train_data)
        
        # 2. Boxplots de instancias por bolsa (sobre dataset completo)
        plot_instances_per_bag_boxplot(dataset_full, dataset_name, output_dir="results/eda")
        
        # 3. Extraer Estadísticas Básicas de Instancias
        inst_counts = [len(b.instances) for b in dataset_full.bags]
        labels = [int(float(b.label)) for b in dataset_full.bags]
        
        unique_labels = list(set(labels))
        # Asumiremos la etiqueta de mayor valor (usualmente 1) como la positiva
        pos_label = max(unique_labels)
        n_pos = sum(1 for l in labels if l == pos_label)
        n_neg = len(labels) - n_pos
        
        # 4. Preparar bolsas para matrices ordenándolas por etiqueta
        # (Así el heatmap revelará bloques contiguos intra-clase)
        sorted_bags = sorted(train_scaled.bags, key=lambda b: int(float(b.label)))
        bag_ids_sorted = [b.bag_id for b in sorted_bags]
        
        # 5. Cálculo y Análisis con Hausdorff
        print("  - Calculando matriz Hausdorff...")
        dist_hau = compute_distance_matrix(sorted_bags, hausdorff_distance, "hausdorff")
        _, _, sep_hau, _ = analyze_distances(dist_hau, sorted_bags)
        
        # Heatmap Hausdorff
        plot_distance_heatmap(
            distance_matrix=dist_hau, 
            bag_ids=bag_ids_sorted,
            title=f"Heatmap {dataset_name} - Hausdorff",
            metric="hausdorff",
            output_dir="results/output_heatmaps",
            filename=f"heatmap_{dataset_name}_hausdorff"
        )
        
        # 6. Cálculo y Análisis con Cauchy-Schwarz
        print("  - Calculando matriz Cauchy-Schwarz...")
        dist_cs = compute_distance_matrix(sorted_bags, cauchy_schwarz_distance, "cauchy_schwarz")
        _, _, sep_cs, _ = analyze_distances(dist_cs, sorted_bags)
        
        # Heatmap Cauchy-Schwarz
        plot_distance_heatmap(
            distance_matrix=dist_cs, 
            bag_ids=bag_ids_sorted,
            title=f"Heatmap {dataset_name} - Cauchy-Schwarz",
            metric="cauchy_schwarz",
            output_dir="results/output_heatmaps",
            filename=f"heatmap_{dataset_name}_cauchy_schwarz"
        )
        
        # 7. Imprimir resultados parciales
        print(f"  > Bags: {len(dataset_full.bags)} (Pos: {n_pos}, Neg: {n_neg})")
        print(f"  > Instancias (Min/Avg/Max): {np.min(inst_counts)} / {np.mean(inst_counts):.1f} / {np.max(inst_counts)}")
        print(f"  > Sep Ratio (Hausdorff): {sep_hau:.4f} | (Cauchy-Schwarz): {sep_cs:.4f}")
        
        if sep_hau < 1.2 and sep_cs < 1.2:
            print("  [!] ADVERTENCIA: Solapamiento severo detectado (Sep Ratio < 1.2). Dataset de alta dificultad.")
        
        summary_data.append({
            "Dataset": dataset_name,
            "n_bags": len(dataset_full.bags),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "inst_min": np.min(inst_counts),
            "inst_avg": round(np.mean(inst_counts), 2),
            "inst_max": np.max(inst_counts),
            "inst_std": round(np.std(inst_counts), 2),
            "sep_ratio_hau": round(sep_hau, 4),
            "sep_ratio_cs": round(sep_cs, 4)
        })
        
    # 8. Guardar todo en un CSV de resumen
    csv_file = "results/eda/eda_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "n_bags", "n_pos", "n_neg", "inst_min", 
            "inst_avg", "inst_max", "inst_std", "sep_ratio_hau", "sep_ratio_cs"
        ])
        writer.writeheader()
        writer.writerows(summary_data)
        
    print(f"\n{'='*70}")
    print(f"  PHASE 1 COMPLETADA.")
    print(f"  -> Gráficos generados en: results/output_heatmaps/ y results/eda/")
    print(f"  -> Resumen de métricas guardado en: {csv_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_phase_1()
