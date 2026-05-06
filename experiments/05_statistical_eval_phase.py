"""
Fase 5 — Validación Estadística
Propósito: Proporcionar rigor científico a las comparaciones del TFG mediante el test de Wilcoxon de rangos con signo.
"""

import os
import sys
import glob
import csv
from datetime import datetime
import numpy as np
from scipy.stats import wilcoxon, norm

# Configurar PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config.settings import RESULTS_DIR

SUBGROUP_A = ["musk1", "musk2", "mutagenesis3_atoms", "mutagenesis3_chains", "Harddrive1"]
SUBGROUP_B = ["BirdsChestnut", "BirdsHammonds", "Thioredoxin", "Newsgroups1", "ImageElephant"]

def get_latest_full_eval_csv():
    files = glob.glob(os.path.join(RESULTS_DIR, "full_eval_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def calculate_effect_size(p_value, n):
    # Usando la aproximación normal para el tamaño del efecto r = Z / sqrt(N)
    # Z se puede obtener a partir del p-valor bilateral
    z = norm.ppf(1 - p_value / 2)
    r = z / np.sqrt(n)
    return r

def interpret_effect_size(r):
    if np.isnan(r):
        return "N/A"
    if r < 0.3:
        return "pequeño"
    elif r < 0.5:
        return "moderado"
    else:
        return "grande"

def run_wilcoxon_test(data1, data2, metric_name, subgroup_name="Todos"):
    n = len(data1)
    if n == 0:
        return None
    
    differences = np.array(data1) - np.array(data2)
    if np.all(differences == 0):
        print(f"[{subgroup_name}] {metric_name}: Todas las diferencias son cero (p-valor no calculable).")
        return None

    try:
        stat, p_value = wilcoxon(data1, data2, alternative='two-sided')
        r = calculate_effect_size(p_value, n)
        significant = "Sí" if p_value < 0.05 else "No"
        interpretation = interpret_effect_size(r)
        
        return {
            "Subgroup": subgroup_name,
            "Metric": metric_name,
            "W": stat,
            "p-value": p_value,
            "Significant (a=0.05)": significant,
            "Effect size r": r,
            "Interpretation": interpretation
        }
    except ValueError as e:
        print(f"[{subgroup_name}] ValueError en test {metric_name}: {e}")
        return None
    except Exception as e:
        print(f"[{subgroup_name}] Error en test {metric_name}: {e}")
        return None

def main():
    print("="*80)
    print("INICIANDO FASE 5: VALIDACIÓN ESTADÍSTICA (TEST DE WILCOXON)")
    print("="*80)

    csv_path = get_latest_full_eval_csv()
    if not csv_path:
        print("[!] No se encontró ningún archivo full_eval_*.csv en la carpeta de resultados.")
        print("Por favor, ejecute la Fase 4 primero.")
        return
        
    print(f"[*] Cargando resultados de: {os.path.basename(csv_path)}")
    
    results_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_data.append(row)
            
    f1_dbscan, f1_knn = [], []
    acc_dbscan, acc_knn = [], []
    rec_dbscan, rec_knn = [], []
    
    sgA_f1_dbscan, sgA_f1_knn = [], []
    sgB_f1_dbscan, sgB_f1_knn = [], []

    for row in results_data:
        dataset = row["Dataset"]
        try:
            f1_d = float(row["F1-DBSCAN"])
            f1_k = float(row["F1-KNN (best k)"])
            acc_d = float(row["DBSCAN-Acc"])
            acc_k = float(row["KNN-Acc"])
            rec_d = float(row["DBSCAN-Rec"])
            rec_k = float(row["KNN-Rec"])
        except (ValueError, KeyError):
            continue
            
        f1_dbscan.append(f1_d)
        f1_knn.append(f1_k)
        acc_dbscan.append(acc_d)
        acc_knn.append(acc_k)
        rec_dbscan.append(rec_d)
        rec_knn.append(rec_k)
        
        if dataset in SUBGROUP_A:
            sgA_f1_dbscan.append(f1_d)
            sgA_f1_knn.append(f1_k)
        elif dataset in SUBGROUP_B:
            sgB_f1_dbscan.append(f1_d)
            sgB_f1_knn.append(f1_k)

    test_results = []
    
    test_results.append(run_wilcoxon_test(f1_dbscan, f1_knn, "DBSCAN vs KNN (F1)", "Global"))
    test_results.append(run_wilcoxon_test(acc_dbscan, acc_knn, "DBSCAN vs KNN (Accuracy)", "Global"))
    test_results.append(run_wilcoxon_test(rec_dbscan, rec_knn, "DBSCAN vs KNN (Recall)", "Global"))
    
    test_results.append(run_wilcoxon_test(sgA_f1_dbscan, sgA_f1_knn, "DBSCAN vs KNN (F1 - Subgrupo A)", "Subgrupo A"))
    test_results.append(run_wilcoxon_test(sgB_f1_dbscan, sgB_f1_knn, "DBSCAN vs KNN (F1 - Subgrupo B)", "Subgrupo B"))

    test_results = [res for res in test_results if res is not None]

    print("\nResultados del Test de Wilcoxon de Rangos con Signo:\n")
    print(f"{'Comparación':<40} | {'W':<6} | {'p-valor':<8} | {'Sig(0.05)':<10} | {'Effect r':<10} | {'Interpretación'}")
    print("-" * 105)
    for res in test_results:
        comp = res['Metric']
        print(f"{comp:<40} | {res['W']:<6.1f} | {res['p-value']:<8.4f} | {res['Significant (a=0.05)']:<10} | {res['Effect size r']:<10.4f} | {res['Interpretation']}")

    os.makedirs(os.path.join(RESULTS_DIR, "statistical_tests"), exist_ok=True)
    ts = datetime.now().strftime("%d%m%Y%H%M")
    out_csv = os.path.join(RESULTS_DIR, "statistical_tests", f"wilcoxon_results_{ts}.csv")
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Subgroup", "Metric", "W", "p-value", "Significant (a=0.05)", "Effect size r", "Interpretation"])
        writer.writeheader()
        for res in test_results:
            writer.writerow(res)
            
    print(f"\n[+] Tabla de p-valores guardada en: {out_csv}")
    
    # Conclusión automática orientativa
    print("\n--- Conclusión Orientativa ---")
    f1_global = next((r for r in test_results if r["Metric"] == "DBSCAN vs KNN (F1)"), None)
    if f1_global:
        rechazo = "rechazando" if f1_global["Significant (a=0.05)"] == "Sí" else "no rechazando"
        sig = "es" if f1_global["Significant (a=0.05)"] == "Sí" else "no es"
        print(f"El test de Wilcoxon de rangos con signo arroja un p-valor de {f1_global['p-value']:.4f} "
              f"(W = {f1_global['W']:.1f}, a = 0.05), {rechazo} la hipótesis nula de igualdad de rendimiento "
              f"entre MIDBSCAN y MIKnn. El tamaño del efecto es {f1_global['Interpretation']} (r = {f1_global['Effect size r']:.4f}). "
              f"Este resultado {sig} estadísticamente significativo.")

    print("\n" + "="*80)
    print("FASE 5 COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()
