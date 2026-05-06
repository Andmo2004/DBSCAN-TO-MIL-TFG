"""
Phase 2: Optimización de Hiperparámetros (Tuning con Optuna)
Propósito: Encontrar la configuración óptima (scaler, metric, min_pts, eps) 
para cada dataset de forma sistemática y reproducible.
"""

import os
import sys
import logging

# Agregar el directorio raíz al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import optuna.visualization as vis
from optimization.best_params import run_optuna_search

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("="*70)
    print("INICIANDO PHASE 2: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*70)
    
    # Asegurar que la carpeta de resultados para las tramas de optuna existe
    os.makedirs("results/optuna_plots", exist_ok=True)
    
    # 2.1 Ejecutar la búsqueda de hiperparámetros
    # Nota: n_trials=100 como se define en 00_EXPERIMENTAL_PHASE.md
    results, studies = run_optuna_search(n_trials=100)
    
    if not studies:
        print("[!] No se generaron estudios de Optuna. Abortando visualizaciones.")
        return

    # Datasets representativos para los gráficos de convergencia (Phase 2.4)
    representative_datasets = ["musk1", "BirdsHammonds", "Thioredoxin", "Newsgroups1", "ImageElephant", "BirdsChestnut"]
    
    print("\nGenerando visualizaciones de Optuna...")
    for dataset_name, study in studies.items():
        try:
            # 2.2 Análisis de importancia de parámetros
            fig_importance = vis.plot_param_importances(study)
            # Guardar en HTML siempre funciona
            fig_importance.write_html(f"results/optuna_plots/param_importance_{dataset_name}.html")
            try:
                # Requiere la librería kaleido o orca instalada
                fig_importance.write_image(f"results/optuna_plots/param_importance_{dataset_name}.png")
            except ValueError:
                pass # kaleido no está instalado
                
            # 2.4 Gráficos de convergencia de Optuna
            if dataset_name in representative_datasets:
                fig_history = vis.plot_optimization_history(study)
                fig_history.write_html(f"results/optuna_plots/optimization_history_{dataset_name}.html")
                try:
                    fig_history.write_image(f"results/optuna_plots/optimization_history_{dataset_name}.png")
                except ValueError:
                    pass
                    
            print(f"  [+] Visualizaciones generadas para {dataset_name}.")
        except Exception as e:
            print(f"  [!] No se pudieron generar visualizaciones para {dataset_name}: {e}")
            
    print("\n" + "="*70)
    print("PHASE 2 COMPLETADA")
    print("Revisar CSV generado en 'results/' y visualizaciones en 'results/optuna_plots/'")
    print("="*70)

if __name__ == "__main__":
    main()
