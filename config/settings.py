"""
Módulo de configuración principal del proyecto.
Centraliza las constantes, rutas y configuraciones óptimas de los datasets,
facilitando su mantenimiento y evitando dependencias circulares o imports
desde módulos de tests hacia código de producción.
"""

import os
import glob
import csv
from preprocessing.scaler import MinMaxScaler, StandardScaler

# ── Rutas Base ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Configuraciones Óptimas por Dataset ──────────────────────────────────────
# Mapeo para recuperar las clases reales de scaler a partir de sus nombres en string
_SCALER_MAP = {
    "MinMaxScaler": MinMaxScaler,
    "StandardScaler": StandardScaler,
}

_ARFF_NAMES = {
    "BirdsChestnut": "BirdsChestnut-backedChickadee",
    "BirdsHammonds": "BirdsHammondsFlycatcher",
    "mutagenesis_atoms": "mutagenesis3_atoms",
    "mutagenesis_chains": "mutagenesis3_chains",
}

# Valores de configuración por defecto (se actualizarán automáticamente si hay un CSV de Optuna)
KNOWN_BESTS = {
    "musk1":              {"scaler": "MinMaxScaler",   "metric": "hausdorff",      "min_pts": 2, "eps_abs": 2.821350},
    "musk2":              {"scaler": "StandardScaler", "metric": "hausdorff",      "min_pts": 2, "eps_abs": 10.495062},
    "ImageElephant":      {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2, "eps_abs": 0.112952},
    "BirdsChestnut":      {"scaler": "MinMaxScaler",   "metric": "hausdorff_avg",  "min_pts": 3, "eps_abs": 0.547650},
    "BirdsHammonds":      {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2, "eps_abs": 0.005651},
    "Harddrive1":         {"scaler": "MinMaxScaler",   "metric": "hausdorff_avg",  "min_pts": 2, "eps_abs": 0.193147},
    "mutagenesis_atoms":  {"scaler": "StandardScaler", "metric": "hausdorff",      "min_pts": 3, "eps_abs": 0.475836},
    "mutagenesis_chains": {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 3, "eps_abs": 0.006637},
    "Newsgroups1":        {"scaler": "StandardScaler", "metric": "hausdorff",      "min_pts": 2, "eps_abs": 49.300286},
    "Thioredoxin":        {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2, "eps_abs": 0.001186},
}

def _load_latest_optuna_params():
    search_pattern = os.path.join(RESULTS_DIR, "optuna_best_params_*.csv")
    csv_files = glob.glob(search_pattern)
    if not csv_files:
        return
    
    latest_csv = max(csv_files, key=os.path.getmtime)
    try:
        with open(latest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("dataset")
                if dataset in KNOWN_BESTS:
                    KNOWN_BESTS[dataset].update({
                        "scaler": row["scaler"],
                        "metric": row["metric"],
                        "min_pts": int(row["min_pts"]),
                        "eps_abs": float(row["eps_absolute"])
                    })
    except Exception as e:
        print(f"[!] Error al cargar parámetros óptimos desde {latest_csv}: {e}")

# Actualizar KNOWN_BESTS con el último CSV de resultados si existe
_load_latest_optuna_params()

DATASETS_CONFIG = [
    {
        "dataset_name": dataset_name,
        "arff_name": _ARFF_NAMES.get(dataset_name, dataset_name),
        "best_scaler": _SCALER_MAP[params["scaler"]],
        "best_distance": params["metric"],
        "best_eps": params["eps_abs"],
        "best_min_pts": params["min_pts"],
    }
    for dataset_name, params in KNOWN_BESTS.items()
]
