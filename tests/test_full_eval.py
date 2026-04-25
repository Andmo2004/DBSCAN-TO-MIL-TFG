"""
tests/test_full_evaluation.py
 
Evaluación completa para todos los datasets: CVIs internos + métricas externas.
 
CVIs internos (sin ground-truth):
  Grupo 1 — Solo Compactibilidad:
    SED, DD, Hc
  Grupo 2 — Compactibilidad + Separación:
    VRC, I
 
Métricas externas (con ground-truth):
    Precision, Recall, F1-Score, Specificity
 
Información adicional:
    - Info del dataset: distribución de clases, instancias por bolsa
    - Composición de cada cluster: positivos/negativos por cluster
    - Matriz de confusión: TN, FP, FN, TP
 
Salida:
    - Reporte detallado por dataset en consola
    - Tabla resumen en consola al final
    - CSV en results/full_eval_<timestamp>.csv
"""

import os
import sys
import csv
import logging
from datetime import datetime
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
 
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
 
from data.midata import MIData
from models.midbscan import MIDBSCAN
from preprocessing.scaler import MinMaxScaler, StandardScaler
from evaluation.cvi import SEDIndex, DDIndex, HcIndex, VRCIndex, IIndex
from evaluation.bcm import MILEvaluator
 
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test_full_evaluation")

# ── Datasets a evaluar ─────────────────────────────────────────────────────────

DATASETS_CONFIG = [
    {"dataset_name": "musk1",              "arff_name": "musk1",                         "best_scaler": MinMaxScaler,   "best_distance": "hausdorff",      "best_eps": 2.1673,   "best_min_pts": 2},
    {"dataset_name": "musk2",              "arff_name": "musk2",                         "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.02026,  "best_min_pts": 3},
    {"dataset_name": "ImageElephant",      "arff_name": "ImageElephant",                 "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.11840,  "best_min_pts": 2},
    {"dataset_name": "BirdsChestnut",      "arff_name": "BirdsChestnut-backedChickadee", "best_scaler": StandardScaler, "best_distance": "cauchy_schwarz", "best_eps": 0.2988,   "best_min_pts": 10},
    {"dataset_name": "BirdsHammonds",      "arff_name": "BirdsHammondsFlycatcher",       "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.00565,  "best_min_pts": 2},
    {"dataset_name": "Harddrive1",         "arff_name": "Harddrive1",                    "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.003467, "best_min_pts": 3},
    {"dataset_name": "mutagenesis_atoms",  "arff_name": "mutagenesis3_atoms",            "best_scaler": StandardScaler, "best_distance": "hausdorff",      "best_eps": 0.4748,   "best_min_pts": 3},
    {"dataset_name": "mutagenesis_chains", "arff_name": "mutagenesis3_chains",           "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.006638, "best_min_pts": 3},
    {"dataset_name": "Newsgroups1",        "arff_name": "Newsgroups1",                   "best_scaler": StandardScaler, "best_distance": "hausdorff",      "best_eps": 50.434,   "best_min_pts": 2},
    {"dataset_name": "Thioredoxin",        "arff_name": "Thioredoxin",                   "best_scaler": MinMaxScaler,   "best_distance": "cauchy_schwarz", "best_eps": 0.001185, "best_min_pts": 2},
]

DATASETS_DIR = "datasets"
RESULTS_DIR  = "results"
 
# CVIs internos: (instancia, necesita_X)
INTERNAL_CVIS = [
    (SEDIndex(), True),
    (DDIndex(),  True),
    (HcIndex(),  False),
    (VRCIndex(), True),
    (IIndex(),   True),
]
# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_X(dataset: MIData, bag_ids: list) -> np.ndarray:
    """Centroides de bolsas (N x n_features) alineados con bag_ids."""
    bag_index = {bag.bag_id: bag for bag in dataset.bags}
    return np.array([
        np.mean(bag_index[bid].as_matrix(), axis=0)
        for bid in bag_ids
    ])
 
 
def _label_as_int(raw_label) -> int:
    """Convierte la etiqueta de una bolsa a entero (0 o 1)."""
    return int(float(raw_label)) if isinstance(raw_label, (str, float)) else int(raw_label)
 
 
def compute_dataset_stats(dataset: MIData, split_name: str) -> dict:
    """Estadisticas generales de un split del dataset."""
    bags        = dataset.bags
    n_bags      = len(bags)
    n_pos       = sum(1 for b in bags if _label_as_int(b.label) == 1)
    n_neg       = n_bags - n_pos
    inst_counts = [len(b) for b in bags]
 
    return {
        "split":    split_name,
        "n_bags":   n_bags,
        "n_pos":    n_pos,
        "n_neg":    n_neg,
        "inst_min": int(np.min(inst_counts)),
        "inst_max": int(np.max(inst_counts)),
        "inst_avg": float(np.mean(inst_counts)),
        "inst_std": float(np.std(inst_counts)),
    }
 
 
def compute_cluster_composition(dataset: MIData, labels: dict) -> dict:
    """Para cada cluster, cuenta bolsas positivas y negativas."""
    composition: dict = {}
    for bag in dataset.bags:
        cid       = labels.get(bag.bag_id, -1)
        label_val = _label_as_int(bag.label)
        if cid not in composition:
            composition[cid] = {"pos": 0, "neg": 0, "total": 0}
        composition[cid]["total"] += 1
        if label_val == 1:
            composition[cid]["pos"] += 1
        else:
            composition[cid]["neg"] += 1
    return composition
 
 
def compute_external_metrics(dataset: MIData, predicted_labels: dict) -> dict:
    """
    Mapeo hungaro + metricas externas + matriz de confusion.
    No imprime nada: devuelve todo en un dict.
    """
    y_true, y_pred_raw = [], []
    for bag in dataset.bags:
        if bag.bag_id in predicted_labels:
            y_true.append(_label_as_int(bag.label))
            y_pred_raw.append(predicted_labels[bag.bag_id])
 
    if not y_true:
        logger.warning("No hay etiquetas para evaluar.")
        return {}
 
    y_true     = np.array(y_true)
    y_pred_raw = np.array(y_pred_raw)
 
    y_pred_mapped, mapping = MILEvaluator.hungarian_map_clusters_to_labels(
        y_true, y_pred_raw
    )
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_mapped, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
 
    return {
        "Precision":   float(precision_score(y_true, y_pred_mapped, zero_division=0)),
        "Recall":      float(recall_score(y_true, y_pred_mapped, zero_division=0)),
        "F1-Score":    float(f1_score(y_true, y_pred_mapped, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "Accuracy":    float((tp + tn) / total) if total > 0 else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "mapping": mapping,
    }

# ── Evaluación ────────────────────────────────────────────────────────────────

def run_evaluation(config: dict) -> dict:
    """
    Pipeline completo para un dataset.
    Devuelve un dict con todos los resultados (para el CSV de resumen).
    """
    name       = config["dataset_name"]
    arff_name  = config["arff_name"]
    scaler_cls = config["best_scaler"]
    metric     = config["best_distance"]
    eps        = config["best_eps"]
    min_pts    = config["best_min_pts"]
 
    summary = {
        "dataset":   name,
        "scaler":    scaler_cls.__name__,
        "metric":    metric,
        "eps":       eps,
        "min_pts":   min_pts,
        "status":    "Failed",
        "error":     "",
        "train_bags": 0, "test_bags": 0,
        "train_pos": 0,  "train_neg": 0,
        "test_pos": 0,   "test_neg": 0,
        "clusters":  0,  "noise_pct": 0.0, "core_pts": 0,
        "SED": None, "DD": None, "Hc": None, "VRC": None, "I": None,
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "Specificity": 0.0, "Accuracy": 0.0,
        "TN": 0, "FP": 0, "FN": 0, "TP": 0,
    }
 
    try:
        # 1. Cargar y dividir
        path    = os.path.join(DATASETS_DIR, f"{arff_name}.arff")
        dataset = MIData.from_arff(path)
        train, test = dataset.split_data(percentage_train=70, seed=42)
 
        # 2. Estadisticas del dataset
        train_stats = compute_dataset_stats(train, "train")
        test_stats  = compute_dataset_stats(test,  "test")
 
        summary.update({
            "train_bags": train_stats["n_bags"], "test_bags": test_stats["n_bags"],
            "train_pos":  train_stats["n_pos"],  "train_neg": train_stats["n_neg"],
            "test_pos":   test_stats["n_pos"],   "test_neg":  test_stats["n_neg"],
        })
 
        # 3. Escalar
        scaler       = scaler_cls()
        train_scaled = scaler.fit_transform(train)
        test_scaled  = scaler.transform(test)
 
        # 4. Entrenar
        model = MIDBSCAN(epsilon=eps, min_pts=min_pts, metric=metric)
        model.fit(train_scaled)
        model_stats = model.get_statistics()
 
        summary.update({
            "clusters":  model_stats["num_clusters"],
            "noise_pct": round(model_stats["noise_percentage"], 1),
            "core_pts":  model_stats["num_core_points"],
        })
 
        # 5. Composicion de clusters (etiquetas reales de train)
        cluster_composition = compute_cluster_composition(train, model.labels)
 
        # 6. Predecir sobre test
        test_labels = model.predict(test_scaled)
 
        # 7. CVIs internos
        bag_ids = [bag.bag_id for bag in train_scaled.bags]
        dm      = np.zeros((len(bag_ids), len(bag_ids)))
        X       = compute_X(train_scaled, bag_ids)
 
        internal_results: dict = {}
        if model_stats["num_clusters"] == 0:
            for cvi, _ in INTERNAL_CVIS:
                internal_results[cvi.name] = None
        else:
            for cvi, needs_X in INTERNAL_CVIS:
                try:
                    val = cvi.compute(dm, model.labels, bag_ids, X=X if needs_X else None)
                    internal_results[cvi.name] = float(val)
                except Exception as exc:
                    logger.warning(f"[{name}][{cvi.name}] Error: {exc}")
                    internal_results[cvi.name] = None
 
        for k, v in internal_results.items():
            summary[k] = round(v, 6) if v is not None else None
 
        # 8. Metricas externas
        external_results = compute_external_metrics(test_scaled, test_labels)
 
        summary.update({
            "Precision":   round(external_results.get("Precision",   0.0), 4),
            "Recall":      round(external_results.get("Recall",      0.0), 4),
            "F1":          round(external_results.get("F1-Score",    0.0), 4),
            "Specificity": round(external_results.get("Specificity", 0.0), 4),
            "Accuracy":    round(external_results.get("Accuracy",    0.0), 4),
            "TN": external_results.get("TN", 0), "FP": external_results.get("FP", 0),
            "FN": external_results.get("FN", 0), "TP": external_results.get("TP", 0),
        })
 
        summary["status"] = "Success"
 
        # 9. Reporte detallado en consola
        _print_report(
            name=name, scaler_name=scaler_cls.__name__, metric=metric,
            eps=eps, min_pts=min_pts,
            train_stats=train_stats, test_stats=test_stats,
            model_stats=model_stats, composition=cluster_composition,
            internal=internal_results, external=external_results,
        )
 
    except Exception as exc:
        summary["error"] = str(exc)
        logger.error(f"[{name}] Error fatal: {exc}", exc_info=True)
        print(f"\n  x {name}: {exc}\n")
 
    return summary
 

# ── Reporte detallado (por dataset) ──────────────────────────────────────────
 
def _print_report(
    name, scaler_name, metric, eps, min_pts,
    train_stats, test_stats, model_stats,
    composition, internal, external,
):
    W = 60
 
    print(f"\n{'='*W}")
    print(f"  EVALUACION COMPLETA -- {name}")
    print(f"{'='*W}")
    print(f"  Scaler   : {scaler_name}")
    print(f"  Metrica  : {metric}")
    print(f"  eps      : {eps}   min_pts: {min_pts}")
 
    # -- Info del dataset
    print(f"\n  -- Info del Dataset")
    print(f"  {'':4} {'Split':<8} {'Bolsas':>7} {'Pos':>6} {'Neg':>6}  "
          f"Inst/bolsa (min / avg / max / std)")
    print(f"  {'_'*58}")
    for st in (train_stats, test_stats):
        print(
            f"  {'':4} {st['split']:<8} {st['n_bags']:>7} "
            f"{st['n_pos']:>6} {st['n_neg']:>6}  "
            f"{st['inst_min']} / {st['inst_avg']:.1f} / "
            f"{st['inst_max']} / {st['inst_std']:.1f}"
        )
 
    # -- Estadisticas del modelo
    print(f"\n  -- Modelo (entrenado sobre train)")
    print(f"  Clusters encontrados : {model_stats['num_clusters']}")
    print(f"  Puntos nucleo        : {model_stats['num_core_points']}")
    print(f"  Bolsas ruido         : {model_stats['noise_points_count']}  "
          f"({model_stats['noise_percentage']:.1f}%)")
 
    # -- Composicion de clusters
    print(f"\n  -- Composicion de Clusters (train -- etiquetas reales)")
    print(f"  {'Cluster':<14} {'Total':>7} {'Pos':>6} {'Neg':>6}  "
          f"{'% Pos':>7}  {'Pureza':>7}")
    print(f"  {'_'*54}")
    for cid in sorted(composition.keys()):
        info    = composition[cid]
        total   = info["total"]
        pos     = info["pos"]
        neg     = info["neg"]
        pct_pos = (pos / total * 100) if total > 0 else 0.0
        purity  = (max(pos, neg) / total * 100) if total > 0 else 0.0
        label   = f"Ruido (-1)" if cid == -1 else f"Cluster {cid}"
        print(f"  {label:<14} {total:>7} {pos:>6} {neg:>6}  "
              f"{pct_pos:>6.1f}%  {purity:>6.1f}%")
 
    # -- CVIs internos
    groups = [
        ("Solo Compactibilidad   (menor es mejor)", ["SED", "DD", "Hc"]),
        ("Compact. + Separacion  (mayor es mejor)", ["VRC", "I"]),
    ]
    print(f"\n  -- CVIs Internos (sobre train escalado)")
    for group_label, cvi_names in groups:
        print(f"\n  {group_label}")
        for cvi_name in cvi_names:
            val = internal.get(cvi_name)
            if val is None:
                val_str = "        N/A"
            elif abs(val) >= 1e15:
                val_str = "         inf"
            else:
                val_str = f"{val:>11.4f}"
            print(f"    {cvi_name:<6} {val_str}")
 
    # -- Metricas externas
    print(f"\n  -- Metricas Externas (sobre test -- mayor es mejor)")
    for key in ["Precision", "Recall", "F1-Score", "Specificity", "Accuracy"]:
        val     = external.get(key)
        val_str = f"{val:>11.4f}" if val is not None else "        N/A"
        print(f"    {key:<14} {val_str}")
 
    # -- Matriz de confusion
    if all(k in external for k in ("TN", "FP", "FN", "TP")):
        tn, fp = external["TN"], external["FP"]
        fn, tp = external["FN"], external["TP"]
        total  = tn + fp + fn + tp
        acc    = (tp + tn) / total if total > 0 else 0.0
 
        print(f"\n  -- Matriz de Confusion (test)")
        print(f"                  Pred 0      Pred 1")
        print(f"  Real 0      {tn:>8}    {fp:>8}   (Neg reales: {tn+fp})")
        print(f"  Real 1      {fn:>8}    {tp:>8}   (Pos reales: {fn+tp})")
        print(f"  {'_'*40}")
        print(f"  Accuracy: {acc:.4f}  ({tp+tn}/{total} correctos)")
 
    # -- Mapeo hungaro
    if external.get("mapping"):
        print(f"\n  -- Mapeo Hungaro: Cluster -> Clase (test)")
        for cid, cls in sorted(external["mapping"].items()):
            cls_str = "Positivo (1)" if cls == 1 else "Negativo (0)"
            lbl     = f"Ruido ({cid})" if cid < 0 else f"Cluster {cid}"
            print(f"    {lbl:<14}  ->  {cls_str}")
 
    print(f"\n{'='*W}\n")
 
 
# ── Tabla resumen en consola ──────────────────────────────────────────────────
 
def _print_summary(all_summaries: list):
    print("\n" + "=" * 104)
    print(
        f"  {'DATASET':<22} {'STATUS':<9} {'CLUSTERS':>8} {'RUIDO%':>7} "
        f"{'F1':>7} {'PREC':>7} {'REC':>7} {'SPEC':>7} {'ACC':>7}"
    )
    print("  " + "-" * 100)
    for s in all_summaries:
        icon = "OK" if s["status"] == "Success" else "!!"
        print(
            f"  [{icon}] {s['dataset']:<20} {s['status']:<9} "
            f"{s['clusters']:>8} {s['noise_pct']:>6.1f}% "
            f"{s['F1']:>7.4f} {s['Precision']:>7.4f} "
            f"{s['Recall']:>7.4f} {s['Specificity']:>7.4f} {s['Accuracy']:>7.4f}"
        )
    print("=" * 104 + "\n")
 
 
# ── Guardar CSV ───────────────────────────────────────────────────────────────
 
def _save_csv(all_summaries: list) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%d%m%Y%H%M")
    out_path = os.path.join(RESULTS_DIR, f"full_eval_{ts}.csv")
 
    fieldnames = [
        "dataset", "scaler", "metric", "eps", "min_pts", "status", "error",
        "train_bags", "train_pos", "train_neg",
        "test_bags",  "test_pos",  "test_neg",
        "clusters", "noise_pct", "core_pts",
        "SED", "DD", "Hc", "VRC", "I",
        "Precision", "Recall", "F1", "Specificity", "Accuracy",
        "TN", "FP", "FN", "TP",
    ]
 
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_summaries)
 
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    print(f"\n{'#'*60}")
    print(f"  TEST COMPLETO -- {len(DATASETS_CONFIG)} datasets")
    print(f"{'#'*60}")
 
    all_summaries = []
 
    for i, config in enumerate(DATASETS_CONFIG, start=1):
        print(f"\n[{i}/{len(DATASETS_CONFIG)}] Procesando: {config['dataset_name']}",
              flush=True)
        summary = run_evaluation(config)
        all_summaries.append(summary)
 
    _print_summary(all_summaries)
 
    csv_path = _save_csv(all_summaries)
    print(f"CSV guardado -> {csv_path}\n")
 
 
if __name__ == "__main__":
    main()