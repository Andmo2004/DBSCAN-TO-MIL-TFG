"""
miclustering/run/_pipeline.py

Lógica pura del pipeline de entrenamiento/predicción/evaluación MIL.
No hace I/O (no lee archivos, no escribe resultados): recibe un MIData
y un RunConfig, y devuelve un dict de métricas.

Esto permite:
  - Usarlo desde json_runner con datasets en disco.
  - Usarlo directamente desde código con datasets ya cargados en memoria.
  - Testearlo de forma aislada.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn import metrics as sk_metrics

from miclustering.data.midata import MIData
from miclustering.data.utils import parse_label
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.mikmeans import MIKMeans
from miclustering.models.mikmedoids import MIKMedoids
from miclustering.models.miknn import MIKnn
from miclustering.models.cosmic import COSMIC
from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler

from ._config import RunConfig

logger = logging.getLogger(__name__)

#  Registro de constructores de modelos 

_MODEL_REGISTRY = {
    "midbscan":   MIDBSCAN,
    "miknn":      MIKnn,
    "mikmeans":   MIKMeans,
    "mikmedoids": MIKMedoids,
    "cosmic":     COSMIC,
}

# Parámetros que aceptan random_state
_SUPPORTS_RANDOM_STATE = {"mikmeans", "mikmedoids"}

# Parámetros que aceptan precomputed_matrix en fit()
_SUPPORTS_PRECOMPUTED = {"midbscan", "mikmedoids", "cosmic"}

#  Función principal 


def run_pipeline(
    train_data: MIData,
    test_data: MIData,
    config: RunConfig,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo sobre datos ya cargados en memoria.

    Flujo:
      1. Escalar (opcional)
      2. [Opcional] Búsqueda de hiperparámetros con Optuna
      3. Construir e instanciar el modelo con los hiperparámetros finales
      4. Entrenar (con matriz precomputada si el modelo la soporta)
      5. Predecir sobre test
      6. Mapear clusters → clases (solo modelos no supervisados)
      7. Calcular métricas de evaluación

    Args:
        train_data: Conjunto de entrenamiento.
        test_data:  Conjunto de test.
        config:     RunConfig validado.

    Returns:
        Dict con claves:
            "metrics"    : Dict[str, float] con Accuracy, Precision, Recall,
                           F1-Score, Specificity.
            "model_stats": Dict con estadísticas internas del modelo (si existen).
            "hyperparams": Dict de hiperparámetros usados finalmente.
            "mapping"    : Dict cluster→clase (solo clustering), vacío para KNN.
    """

    #  1. Escalar 
    train_scaled, test_scaled = _apply_scaler(train_data, test_data, config)

    y_true_train = _labels_array(train_scaled)
    y_true_test  = _labels_array(test_scaled)

    #  2. Obtener función de distancia 
    metric_func = DISTANCE_REGISTRY[config.distance_metric]

    #  3. Precomputar matriz de distancias si el modelo la necesita 
    dist_matrix: Optional[np.ndarray] = None
    if config.algorithm in _SUPPORTS_PRECOMPUTED:
        logger.info(
            f"[pipeline] Calculando matriz de distancias "
            f"({train_scaled.get_num_bags()}×{train_scaled.get_num_bags()}) "
            f"con métrica '{config.distance_metric}'..."
        )
        dist_matrix = compute_distance_matrix(
            train_scaled.bags,
            metric_func,
            config.distance_metric,
            n_jobs=config.n_jobs,
            device=config.device,
        )

    #  4. Resolver hiperparámetros (Optuna o directos) 
    final_hyperparams = _resolve_hyperparams(
        config, train_scaled, dist_matrix, y_true_train
    )

    #  5. Instanciar y entrenar modelo 
    model = _build_model(config.algorithm, final_hyperparams)

    logger.info(f"[pipeline] Entrenando {config.algorithm}...")
    if config.algorithm in _SUPPORTS_PRECOMPUTED and dist_matrix is not None:
        model.fit(train_scaled, precomputed_matrix=dist_matrix)
    else:
        model.fit(train_scaled)

    #  6. Predecir y Mapear clusters → clases 
    mapping: Dict[int, int] = {}

    if config.algorithm == "cosmic":
        # COSMIC es transductivo: evalúa sobre train_scaled
        train_labels_dict = getattr(model, "labels", {})
        noise_label = getattr(model, "NOISE_LABEL", -1)
        y_pred_train_raw = np.array(
            [train_labels_dict.get(bag.bag_id, noise_label) for bag in train_scaled.bags]
        )
        _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(
            y_true_train, y_pred_train_raw
        )
        y_pred_mapped = np.array([mapping.get(int(c), 0) for c in y_pred_train_raw])
        eval_metrics = _compute_metrics(y_true_train, y_pred_mapped)
    elif config.algorithm == "miknn":
        logger.info(f"[pipeline] Prediciendo sobre {test_scaled.get_num_bags()} bolsas de test...")
        pred_dict = model.predict(test_scaled)
        y_pred_raw = np.array(
            [pred_dict.get(bag.bag_id, -1) for bag in test_scaled.bags]
        )
        y_pred_test = y_pred_raw
        eval_metrics = _compute_metrics(y_true_test, y_pred_test)
    else:
        logger.info(f"[pipeline] Prediciendo sobre {test_scaled.get_num_bags()} bolsas de test...")
        pred_dict = model.predict(test_scaled)
        y_pred_raw = np.array(
            [pred_dict.get(bag.bag_id, -1) for bag in test_scaled.bags]
        )
        train_labels_dict = getattr(model, "labels", {}) or model.predict(train_scaled)
        noise_label = getattr(model, "NOISE_LABEL", -1)
        y_pred_train_raw = np.array(
            [train_labels_dict.get(bag.bag_id, noise_label) for bag in train_scaled.bags]
        )
        _, mapping = MILEvaluator.hungarian_map_clusters_to_labels(
            y_true_train, y_pred_train_raw
        )
        y_pred_test = np.array([mapping.get(int(c), 0) for c in y_pred_raw])
        eval_metrics = _compute_metrics(y_true_test, y_pred_test)

    #  9. Estadísticas del modelo 
    model_stats: Dict[str, Any] = {}
    if hasattr(model, "get_statistics"):
        model_stats = model.get_statistics()

    logger.info(
        f"[pipeline] Completado — "
        f"F1={eval_metrics.get('F1-Score', 0):.4f} "
        f"Acc={eval_metrics.get('Accuracy', 0):.4f}"
    )

    return {
        "metrics":     eval_metrics,
        "model_stats": model_stats,
        "hyperparams": final_hyperparams,
        "mapping":     mapping,
    }


#  Helpers internos 


def _apply_scaler(
    train: MIData,
    test: MIData,
    config: RunConfig,
) -> Tuple[MIData, MIData]:
    """Aplica el escalador configurado (o devuelve los datos sin cambios)."""
    if config.scaler is None:
        logger.info("[pipeline] Sin escalado.")
        return train, test

    scaler = MinMaxScaler() if config.scaler == "MinMaxScaler" else StandardScaler()
    logger.info(f"[pipeline] Aplicando {config.scaler}...")
    train_scaled = scaler.fit_transform(train)
    test_scaled  = scaler.transform(test)
    return train_scaled, test_scaled


def _labels_array(dataset: MIData) -> np.ndarray:
    """Extrae las etiquetas del dataset como array numpy de enteros."""
    return np.array([parse_label(bag.label) for bag in dataset.bags])


def _build_model(algorithm: str, hyperparams: Dict[str, Any]):
    """Instancia el modelo con los hiperparámetros dados."""
    ModelClass = _MODEL_REGISTRY[algorithm]
    # Filtramos claves internas que no son parámetros del constructor
    clean = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    try:
        return ModelClass(**clean)
    except TypeError as e:
        raise ValueError(
            f"Error al construir {algorithm} con hiperparámetros {clean}: {e}"
        ) from e


def _resolve_hyperparams(
    config: RunConfig,
    train_scaled: MIData,
    dist_matrix: Optional[np.ndarray],
    y_true_train: np.ndarray,
) -> Dict[str, Any]:
    """Decide los hiperparámetros finales: Optuna o los del config.

    Si use_optuna=True, lanza la búsqueda y sobreescribe los valores
    relevantes. En caso contrario, devuelve los hiperparámetros del config
    más los valores implícitos necesarios (metric, random_state).
    """
    base: Dict[str, Any] = dict(config.hyperparams)
    base.setdefault("metric", config.distance_metric)
    if config.algorithm in _SUPPORTS_RANDOM_STATE:
        base.setdefault("random_state", config.seed)
    if config.algorithm in {"midbscan", "miknn"}:
        base.setdefault("n_jobs", config.n_jobs)

    if not config.use_optuna:
        return base

    #  Búsqueda Optuna 
    try:
        import optuna  # noqa: PLC0415
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "Optuna no está instalado. Instálalo con: pip install optuna"
        ) from exc

    logger.info(
        f"[pipeline] Iniciando búsqueda Optuna "
        f"({config.n_trials} trials, algoritmo={config.algorithm})..."
    )

    objective = _build_optuna_objective(
        config, train_scaled, dist_matrix, y_true_train
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    study.optimize(objective, n_trials=config.n_trials)

    best = study.best_params
    logger.info(f"[pipeline] Optuna best params: {best}")

    # Convertir eps_percentile → epsilon para MIDBSCAN y COSMIC
    if config.algorithm in {"midbscan", "cosmic"} and dist_matrix is not None:
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        upper_pos = upper[upper > 0]
        if len(upper_pos) > 0:
            base["epsilon"] = float(
                np.percentile(upper_pos, best.get("eps_percentile", 15.0))
            )
            if config.algorithm == "cosmic" and "eps_prime_percentile" in best:
                base["epsilon_prime"] = float(
                    np.percentile(upper_pos, best.get("eps_prime_percentile", 10.0))
                )
        base["min_pts"] = best.get("min_pts", base.get("min_pts", 2))
    else:
        base["k"] = best.get("k", base.get("k", 3))

    return base


def _build_optuna_objective(
    config: RunConfig,
    train_scaled: MIData,
    dist_matrix: Optional[np.ndarray],
    y_true_train: np.ndarray,
):
    """Devuelve la función objetivo de Optuna para el algoritmo dado."""
    from miclustering.evaluation.scoring import score_labels, detect_imbalance_ratio  # noqa: PLC0415

    imbalance = detect_imbalance_ratio(train_scaled)

    def objective(trial):
        import optuna  # noqa: PLC0415

        hp: Dict[str, Any] = {"metric": config.distance_metric}

        if config.algorithm == "midbscan":
            hp["min_pts"]  = trial.suggest_int("min_pts", 2, 20)
            eps_pct        = trial.suggest_float("eps_percentile", 1.0, 40.0)
            if dist_matrix is None:
                raise optuna.exceptions.TrialPruned()
            upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            upper_pos = upper[upper > 0]
            if len(upper_pos) == 0:
                raise optuna.exceptions.TrialPruned()
            hp["epsilon"] = float(np.percentile(upper_pos, eps_pct))

        elif config.algorithm == "cosmic":
            hp["min_pts"] = trial.suggest_int("min_pts", 2, 20)
            eps_pct = trial.suggest_float("eps_percentile", 10.0, 60.0)
            if dist_matrix is None:
                raise optuna.exceptions.TrialPruned()
            upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            upper_pos = upper[upper > 0]
            if len(upper_pos) == 0:
                raise optuna.exceptions.TrialPruned()
            hp["epsilon"] = float(np.percentile(upper_pos, eps_pct))
            eps_prime_pct = trial.suggest_float("eps_prime_percentile", 1.0, eps_pct)
            hp["epsilon_prime"] = float(np.percentile(upper_pos, eps_prime_pct))

        elif config.algorithm == "miknn":
            hp["k"] = trial.suggest_int("k", 1, 15)

        elif config.algorithm in {"mikmeans", "mikmedoids"}:
            hp["k"]            = trial.suggest_int("k", 2, 15)
            hp["random_state"] = config.seed

        try:
            model = _build_model(config.algorithm, hp)

            if config.algorithm == "miknn":
                # Para KNN, validación interna 80/20
                sub_train, sub_val = train_scaled.split_data(
                    percentage_train=80, seed=config.seed
                )
                model.fit(sub_train)
                preds = model.predict(sub_val)
                y_val = _labels_array(sub_val)
                y_pred_val = np.array(
                    [preds.get(bag.bag_id, 0) for bag in sub_val.bags]
                )
                return float(
                    sk_metrics.f1_score(
                        y_val, y_pred_val, zero_division=0, average="weighted"
                    )
                )
            else:
                if config.algorithm in _SUPPORTS_PRECOMPUTED and dist_matrix is not None:
                    model.fit(train_scaled, precomputed_matrix=dist_matrix)
                else:
                    model.fit(train_scaled)

                if config.algorithm == "midbscan" and getattr(model, "cluster_count", 0) == 0:
                    return 0.0

                return score_labels(train_scaled, model.labels, imbalance_ratio=imbalance)

        except Exception:
            raise optuna.exceptions.TrialPruned()

    return objective


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula el conjunto estándar de métricas de clasificación binaria."""
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415

    acc  = float(sk_metrics.accuracy_score(y_true, y_pred))
    prec = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0, average="weighted"))
    rec  = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0, average="weighted"))
    f1   = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0, average="weighted"))
    f1_macro = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0, average="macro"))

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0

    return {
        "Accuracy":    acc,
        "Precision":   prec,
        "Recall":      rec,
        "F1-Score":    f1,
        "F1-Macro":    f1_macro,
        "Specificity": specificity,
    }