"""
miclustering/run/json_runner.py

Punto de entrada público del módulo miclustering.run.
Expone run_json(), que lee una configuración JSON, carga el dataset,
ejecuta el pipeline y devuelve (y opcionalmente guarda) los resultados.

Uso programático:
    from miclustering.run import run_json
    results = run_json("config.json")

Uso con guardado automático:
    results = run_json("config.json", output_dir="results/")

Uso con dataset ya en memoria (evita búsqueda de archivo):
    results = run_json("config.json", train_data=train, test_data=test)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from miclustering.data.arff_reader import ArffToMIData
from miclustering.data.midata import MIData

from ._config import RunConfig
from ._pipeline import run_pipeline

logger = logging.getLogger(__name__)


def run_json(
    config_path: str,
    *,
    datasets_dir: str = "datasets",
    output_dir:   Optional[str] = None,
    train_data:   Optional[MIData] = None,
    test_data:    Optional[MIData] = None,
    verbose:      bool = True,
) -> Dict[str, Any]:
    """Ejecuta un experimento MIL completo a partir de un archivo JSON.

    La función:
      1. Lee y valida el JSON de configuración.
      2. Carga el dataset desde disco (a menos que se pasen ``train_data``
         y ``test_data`` directamente, lo que omite el paso de carga).
      3. Divide en train/test según el porcentaje configurado.
      4. Llama a ``run_pipeline`` con la configuración validada.
      5. Opcionalmente guarda los resultados en ``output_dir``.

    Args:
        config_path:  Ruta al archivo ``.json`` de configuración.
        datasets_dir: Directorio donde se busca el archivo ``.arff``.
                      Se busca en: ``datasets_dir/<dataset>.arff``.
                      Ignorado si se pasan ``train_data`` y ``test_data``.
        output_dir:   Si se indica, guarda un ``.json`` con los resultados
                      en ese directorio.
        train_data:   Datos de entrenamiento ya cargados en memoria.
                      Si se pasa, también debe pasarse ``test_data``.
        test_data:    Datos de test ya cargados en memoria.
        verbose:      Si True, imprime un resumen de los resultados por consola.

    Returns:
        Diccionario con:
            ``"config"``       : dict original del JSON.
            ``"metrics"``      : métricas de clasificación (Accuracy, F1, ...).
            ``"model_stats"``  : estadísticas internas del modelo.
            ``"hyperparams"``  : hiperparámetros usados en el modelo final.
            ``"mapping"``      : mapeo cluster→clase (vacío para MIKnn).
            ``"output_file"``  : ruta al JSON guardado, o None.

    Raises:
        FileNotFoundError: Si ``config_path`` o el dataset ARFF no existen.
        ValueError:        Si la configuración es inválida.
    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw_config: Dict[str, Any] = json.load(fh)

    config = RunConfig.from_dict(raw_config)

    if verbose:
        logger.info("\n" + str(config))

    if train_data is not None and test_data is not None:
        logger.info(
            f"[run_json] Usando datasets proporcionados en memoria "
            f"(train={train_data.get_num_bags()} bolsas, "
            f"test={test_data.get_num_bags()} bolsas)."
        )
    elif train_data is not None or test_data is not None:
        raise ValueError(
            "Debes proporcionar AMBOS 'train_data' y 'test_data', o ninguno."
        )
    else:
        train_data, test_data = _load_and_split(config, datasets_dir)

    result = run_pipeline(train_data, test_data, config)

    output: Dict[str, Any] = {
        "config":      raw_config,
        "metrics":     result["metrics"],
        "model_stats": result["model_stats"],
        "hyperparams": result["hyperparams"],
        "mapping":     result["mapping"],
        "output_file": None,
    }

    if verbose:
        _print_summary(config, output)

    if output_dir is not None:
        output["output_file"] = _save_results(output, config, output_dir)

    return output

def _load_and_split(config: RunConfig, datasets_dir: str):
    """Carga el ARFF y lo divide en train/test."""
    arff_path = os.path.join(datasets_dir, f"{config.dataset}.arff")
    if not os.path.isfile(arff_path):
        raise FileNotFoundError(
            f"Dataset ARFF no encontrado: {arff_path}\n"
            f"Asegúrate de que el archivo existe o indica 'datasets_dir' correcto."
        )

    logger.info(f"[run_json] Cargando dataset: {arff_path}")
    full_dataset = ArffToMIData.from_arff(arff_path, dataset_name=config.dataset)
    train_data, test_data = full_dataset.split_data(
        percentage_train=config.train_pct, seed=config.seed
    )
    logger.info(
        f"[run_json] Split {config.train_pct:.0f}/{100 - config.train_pct:.0f} → "
        f"train={train_data.get_num_bags()} bolsas, "
        f"test={test_data.get_num_bags()} bolsas"
    )
    return train_data, test_data


def _print_summary(config: RunConfig, output: Dict[str, Any]) -> None:
    """Imprime un resumen legible de los resultados."""
    m = output["metrics"]
    ms = output["model_stats"]

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RESULTADOS — {config.dataset.upper()} / {config.algorithm.upper()}")
    print(f"{sep}")
    print(f"  {'Métrica':<18} {'Valor':>10}")
    print(f"  {'-'*30}")
    for name, val in m.items():
        print(f"  {name:<18} {val:>10.4f}")

    if ms and ms.get("status") != "not_fitted":
        print(f"\n  Estadísticas del modelo:")
        for k, v in ms.items():
            if k not in ("status", "cluster_sizes"):
                print(f"    {k}: {v}")
        if "cluster_sizes" in ms:
            print(f"    cluster_sizes: {ms['cluster_sizes']}")

    if output["mapping"]:
        print(f"\n  Mapeo cluster→clase:")
        for cid, cls in sorted(output["mapping"].items()):
            label = "Positivo (1)" if cls == 1 else "Negativo (0)"
            cname = f"Cluster {cid}" if cid >= 0 else f"Ruido ({cid})"
            print(f"    {cname:<14} → {label}")

    print(f"{sep}\n")


def _save_results(
    output: Dict[str, Any],
    config: RunConfig,
    output_dir: str,
) -> str:
    """Serializa los resultados a un archivo JSON y devuelve la ruta."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{config.algorithm}_{config.dataset}_{ts}.json"
    filepath = os.path.join(output_dir, filename)

    # Preparamos un dict serializable (convertimos numpy types)
    serializable = _make_serializable(output)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=4, ensure_ascii=False)

    logger.info(f"[run_json] Resultados guardados en: {filepath}")
    return filepath


def _make_serializable(obj: Any) -> Any:
    """Convierte recursivamente tipos no serializables por json.dump."""
    import numpy as np  # noqa: PLC0415

    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj