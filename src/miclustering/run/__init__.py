"""
miclustering/run/__init__.py

Módulo público de ejecución de experimentos MIL desde configuración JSON.

Exporta:
    run_json   — función principal, acepta una ruta a un JSON de config.
    RunConfig  — dataclass de configuración, útil para construir configs
                 programáticamente sin necesidad de un archivo.
    run_pipeline — función de pipeline puro (datos ya en memoria).

Ejemplos de uso:
    # Caso más común: desde un archivo JSON
    from miclustering.run import run_json
    results = run_json("config.json")

    # Con guardado automático de resultados
    results = run_json("config.json", output_dir="results/")

    # Acceder a métricas
    print(results["metrics"]["F1-Score"])

    # Con datasets ya en memoria (útil en notebooks)
    from miclustering.run import run_json
    results = run_json("config.json", train_data=train, test_data=test)

    # Pipeline puro (sin I/O de archivos)
    from miclustering.run import run_pipeline, RunConfig
    config = RunConfig(dataset="musk1", algorithm="midbscan",
                       hyperparams={"epsilon": 2.8, "min_pts": 2})
    result = run_pipeline(train_data, test_data, config)
"""

from .json_runner import run_json
from ._config import RunConfig
from ._pipeline import run_pipeline

__all__ = ["run_json", "RunConfig", "run_pipeline"]