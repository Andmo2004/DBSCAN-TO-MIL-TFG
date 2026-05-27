"""
miclustering/run/_config.py

Dataclass tipada que representa la configuración de una ejecución.
Se construye desde un dict (JSON) y valida los campos antes de ejecutar
el pipeline, evitando errores tardíos dentro de la lógica de negocio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from miclustering.distances import DISTANCE_REGISTRY

VALID_ALGORITHMS = {"midbscan", "miknn", "mikmeans", "mikmedoids"}
VALID_SCALERS = {"MinMaxScaler", "StandardScaler", None}
VALID_METRICS_EVAL = {"F1-Score", "Accuracy", "Precision", "Recall", "Specificity"}

# Mapeo tolerante de alias comunes en los JSON de configuración
_KEY_ALIASES: Dict[str, str] = {
    # algoritmo
    "algoritmo":                         "algorithm",
    "model":                             "algorithm",
    "medida_de_distancia":               "distance_metric",
    "medida_distancia":                  "distance_metric",
    "distance":                          "distance_metric",
    # scaler
    "metodo_de_escalado":                "scaler",
    "metodo_escalado":                   "scaler",
    "scaling_method":                    "scaler",
    # semilla
    "semilla":                           "seed",
    "random_seed":                       "seed",
    # metrica evaluación
    "metrica_de_rendimiento_a_optimizar": "eval_metric",
    "metrica_optimizacion":              "eval_metric",
    "performance_metric":                "eval_metric",
    # hiperparámetros
    "hiperparametros":                   "hyperparams",
    "hyperparameters":                   "hyperparams",
    # optuna
    "optimizar_optuna":                  "use_optuna",
    "optimize_optuna":                   "use_optuna",
    "optuna_trials":                     "n_trials",
    # partición
    "porcentaje_entrenamiento":          "train_pct",
    "percentage_train":                  "train_pct",
}


#  Dataclass principal 

@dataclass
class RunConfig:
    """Configuración completa y validada de una ejecución del pipeline MIL.

    Attributes:
        dataset:         Nombre del dataset (sin extensión .arff).
        algorithm:       Nombre del modelo a usar (midbscan, miknn, mikmeans, mikmedoids).
        distance_metric: Nombre de la función de distancia del DISTANCE_REGISTRY.
        scaler:          Nombre del escalador o None para no escalar.
        seed:            Semilla de aleatoriedad.
        eval_metric:     Métrica que se reportará como principal en los resultados.
        hyperparams:     Dict de hiperparámetros que se inyectarán al constructor del modelo.
        use_optuna:      Si True, busca hiperparámetros con Optuna antes de entrenar.
        n_trials:        Número de trials de Optuna (solo relevante si use_optuna=True).
        train_pct:       Porcentaje de bolsas para entrenamiento (0-100).
        extra:           Campos adicionales no reconocidos (se conservan para debug).
    """

    dataset:         str
    algorithm:       str      = "midbscan"
    distance_metric: str                  = "hausdorff"
    scaler:          Optional[str]        = "MinMaxScaler"
    seed:            int                  = 42
    eval_metric:     str                  = "F1-Score"
    hyperparams:     Dict[str, Any]       = field(default_factory=dict)
    use_optuna:      bool                 = False
    n_trials:        int                  = 30
    train_pct:       float                = 70.0
    extra:           Dict[str, Any]       = field(default_factory=dict)

    #  Constructor alternativo 

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RunConfig":
        """Construye un RunConfig desde un dict crudo (por ejemplo, leído de JSON).

        Aplica el mapeo de aliases antes de construir la dataclass y valida
        los campos obligatorios y los valores permitidos.

        Args:
            raw: Diccionario con las claves en cualquier formato soportado.

        Returns:
            Instancia de RunConfig validada.

        Raises:
            ValueError: Si algún campo obligatorio falta o contiene un valor inválido.
        """
        # Normalizar claves (minúsculas + strip) y aplicar aliases
        normalized: Dict[str, Any] = {}
        for k, v in raw.items():
            canonical = _KEY_ALIASES.get(k.lower().strip(), k.lower().strip())
            normalized[canonical] = v

        #  Campo obligatorio: dataset 
        if "dataset" not in normalized:
            raise ValueError(
                "El campo 'dataset' es obligatorio en la configuración JSON."
            )

        #  Defaults + extracción de campos conocidos 
        known_fields = {
            f.name for f in cls.__dataclass_fields__.values()  # type: ignore[attr-defined]
        } - {"extra"}

        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for k, v in normalized.items():
            if k in known_fields:
                kwargs[k] = v
            else:
                extra[k] = v

        kwargs["extra"] = extra

        #  Normalización de tipos 
        if "algorithm" in kwargs:
            kwargs["algorithm"] = str(kwargs["algorithm"]).lower().strip()

        if "scaler" in kwargs and kwargs["scaler"] in ("", "none", "None", None):
            kwargs["scaler"] = None

        if "hyperparams" in kwargs and not isinstance(kwargs["hyperparams"], dict):
            raise ValueError(
                f"El campo 'hyperparams' debe ser un objeto JSON (dict). "
                f"Recibido: {type(kwargs['hyperparams'])}"
            )

        instance = cls(**kwargs)
        instance._validate()
        return instance

    #  Validación interna 

    def _validate(self) -> None:
        """Lanza ValueError si algún campo contiene un valor inválido."""

        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Algoritmo '{self.algorithm}' no reconocido. "
                f"Disponibles: {sorted(VALID_ALGORITHMS)}"
            )

        if self.distance_metric not in DISTANCE_REGISTRY:
            raise ValueError(
                f"Métrica de distancia '{self.distance_metric}' no registrada. "
                f"Disponibles: {sorted(DISTANCE_REGISTRY.keys())}"
            )

        if self.scaler not in VALID_SCALERS:
            raise ValueError(
                f"Escalador '{self.scaler}' no reconocido. "
                f"Disponibles: {sorted(s for s in VALID_SCALERS if s)}"
            )

        if not (0 < self.train_pct < 100):
            raise ValueError(
                f"'train_pct' debe estar en (0, 100). Recibido: {self.train_pct}"
            )

        if self.n_trials < 1:
            raise ValueError(
                f"'n_trials' debe ser >= 1. Recibido: {self.n_trials}"
            )

    #  Representación 

    def __str__(self) -> str:
        lines = [
            f"RunConfig:",
            f"  dataset        : {self.dataset}",
            f"  algorithm      : {self.algorithm}",
            f"  distance_metric: {self.distance_metric}",
            f"  scaler         : {self.scaler}",
            f"  seed           : {self.seed}",
            f"  train_pct      : {self.train_pct}%",
            f"  hyperparams    : {self.hyperparams}",
            f"  use_optuna     : {self.use_optuna} (trials={self.n_trials})",
        ]
        return "\n".join(lines)