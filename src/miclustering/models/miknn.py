""" 
Implementación del algoritmo k-Nearest Neighbors adaptado para
Multi-Instance Learning (MIL).
 
Estrategia:
  - En MIL, la unidad de clasificación es la BOLSA (Bag), no la instancia.
  - La distancia entre dos bolsas se calcula con las métricas ya implementadas
    en el proyecto (Hausdorff, Cauchy-Schwarz).
  - Clasificación: dado un bag de test, se buscan sus k vecinos más cercanos
    en el conjunto de entrenamiento y se asigna la clase mayoritaria (majority
    voting) con desempate por distancia acumulada.
 
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import logging
import os
import sys
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

# Configurar PYTHONPATH para ejecución individual
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
 
from miclustering.data.bag import Bag
from miclustering.data.midata import MIData
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.distance_matrix import compute_distance_matrix
 
logger = logging.getLogger(__name__)

class MIKnn(BaseEstimator, ClassifierMixin):
    '''
    Clasificador k-Nearest Neighbors para Multi-Instance Learning.
    '''

 

    def __init__(self, k: int = 3, metric: str = "hausdorff"):
        """Constructor del clasificador MIKnn.
 
        Args:
            k: Número de vecinos más cercanos a considerar. Debe ser >= 1.
            metric: Nombre de la función de distancia entre bolsas.
                    Valores válidos: 'hausdorff', 'hausdorff_min', 'hausdorff_avg', 
                    'cauchy_schwarz', 'earth_movers', 'mahalanobis'.
 
        Raises:
            ValueError: Si k < 1 o si la métrica no está registrada.
        """
        if k < 1:
            raise ValueError(
                f"El parámetro 'k' debe ser >= 1. Recibido: {k}"
            )
        if metric.lower() not in DISTANCE_REGISTRY:
            valid = list(DISTANCE_REGISTRY.keys())
            raise ValueError(
                f"Métrica '{metric}' no reconocida. Disponibles: {valid}"
            )
 
        self._k            = k
        self._metric_name  = metric.lower()
        self._metric_func: Callable[[Bag, Bag], float] = (
            DISTANCE_REGISTRY[self._metric_name]
        )
 
        # Estado interno
        self._train_bags:   List[Bag] = []
        self._train_labels: List[int] = []   # etiquetas numéricas (0/1)
        self._fitted:       bool      = False
 
        logger.debug(f"MIKnn creado: k={k}, metric='{metric}'")

    #  Propiedades 
 
    @property
    def k(self) -> int:
        """Número de vecinos."""
        return self._k
 
    @property
    def metric_name(self) -> str:
        """Nombre de la métrica de distancia."""
        return self._metric_name
 
    @property
    def is_fitted(self) -> bool:
        """True si el modelo ha sido entrenado con fit()."""
        return self._fitted
 
    @property
    def n_train_bags(self) -> int:
        """Número de bolsas de entrenamiento almacenadas."""
        return len(self._train_bags)
    
    #  Publico 
 
    def fit(self, dataset: MIData) -> "MIKnn":
        """Memoriza el conjunto de entrenamiento (lazy learning).
 
        k-NN es un algoritmo de aprendizaje perezoso: fit() únicamente
        almacena las bolsas y sus etiquetas; no construye ningún modelo.
 
        Args:
            dataset: MIData con las bolsas y etiquetas de entrenamiento.

        Returns:
            Self, para encadenamiento de métodos.
 
        Raises:
            ValueError: Si el dataset está vacío.
        """
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de entrenamiento está vacío.")
 
        if self._k > dataset.get_num_bags():
            logger.warning(
                f"k={self._k} > n_bags={dataset.get_num_bags()}. "
                f"Se usarán todos los bags como vecinos."
            )
 
        self._train_bags   = list(dataset.bags)
        self._train_labels = [
            self._parse_label(bag.label) for bag in self._train_bags
        ]
        self._fitted = True
 
        logger.info(
            f"MIKnn entrenado: {len(self._train_bags)} bolsas, "
            f"k={self._k}, métrica='{self._metric_name}'"
        )
        return self
    
    def predict(self, dataset: MIData) -> Dict[str, int]:
        """Predice la etiqueta de clase para cada bolsa del dataset.
 
        Para cada bolsa de test:
          1. Calcula la distancia a todas las bolsas de train.
          2. Selecciona los k vecinos más cercanos.
          3. Asigna la clase mayoritaria (con desempate por distancia).
 
        Args:
            dataset: MIData con las bolsas a clasificar.

        Returns:
            Diccionario {bag_id: clase_predicha (0 ó 1)}.
 
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
            ValueError: Si el dataset está vacío.
        """
        self._check_fitted()
 
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")
 
        predictions: Dict[str, int] = {}
 
        for test_bag in dataset.bags:
            distances = self._compute_distances_to_train(test_bag)
            label     = self._classify(distances)
            predictions[test_bag.bag_id] = label
 
        logger.info(
            f"MIKnn predijo {len(predictions)} bolsas."
        )
        return predictions
    
    def predict_bag(self, bag: Bag) -> int:
        """Predice la etiqueta de una única bolsa.
 
        Args:
            bag: Objeto Bag a clasificar.

        Returns:
            Clase predicha (int, normalmente 0 ó 1).
 
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
        """
        self._check_fitted()
        distances = self._compute_distances_to_train(bag)
        return self._classify(distances)    
    
    def predict_proba(self, dataset: MIData) -> Dict[str, Dict[int, float]]:
        """Estima la probabilidad de cada clase mediante la proporción de votos
        de los k vecinos.
 
        Args:
            dataset: MIData con las bolsas a clasificar.

        Returns:
            Diccionario {bag_id: {clase: proporción_de_votos}}.
            Ejemplo: {"bag_001": {0: 0.33, 1: 0.67}}
 
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
            ValueError: Si el dataset está vacío.
        """
        self._check_fitted()
 
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")
 
        probas: Dict[str, Dict[int, float]] = {}
 
        for test_bag in dataset.bags:
            distances = self._compute_distances_to_train(test_bag)
            neighbors = self._get_k_neighbors(distances)
 
            neighbor_labels = [self._train_labels[idx] for idx, _ in neighbors]
            counts          = Counter(neighbor_labels)
            total           = len(neighbor_labels)
 
            classes  = sorted(set(self._train_labels))
            proba    = {cls: counts.get(cls, 0) / total for cls in classes}
            probas[test_bag.bag_id] = proba
 
        return probas
 
    def fit_predict(self, X: MIData, y: Optional[MIData] = None) -> Dict[str, int]:  # type: ignore
        """
        Entrena el modelo y devuelve las predicciones.
        Para cumplir con scikit-learn, recibe X y opcionalmente y.
        """
        self.fit(X)
        if y is not None:
            return self.predict(y)
        return self.predict(X)
 
    def get_neighbors(
        self, bag: Bag
    ) -> List[Tuple[str, int, float]]:
        """Devuelve los k vecinos más cercanos de una bolsa junto con sus
        etiquetas y distancias.
 
        Args:
            bag: Bolsa de consulta.

        Returns:
            Lista de tuplas (bag_id, etiqueta, distancia) ordenada de menor
            a mayor distancia.
 
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
        """
        self._check_fitted()
        distances = self._compute_distances_to_train(bag)
        neighbors = self._get_k_neighbors(distances)
 
        return [
            (
                self._train_bags[idx].bag_id,
                self._train_labels[idx],
                dist,
            )
            for idx, dist in neighbors
        ]
 
    def get_statistics(self) -> Dict[str, Any]:
        """Devuelve un resumen del estado del modelo.
 
        Returns:
            Diccionario con claves:
                - ``k``            : número de vecinos.
                - ``metric``       : nombre de la métrica.
                - ``n_train_bags`` : bolsas de entrenamiento almacenadas.
                - ``label_counts`` : distribución de clases en train.
                - ``fitted``       : bool.
        """
        if not self._fitted:
            return {"fitted": False}
 
        label_counts = dict(Counter(self._train_labels))
        return {
            "k":            self._k,
            "metric":       self._metric_name,
            "n_train_bags": len(self._train_bags),
            "label_counts": label_counts,
            "fitted":       True,
        }
 
    #  Métodos internos 
 
    def _compute_distances_to_train(self, test_bag: Bag) -> np.ndarray:
        """Calcula la distancia de test_bag a cada bolsa de entrenamiento.
 
        Args:
            test_bag: Bolsa de consulta.

        Returns:
            Array (n_train,) con las distancias.
        """
        distances = np.empty(len(self._train_bags), dtype=float)
        for i, train_bag in enumerate(self._train_bags):
            distances[i] = self._metric_func(test_bag, train_bag)
        return distances
 
    def _get_k_neighbors(
        self, distances: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Selecciona los k vecinos más cercanos.
 
        Args:
            distances: Array (n_train,) con distancias al bag de consulta.

        Returns:
            Lista de (índice_en_train, distancia) ordenada de menor a mayor,
            con longitud min(k, n_train).
        """
        k_eff = min(self._k, len(distances))
        # argsort parcial (los k primeros)
        sorted_idx = np.argsort(distances)[:k_eff]
        return [(int(idx), float(distances[idx])) for idx in sorted_idx]
 
    def _classify(self, distances: np.ndarray) -> int:
        """Asigna la clase a un bag a partir de su vector de distancias a train.
 
        Estrategia:
          1. Majority voting sobre los k vecinos.
          2. Desempate: gana la clase con menor distancia acumulada
             entre sus representantes en el vecindario.
 
        Args:
            distances: Array (n_train,) con distancias al bag de consulta.

        Returns:
            Clase predicha (int).
        """
        neighbors = self._get_k_neighbors(distances)
 
        # Acumular votos y distancias por clase
        vote_count:   Dict[int, int]   = {}
        dist_accum:   Dict[int, float] = {}
 
        for idx, dist in neighbors:
            label = self._train_labels[idx]
            vote_count[label]  = vote_count.get(label, 0) + 1
            dist_accum[label]  = dist_accum.get(label, 0.0) + dist
 
        # Clase con más votos
        max_votes = max(vote_count.values())
        candidates = [
            cls for cls, votes in vote_count.items() if votes == max_votes
        ]
 
        if len(candidates) == 1:
            return candidates[0]
 
        # Desempate: menor distancia acumulada
        return min(candidates, key=lambda cls: dist_accum[cls])
 
    @staticmethod
    def _parse_label(raw_label: Any) -> int:
        """Convierte etiquetas de bolsa (str, float, int, bytes) a int.
 
        Args:
            raw_label: Etiqueta en formato crudo (e.g. '1.0', b'0', 1).

        Returns:
            Etiqueta como int.
        """
        if isinstance(raw_label, bytes):
            raw_label = raw_label.decode("utf-8")
        return int(float(raw_label))
 
    def _check_fitted(self):
        """Lanza RuntimeError si el modelo no ha sido entrenado.
 
        Raises:
            RuntimeError: Si ``is_fitted`` es False.
        """
        if not self._fitted:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Ejecuta fit() primero."
            )
 
    #  Representaciones 
    
    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (
            f"<MIKnn(k={self._k}, metric='{self._metric_name}', "
            f"status={state})>"
        )
 
    def __str__(self) -> str:
        if not self._fitted:
            return f"MIKnn (Unfitted): k={self._k}, metric='{self._metric_name}'"
 
        stats = self.get_statistics()
        label_dist = "  ".join(
            f"clase {c}: {n}" for c, n in sorted(stats["label_counts"].items())
        )
        return (
            f"MIKnn Model:\n"
            f"  - Config  : k={self._k}, metric='{self._metric_name}'\n"
            f"  - Status  : Fitted on {stats['n_train_bags']} bags\n"
            f"  - Clases  : {label_dist}"
        )
 
 
#  Prueba individual 
 
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
 
    from miclustering.preprocessing.scaler import MinMaxScaler
    from miclustering.evaluation.bcm import MILEvaluator
 
    try:
        # 1. Cargar dataset
        logger.info("Cargando dataset musk1.arff...")
        full_data = MIData.from_arff("datasets/musk1.arff")
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
 
        # 2. Normalizar
        scaler       = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled  = scaler.transform(test_data)
 
        # 3. Entrenar y predecir
        knn = MIKnn(k=3, metric="hausdorff")
        knn.fit(train_scaled)
 
        print("\n" + "=" * 50)
        print(knn)
        print("=" * 50)
 
        predictions = knn.predict(test_scaled)
 
        # 4. Evaluar
        MILEvaluator.evaluate(test_scaled, predictions, title="MIKnn — musk1")
 
        # 5. Ejemplo predict_proba
        first_bag = test_scaled.bags[0]
        probas    = knn.predict_proba(test_scaled)
        print(f"\nProbabilidades para '{first_bag.bag_id}': {probas[first_bag.bag_id]}")
 
        # 6. Vecinos más cercanos de la primera bolsa
        neighbors = knn.get_neighbors(first_bag)
        print(f"\nVecinos más cercanos de '{first_bag.bag_id}':")
        for bag_id, label, dist in neighbors:
            print(f"  {bag_id:<20} clase={label}  dist={dist:.4f}")
 
    except Exception as e:
        logger.error(f"Error en la prueba: {e}", exc_info=True)