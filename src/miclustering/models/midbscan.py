from typing import List, Dict, Optional, Callable, Tuple, Any, Set
import numpy as np
import logging
from collections import Counter
from sklearn.base import BaseEstimator, ClusterMixin

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)

class MIDBSCAN(BaseEstimator, ClusterMixin):
    """
    Implementación del Algoritmo DBSCAN adaptado para el Multi-Instance Learning.
    """

    NOISE_LABEL = -1

    def __init__(
        self,
        epsilon: float,
        min_pts: int,
        metric: str = 'hausdorff',
        n_jobs: int = 1,
        device: str = "cpu",
    ):
        """Constructor del modelo MIDBSCAN.
        
        Args:
            epsilon: Distancia máxima entre dos muestras para que se consideren en la vecindad.
            min_pts: El número de muestras mínimo en vecindad para que un punto se considere núcleo.
            metric: Métrica de distancia usada.
            n_jobs: Número de procesos paralelos para cómputo (-1 para todos los núcleos).
            device: Dispositivo de cómputo ('cpu', 'cuda', 'mps', 'auto').

        Raises:
            ValueError: Si epsilon <= 0 o min_pts < 1.
        """

        if epsilon <= 0:
            raise ValueError(f"El parámetro 'epsilon' debe ser > 0. Recibido: {epsilon}")
        if min_pts < 1:
            raise ValueError(f"El parámetro 'min_pts' debe ser >= 1. Recibido: {min_pts}")
        
        # Uso de Encapsulamiento _*
        # Parametros del algoritmo
        self._epsilon = epsilon
        self._min_pts = min_pts
        self._metric_name = metric.lower()
        self._n_jobs = n_jobs
        self._device = (device or "cpu").lower().strip()

        # Función de métrica a usar
        self._metric_func = self._get_metric_function(self._metric_name)

        # Estado del modelo
        self._labels: Dict[str, int] = {}
        self._cluster_count = 0
        self._fitted = False
        self._train_bags: List[Bag] = []

        # Almacenamos matriz de distancias
        self._distance_matrix: Optional[np.ndarray] = None

        # Guardamos Cores y sus labels, para el predict
        self._core_bags: List[Bag] = []
        self._core_bag_labels: Dict[str, int] = {}

        logger.debug(f"MIDBSCAN inicializado: epsilon={epsilon}, min_pts={min_pts}")

    # Propiedades @property (Solo lectura)
    @property
    def epsilon(self) -> float:
        """Radio de vecindad (solo lectura)."""
        return self._epsilon

    @property
    def min_pts(self) -> int:
        """Mínimo de puntos para núcleo (solo lectura)."""
        return self._min_pts

    @property
    def n_jobs(self) -> int:
        """Número de procesos paralelos para cómputo."""
        return self._n_jobs

    @property
    def device(self) -> str:
        """Dispositivo de cómputo configurado."""
        return self._device

    @property
    def cluster_count(self) -> int:
        """Número de clústeres encontrados (excluyendo ruido)."""
        return self._cluster_count

    @property
    def labels(self) -> Dict[str, int]:
        """
        Devuelve un diccionario con las etiquetas asignadas {bag_id: cluster_id}.
        """
        return self._labels.copy()

    @property
    def labels_(self) -> np.ndarray:
        """Etiquetas de clúster asignadas a cada bolsa de entrenamiento (array numpy)."""
        if not self._fitted:
            raise AttributeError("El modelo no ha sido entrenado. Ejecuta fit() primero.")
        return np.array([self._labels.get(bag.bag_id, self.NOISE_LABEL) for bag in self._train_bags])
    
    @property
    def noise_label(self) -> int:
        """Etiqueta utilizada para puntos de ruido."""
        return self.NOISE_LABEL

    # Propiedad is_fitted
    @property
    def is_fitted(self) -> bool:
        """Indica si el modelo ha sido entrenado."""
        return self._fitted

    # Método _reset_state()
    def _reset_state(self):
        """Reinicia el estado interno del modelo antes de un nuevo ajuste."""
        self._labels = {}
        self._cluster_count = 0
        self._fitted = False
        self._core_bags = []
        self._core_bag_labels = {}
        self._train_bags = []
        self._distance_matrix = None

    # Usamos callable para dedevolver una función    
    def _get_metric_function(self, name: str) -> Callable[[Bag, Bag], float]:
            """
            Selecciona la función de distancia basada en el nombre usando el registro central.
            """
            if name not in DISTANCE_REGISTRY:
                valid_keys = list(DISTANCE_REGISTRY.keys())
                raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
            
            return DISTANCE_REGISTRY[name]

    def _compute_distance_matrix(self, bags: List[Bag]) -> np.ndarray:
        """Calcula la matriz de distancias usando el módulo externo.
        
        Args:
            bags: Lista de Bolsas.

        Returns:
            Matriz numpy de distancias (N X N).
        """
        return compute_distance_matrix(
            bags, self._metric_func, self._metric_name, n_jobs=self._n_jobs, device=self._device
        )

    def _add_core_point(self, bag: Bag, cluster_id: int):
        """Registra un punto como núcleo para uso futuro en predicciones."""
        self._core_bags.append(bag)
        self._core_bag_labels[bag.bag_id] = cluster_id

    def fit(
        self,
        dataset: MIData,
        precomputed_matrix: Optional[np.ndarray] = None,
    ) -> "MIDBSCAN":
        """Entrenar el modelo DBSCAN con el dataset.
 
        Args:
            dataset: Objeto MIData con las bolsas de entrenamiento.
            precomputed_matrix: Matriz de distancias (N×N) ya calculada.
                                Si se proporciona, se omite el cálculo interno.
                                Debe estar alineada con ``dataset.bags`` en el mismo orden.
 
        Returns:
            Instancia del modelo para permitir encadenamiento.
 
        Raises:
            ValueError: Si el dataset está vacío o la forma de la matriz no coincide
                        con el número de bolsas.
        """
        if dataset.get_num_bags() == 0:
            error_msg = "El dataset de entrenamiento está vacío."
            logger.error(error_msg)
            raise ValueError(error_msg)
 
        if precomputed_matrix is not None:
            n = dataset.get_num_bags()
            if precomputed_matrix.shape != (n, n):
                raise ValueError(
                    f"precomputed_matrix shape {precomputed_matrix.shape} "
                    f"no coincide con n_bags={n}"
                )
 
        self._reset_state()
 
        bags = dataset.bags
        self._train_bags = bags
        num_bags = len(bags)
 
        if precomputed_matrix is not None:
            self._distance_matrix = precomputed_matrix
            logger.debug("Reutilizando matriz de distancias precomputada.")
        else:
            self._distance_matrix = self._compute_distance_matrix(bags)
 
        logger.info(f"Iniciando clustering DBSCAN (eps={self._epsilon}, min_pts={self._min_pts})...")
    
        visited = np.zeros(num_bags, dtype=bool)
        bag_cluster_map: Dict[str, Optional[int]] = {b.bag_id: None for b in bags}
        current_cluster_id = 0
 
        for i in range(num_bags):
            if visited[i]:
                continue
 
            visited[i] = True
            neighbors_index = np.where(self._distance_matrix[i] <= self._epsilon)[0]
 
            if len(neighbors_index) < self._min_pts:
                bag_cluster_map[bags[i].bag_id] = self.NOISE_LABEL
                continue
            
            logger.debug(f"Cluster {current_cluster_id} iniciado en bolsa {bags[i].bag_id}")
            self._add_core_point(bags[i], current_cluster_id)
            bag_cluster_map[bags[i].bag_id] = current_cluster_id
            self._expand_cluster(
                neighbors_index,
                current_cluster_id,
                self._distance_matrix,
                visited,
                bag_cluster_map,
                bags,
            )
            current_cluster_id += 1
 
        self._labels = {k: (v if v is not None else self.NOISE_LABEL) for k, v in bag_cluster_map.items()}
        self._cluster_count = current_cluster_id
        self._fitted = True
        
        self._distance_matrix = None
        
        return self 

    def _expand_cluster(self, 
                        initial_neighbors: np.ndarray, 
                        cluster_id: int, 
                        dist_matrix: np.ndarray, 
                        visited: np.ndarray, 
                        bag_labels: Dict[str, Optional[int]], 
                        bags: List[Bag]):
            """Expande el clúster visitando vecinos recursivamente."""
            
            queue = list(initial_neighbors)
            seen_in_queue = set(initial_neighbors) # Optimización O(1)
            
            i = 0
            while i < len(queue):
                neighbor_idx = queue[i]
                i += 1
                
                bag = bags[neighbor_idx]
                bag_id = bag.bag_id
                
                # Si era ruido, ahora es parte del borde del clúster
                if bag_labels[bag_id] == self.NOISE_LABEL:
                    bag_labels[bag_id] = cluster_id
                
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    bag_labels[bag_id] = cluster_id
                    
                    new_neighbors = np.where(dist_matrix[neighbor_idx] <= self._epsilon)[0]
                    
                    if len(new_neighbors) >= self._min_pts:
                        self._add_core_point(bag, cluster_id)
                        
                        for n_idx in new_neighbors:
                            if n_idx not in seen_in_queue:
                                seen_in_queue.add(n_idx)
                                queue.append(n_idx)
                
                if bag_labels[bag_id] is None:
                    bag_labels[bag_id] = cluster_id
        
    def predict(self, test_dataset: MIData) -> Dict[str, int]:
        """Predice etiquetas para un nuevo dataset basándose en los clústeres aprendidos.

        Args:
            test_dataset: Dataset de test.

        Returns:
            Diccionario de predicciones {bag_id: cluster_id}.
        
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
            ValueError: Si el dataset está vacío.
        """

        if not self._fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de llamar a predict(). Ejecuta .fit() primero.")
                
        if test_dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")

        if not self._core_bags:
            logger.warning("Modelo entrenado sin puntos núcleo (todo fue ruido). Asignando RUIDO a todo el test set.")
            return {bag.bag_id: self.NOISE_LABEL for bag in test_dataset.bags}
        
        logger.info(f"Prediciendo {test_dataset.get_num_bags()} bolsas de prueba usando {len(self._core_bags)} núcleos...")

        test_bags = list(test_dataset.bags)
        test_labels = {}
        noise_count = 0
        dist_func = self._metric_func

        if self._n_jobs == 1 or len(test_bags) <= 10:
            for test_bag in test_bags:
                best_dist = float('inf')
                assigned_cluster = self.NOISE_LABEL

                # Compararemos solo los puntos núcleo (más optimizado para datasets grandes)
                for core_bag in self._core_bags:
                    dist = dist_func(test_bag, core_bag)

                    if dist <= self._epsilon:
                        if dist < best_dist:
                            best_dist = dist
                            assigned_cluster = self._core_bag_labels[core_bag.bag_id]

                test_labels[test_bag.bag_id] = assigned_cluster
                if assigned_cluster == self.NOISE_LABEL:
                    noise_count += 1
        else:
            from joblib import Parallel, delayed

            def _predict_single(test_bag):
                best_dist = float('inf')
                assigned = self.NOISE_LABEL
                for core_bag in self._core_bags:
                    dist = dist_func(test_bag, core_bag)
                    if dist <= self._epsilon and dist < best_dist:
                        best_dist = dist
                        assigned = self._core_bag_labels[core_bag.bag_id]
                return test_bag.bag_id, assigned

            results = Parallel(n_jobs=self._n_jobs, backend="loky")(
                delayed(_predict_single)(b) for b in test_bags
            )
            for bag_id, assigned in results:
                test_labels[bag_id] = assigned
                if assigned == self.NOISE_LABEL:
                    noise_count += 1

        percentage = (noise_count / len(test_bags)) * 100
        logger.info(f"Predicción completada: {noise_count} bolsas asignadas como ruido ({percentage:.2f}%)")
        
        return test_labels
    
    def fit_predict(self, X: MIData, y: Optional[MIData] = None) -> Dict[str, int]:  # type: ignore
        """
        Entrena el modelo y devuelve las predicciones.
        Para cumplir con scikit-learn, recibe X y opcionalmente y.
        """
        self.fit(X)
        if y is not None:
            return self.predict(y)
        return getattr(self, "labels", {})
    
    def get_cluster_sizes(self) -> Dict[int, int]:
            """Devuelve el conteo de elementos por cluster.

            Returns:
                Diccionario {cluster_id: cantidad}.
            """
            if not self._fitted:
                return {}
            return dict(Counter(self._labels.values()))

    def get_noise_points(self) -> List[str]:
            """Devuelve una lista con los IDs de las bolsas consideradas ruido.

            Returns:
                Lista de strings (bag_ids).
            """
            if not self._fitted:
                return []
            return [bid for bid, label in self._labels.items() if label == self.NOISE_LABEL]

    def get_cluster_members(self, cluster_id: int) -> List[str]:
            """Devuelve los IDs de las bolsas que pertenecen a un cluster específico.
            
            Args:
                cluster_id: ID del cluster a consultar.

            Returns:
                Lista de bag_ids.
            """
            if not self._fitted:
                return []
            return [bid for bid, label in self._labels.items() if label == cluster_id]                 
    
    def get_statistics(self) -> Dict[str, Any]:
        """Genera un reporte completo de estadísticas del modelo entrenado.

        Returns:
            Diccionario con métricas detalladas.
        """
        if not self._fitted:
            return {"status": "not_fitted"}
        
        total_points = len(self._labels)
        noise_points = self.get_noise_points()
        num_noise = len(noise_points)
        noise_pct = (num_noise / total_points * 100) if total_points > 0 else 0
        
        return {
            "epsilon": self._epsilon,
            "min_pts": self._min_pts,
            "total_bags": total_points,
            "num_clusters": self._cluster_count,
            "num_core_points": len(self._core_bags),
            "noise_points_count": num_noise,
            "noise_percentage": noise_pct,
            "cluster_sizes": self.get_cluster_sizes()
        }

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (f"<MIDBSCAN(epsilon={self._epsilon}, min_pts={self._min_pts}, "
                f"clusters={self._cluster_count}, status={state})>")

    def __str__(self) -> str:
        if not self._fitted:
            return f"MIDBSCAN (Unfitted): eps={self._epsilon}, min_pts={self._min_pts}"
        
        stats = self.get_statistics()
        return (f"MIDBSCAN Model:\n"
                f"  - Config: eps={self._epsilon}, min_pts={self._min_pts}\n"
                f"  - Status: Fitted on {stats['total_bags']} bags\n"
                f"  - Clusters Found: {self._cluster_count}\n"
                f"  - Core Points: {stats['num_core_points']}\n"
                f"  - Noise: {stats['noise_points_count']} bags ({stats['noise_percentage']:.2f}%")
    

##### PRUEBA INDIVIDUAL ########
if __name__ == "__main__":
    # Configurar logging para ver la salida profesional
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 1. Cargar
        full_data = MIData.from_arff("datasets/musk1.arff") 
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
        
        # 2. Instanciar (con validación de errores si pones valores negativos)
        dbscan = MIDBSCAN(epsilon=900.0, min_pts=2) # Musk usa valores altos para Hausdorff
        
        # 3. Entrenar
        # Nota: fit() ahora devuelve self, permitiendo method chaining si quisieras
        dbscan.fit(train_data)
        
        # 4. Ver representación string mejorada
        print("\n" + "="*50)
        print(dbscan)
        print("="*50 + "\n")
        
        # 5. Obtener estadísticas detalladas (Nueva Funcionalidad)
        stats = dbscan.get_statistics()
        print("Distribución de Clusters:", stats['cluster_sizes'])
        
        # 6. Predecir
        test_results = dbscan.predict(test_data)
        
        # 7. Verificaciones de la nueva API
        print(f"\n¿Está entrenado? {dbscan.is_fitted}")
        print(f"Ruido detectado en entrenamiento: {len(dbscan.get_noise_points())} bolsas")
        
    except Exception as e:
        logger.error(f"Error fatal en la ejecución: {e}")