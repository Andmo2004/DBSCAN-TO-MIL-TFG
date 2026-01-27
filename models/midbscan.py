from typing import List, Dict, Optional, Callable, Tuple, Any, Set
import numpy as np
import logging
from collections import Counter

from data.midata import MIData
from data.bag import Bag
from distances.hausdorff import hausdorff_distance

logger = logging.getLogger(__name__)

class MIDBSCAN:
    """
    Implementación del Algoritmo DBSCAN adaptado para el Multi-Instance Learning.
    """

    NOISE_LABEL = -1

    def __init__(self, epsilon: float, min_pts: int, metric: str = 'hausdorff'):
        """
        Constructor del modelo MIDBSCAN
        
        :param self:
        :param epsilon: (float) Distancia máxima entre dos muestras para que se consideren en la vecindad.
        :param min_pts: (int) El número de muestras mínimo en vecindad para que un punto se considere núcleo.
        :param metric: (str) métrica de distanci usada

        :Raises ValueError: Si epsilon <= 0 o min_pts < 1

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
    def noise_label(self) -> int:
        """Etiqueta utilizada para puntos de ruido."""
        return self.NOISE_LABEL

    # Propiedad is_fitted
    @property
    def is_fitted(self) -> bool:
        """Indica si el modelo ha sido entrenado."""
        return self._fitted

    # 12. Método _reset_state()
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
            Selecciona la función de distancia basada en el nombre.
            Registro de métricas implementadas
            """
            # Registro de métricas disponibles
            metrics_registry = {
                'hausdorff': hausdorff_distance,
                # 'mean': mean_distance, 
            }

            if name not in metrics_registry:
                valid_keys = list(metrics_registry.keys())
                raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
            
            return metrics_registry[name]

    def _compute_distance_matrix(self, bags: List[Bag]) -> np.ndarray:
        """
        Calcula la matriz de distancias Hausdorff simétrica.        
        :param bags: (List[Bag]) Lista de Bolsas
        :return: (ndarray[_AnyShape, dtype[Any]]) Matriz numpy de distancias (N X N)
        """

        num_bags = len(bags)
        logger.info(f"Calculando matriz ({num_bags}x{num_bags}) usando métrica: '{self._metric_name}'...")

        # Inicializamos matriz a 0
        matrix = np.zeros((num_bags, num_bags))

        dist_func = self._metric_func

        for i in range(num_bags):
            bag_a = bags[i]
            for j in range(i + 1, num_bags):
                bag_b = bags[j]
                
                d = dist_func(bag_a, bag_b)

                matrix[i, j] = d
                matrix[j, i] = d

        logger.debug("Cálculo de matriz de distancias finalizado.")
        return matrix

    def _add_core_point(self, bag: Bag, cluster_id: int):
        """Registra un punto como núcleo para uso futuro en predicciones."""
        self._core_bags.append(bag)
        self._core_bag_labels[bag.bag_id] = cluster_id

    def fit(self, dataset: MIData):
        """
        Entrenar el modelo DBSCAN, con el dataset

        :param dataset:(MIData) Objeto MIData con las bolsas de entrenamiento
        :return: Retorna la instancia del modelo para permitir encadenamiento.

        :raises ValueError: Si el dataset está vacío

        """

        if dataset.get_num_bags() == 0:
            error_msg = "El dataset de entrenamiento está vacío."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._reset_state()

        # Accedemos a la propiedad un única vez
        bags = dataset.bags
        self._train_bags = bags
        num_bags = len(bags)

        # Calculamos la matriz de distancias
        self._distance_matrix = self._compute_distance_matrix(bags)

        logger.info(f"Iniciando clustering DBSCAN (eps={self._epsilon}, min_pts={self._min_pts})...")

        # Vector de visitados a false
        visited = np.zeros(num_bags, dtype=bool)
        
        # Inicializamos etiquetas como None
        bag_cluster_map: Dict[str, Optional[int]] = {b.bag_id: None for b in bags}
        current_cluster_id = 0

        # Comenzamos Algoritmo
        for i in range(num_bags):
            if visited[i]:
                continue

            visited[i] = True
            
            # Buscamos vecinos
            neighbors_index = np.where(self._distance_matrix[i] <= self._epsilon)[0]

            if len(neighbors_index) < self._min_pts:
                # Marcar como ruido (puede cambiar luego si es alcanzable por otro cluster)
                bag_cluster_map[bags[i].bag_id] = self.NOISE_LABEL

            else:
                self._add_core_point(bags[i], current_cluster_id)
                bag_cluster_map[bags[i].bag_id] = current_cluster_id

            logger.debug(f"Cluster {current_cluster_id} iniciado en bolsa {bags[i].bag_id}")

            self._expand_cluster(
                                neighbors_index,
                                current_cluster_id,
                                self._distance_matrix,
                                visited,
                                bag_cluster_map,
                                bags
                            )
            current_cluster_id += 1

            # Limpiamos el diccionario
            # Si v es None, lo cambiamos por self.NOISE_LABEL (que suele ser -1).
            self._labels = {k: (v if v is not None else self.NOISE_LABEL) for k, v in bag_cluster_map.items()}
            self._cluster_count = current_cluster_id
            self._fitted = True 

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
        """
        Predice etiquetas para un nuevo dataset basándose en los clústeres aprendidos.

        :param test_dataset: (MIData) Dataset de test
        :return: (Dict[str, int]) Diccionario de predicciones
        
        :raises RuntimeError: Si el modelo no ha sido entrenado.
        :raises ValueError: Si el dataset está vacío.

        """

        if not self._fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de llamar a predict(). Ejecuta .fit() primero.")
                
        if test_dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")

        if not self._core_bags:
            logger.warning("Modelo entrenado sin puntos núcleo (todo fue ruido). Asignando RUIDO a todo el test set.")
            return {bag.bag_id: self.NOISE_LABEL for bag in test_dataset.bags}
        
        logger.info(f"Prediciendo {test_dataset.get_num_bags()} bolsas de prueba usando {len(self._core_bags)} núcleos...")

        test_labels = {}
        noise_count = 0

        for test_bag in test_dataset.bags:
            best_dist = float('inf')
            assigned_cluster = self.NOISE_LABEL
            dist_func = self._metric_func

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

        percentage = (noise_count / test_dataset.get_num_bags()) * 100
        logger.info(f"Predicción completada: {noise_count} bolsas asignadas como ruido ({percentage:.2f}%)")
        
        return test_labels
    
    def fit_predict(self, train_dataset: MIData, test_dataset: MIData) -> Dict[str, int]:
        """
        Entrena el modelo con train_dataset y devuelve predicciones para test_dataset.
        
        :param train_dataset: (MIData) Dataset de entrenamiento.
        :param test_dataset: (MIData) Dataset de prueba.
        :return: (Dict[str, int]) Diccionario de etiquetas para el dataset de prueba.
        """
        return self.fit(train_dataset).predict(test_dataset)
    
    def get_cluster_sizes(self) -> Dict[int, int]:
            """
            Devuelve el conteo de elementos por cluster.
            :returns: (Dict[int, int]) Diccionario {cluster_id: cantidad}.
            """
            if not self._fitted:
                return {}
            return dict(Counter(self._labels.values()))

    def get_noise_points(self) -> List[str]:
            """
            Devuelve una lista con los IDs de las bolsas consideradas ruido.
            :returns: (List[str]) Lista de strings (bag_ids).
            """
            if not self._fitted:
                return []
            return [bid for bid, label in self._labels.items() if label == self.NOISE_LABEL]

    def get_cluster_members(self, cluster_id: int) -> List[str]:
            """
            Devuelve los IDs de las bolsas que pertenecen a un cluster específico.
            
            :param cluster_id: (int) ID del cluster a consultar.
            :returns: (List[str]) Lista de bag_ids.
            """
            if not self._fitted:
                return []
            return [bid for bid, label in self._labels.items() if label == cluster_id]                 
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Genera un reporte completo de estadísticas del modelo entrenado.
        :returns: (Dict[str, Any]) Diccionario con métricas detalladas.
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

    def __repr__(self) -> str:
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