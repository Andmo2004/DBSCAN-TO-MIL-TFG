from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.attribute import Attribute

logger = logging.getLogger(__name__)

class BaseScaler(ABC):
    """
    Clase base para el preprocesado de datos Multi Instancia.
    Seguimos el patrón sckit-learn
    """
    def __init__(self):
        # Entrenado?
        self._fitted = False
        # Esquema usado
        self._schema: Optional[List[Attribute]] = None
        # Indice numerico
        self._numeric_indices: List[int] = []
        self._numeric_to_position: Dict[int, int] = {}
  
    @property
    def is_fitted(self) -> bool:
        """Indica si el scaler ha sido entrenado."""
        return self._fitted
    
    @property
    def schema(self) -> Optional[List[Attribute]]:
        """Esquema de atributos utilizado en fit."""
        return self._schema
    
    @property
    def numeric_indices(self) -> List[int]:
        """Índices de atributos numéricos."""
        return self._numeric_indices.copy()
    
    @abstractmethod
    def fit(self, dataset: MIData) -> 'BaseScaler':
        """Aprende los parámetros de transformación desde el dataset.

        Args:
            dataset: Dataset de entrenamiento.

        Returns:
            Self para encadenamiento de métodos.
        """
        ...
    
    @abstractmethod
    def transform(self, dataset: MIData, inplace: bool = False) -> MIData:
        """Aplica la transformación al dataset.
        
        Args:
            dataset: Dataset a transformar.
            inplace: Si True, modifica el dataset original, si False, crea copia.

        Returns:
            Dataset transformado.
        """
        ...
    
    def fit_transform(self, dataset: MIData) -> MIData:
        """Entrena y transforma en un solo paso.
        
        Args:
            dataset: Dataset a transformar.

        Returns:
            Dataset transformado.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    def _extract_schema(self, dataset: MIData) -> List[Attribute]:
        """Extrae el esquema de atributos del dataset.
        
        Args:
            dataset: Dataset del cual extraer esquema.

        Returns:
            Lista de atributos.
            
        Raises:
            ValueError: Si el dataset está vacío.
        """
        if len(dataset) == 0:
            raise ValueError("El dataset está vacío")
        
        first_bag = dataset[0]
        if len(first_bag) == 0:
            raise ValueError("La primera bolsa está vacía")
        
        return first_bag[0].schema
    
    def _identify_numeric_indices(self, schema: List[Attribute]) -> List[int]:
        """Identifica índices de atributos numéricos."""
        indices = []
        valid_types = ['real', 'integer', 'numeric', 'float', 'int']
        
        # Debug: Ver qué tipos está leyendo realmente
        if schema:
            logger.debug(f"Tipos encontrados en el esquema: {[attr.type for attr in schema]}")

        for i, attr in enumerate(schema):
            # Limpiamos el string (minusculas y sin espacios extra)
            attr_type_clean = str(attr.type).lower().strip()
            
            if attr_type_clean in valid_types:
                indices.append(i)
        
        self._numeric_to_position = {idx: pos for pos, idx in enumerate(indices)}
        logger.debug(f"Identificados {len(indices)} atributos numéricos")
        return indices
    
    def _validate_schema(self, dataset: MIData):
        """Valida que el dataset tenga el mismo esquema que el entrenamiento.
        
        Args:
            dataset: Dataset a validar.
        
        Raises:
            RuntimeError: Si no está entrenado.
            ValueError: Si el esquema no coincide.
        """

        if not self._fitted:
            raise RuntimeError("El scaler no ha sido entrenado. Ejecuta fit() primero.")
        
        if self._schema is None:
            raise RuntimeError("El esquema no fue guardado durante fit(). Estado interno inconsistente.")
        
        current_schema = self._extract_schema(dataset)

        if len(current_schema) != len(self._schema):
            raise ValueError(
                f"Número de atributos inconsistente: "
                f"entrenado con {len(self._schema)}, recibido {len(current_schema)}"                    
            )

        # Validamos por los tipos de atributos
        for i, (train_attr, curr_attr) in enumerate(zip(self._schema, current_schema)):
            if train_attr.type != curr_attr.type:
                raise ValueError(
                    f"Tipo de atributo {i} inconsistente: "
                    f"entrenado con '{train_attr.type}', recibido '{curr_attr.type}'"
                )
    

    def _collect_numeric_data(self, dataset: MIData) -> np.ndarray:
        """Recolecta todos los valores numéricos del dataset en una matriz.
        
        Args:
            dataset: Dataset del cual extraer datos.

        Returns:
            Matriz numpy (total_instances x num_numeric_features).
        
        Raises:
            ValueError: Si no se encuentran instancias en el dataset.
        """ 
        all_values = []

        for bag in dataset:
            for instance in bag:
                # Extraemos de forma segura los valores nativos numéricos por posición
                instance_vals = list(instance.values)
                numeric_vals = [float(instance_vals[idx]) for idx in self._numeric_indices]
                all_values.append(numeric_vals)

        if not all_values:
            raise ValueError("No se encontraron instancias en el dataset")
        
        # Generamos la matriz limpia y tipada únicamente con atributos numéricos
        return np.array(all_values, dtype=np.float64)

    def _create_transformed_dataset(self, original: MIData, transform_fn) -> MIData:
        """Crea un nuevo dataset aplicando una función de transformación.
        
        Args:
            original: Dataset original.
            transform_fn: Función que transforma valores numéricos.

        Returns:
            Nuevo dataset transformado.
        """

        new_bags = []

        for bag in original:
            new_instances = []

            for instance in bag:
                new_values = list(instance.values)

                # Transformamos solo valores numericos
                for idx in self._numeric_indices:
                    try:
                        old_val = float(new_values[idx])
                        new_values[idx] = transform_fn(idx, old_val)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error transformando atributo {idx}: {e}")
                        continue
                
                # Creamos la nueva instancia
                new_instance = Instance(new_values, instance.schema, instance.weight)
                new_instances.append(new_instance)

            # Creamos la nueva bolsa
            new_bag = Bag(bag.bag_id, bag.label, new_instances)
            new_bags.append(new_bag)

        # Devolvemos el nuevo dataset
        return MIData(new_bags, original.name + "_transformed")
    
    def __repr__(self) -> str:
        """Representación técnica del scaler."""
        status = "fitted" if self._fitted else "not fitted"
        return f"<{self.__class__.__name__}({status})>"
    
    def __str__(self) -> str:
        """Representación legible del scaler."""
        if not self._fitted:
            return f"{self.__class__.__name__} (not fitted)"
        return f"{self.__class__.__name__} (fitted on {len(self.numeric_indices)} numeric features)"

class MinMaxScaler(BaseScaler):
    """
    Escala atributos numéricos a un rango específico [min, max].
    
    Fórmula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """Constructor del MinMaxScaler.
        
        Args:
            feature_range: Rango objetivo (min, max).
            
        Raises:
            ValueError: Si el rango no es válido.
        """

        super().__init__()

        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"Rango inválido: min debe ser < max")
        
        self._feature_range = feature_range
        self._data_min: Optional[np.ndarray] = None
        self._data_range: Optional[np.ndarray] = None
        
        logger.debug(f"MinMaxScaler creado con rango {feature_range}")

    @property
    def feature_range(self) -> Tuple[float, float]:
        """Rango objetivo de la transformación."""
        return self._feature_range
    
    @property
    def data_min(self) -> Optional[np.ndarray]:
        """Valores mínimos aprendidos (solo lectura)."""
        return self._data_min.copy() if self._data_min is not None else None
    
    @property
    def data_max(self) -> Optional[np.ndarray]:
        """Valores máximos aprendidos (solo lectura)."""
        if self._data_min is None or self._data_range is None:
            return None
        return self._data_min + self._data_range

    def fit(self, dataset: MIData) -> 'MinMaxScaler':
        """Aprende los valores mínimos y máximos del dataset.
        
        Args:
            dataset: Dataset de entrenamiento.  

        Returns:
            Self para encadenamiento.
            
        Raises:
            ValueError: Si el dataset está vacío.
        """

        logger.info(f"Entrenando MinMaxScaler en dataset '{dataset.name}'")

        # Extraemos el esquema e identificamos los atributos numéricos
        self._schema = self._extract_schema(dataset)
        self._numeric_indices = self._identify_numeric_indices(self._schema)  

        if not self._numeric_indices:
            logger.warning("No se encontraron atributos numéricos para escalar")
            self._fitted = True
            return self

        # Obtenemos los datos numericos
        numeric_mat = self._collect_numeric_data(dataset)

        # Calcular estadisiticos
        self._data_min = np.min(numeric_mat, axis=0)
        data_max = np.max(numeric_mat, axis=0)
        self._data_range = data_max - self._data_min

        # Capturamos para evitar problemas de tipo
        data_range = self._data_range if self._data_range is not None else np.array([])

        # Evitae división por cero
        zero_range_mask = data_range == 0
        if zero_range_mask.any():
            num_constant = zero_range_mask.sum()
            logger.warning(f"{num_constant} atributos son constantes (rango = 0)")
            data_range[zero_range_mask] = 1.0

        self._fitted = True
        # Usamos variables locales para evitar problemas de tipo en logging
        data_min = self._data_min if self._data_min is not None else np.array([])
        logger.info(f"MinMaxScaler entrenado: min={data_min[:3]}..., "
                   f"max={data_max[:3]}...")
        return self
    
    def transform(self, dataset: MIData, inplace: bool = False) -> MIData:
        """Aplica la transformación Min-Max al dataset.
        
        Args:
            dataset: Dataset a transformar.
            inplace: Si True, modifica el dataset original (NO RECOMENDADO).
            
        Returns:
            Dataset transformado.
            
        Raises:
            RuntimeError: Si no está entrenado.
            ValueError: Si el esquema no coincide.
        """
        self._validate_schema(dataset)
        
        if not self._numeric_indices:
            logger.warning("No hay atributos numéricos que transformar")
            return dataset

        if self._data_min is None or self._data_range is None:
            raise RuntimeError("MinMaxScaler no fue entrenado correctamente. Llama a fit() primero.")
        
        logger.info(f"Transformando dataset '{dataset.name}' con MinMaxScaler")
        
        min_range, max_range = self._feature_range
        range_diff = max_range - min_range
        
        # Capturamos los valores para evitar problemas de tipo
        data_min = self._data_min
        data_range = self._data_range
        
        def transform_value(idx_in_numeric: int, value: float) -> float:
            """Función de transformación Min-Max."""
            # Normalizar a [0, 1]
            normalized = (value - data_min[idx_in_numeric]) / data_range[idx_in_numeric]
            # Escalar al rango objetivo
            return normalized * range_diff + min_range
        
        # Inicialmente esto estará capado
        if inplace:
            logger.warning("Transformación inplace: modificando dataset original")
            # Modificar directamente (peligroso, solo si inplace = True)
            for bag in dataset:
                for instance in bag:
                    for i, idx in enumerate(self._numeric_indices):
                        try:
                            old_val = float(instance.get_value(idx))
                            instance.set_value(idx, transform_value(i, old_val))
                        except (ValueError, TypeError):
                            continue
            return dataset
        else:
            # Crear nuevo dataset
            return self._create_transformed_dataset(
                dataset,
                lambda idx, val: transform_value(
                    self._numeric_to_position[idx], val
                )
            )

    def inverse_transform(self, dataset: MIData, inplace: bool = False) -> MIData:
        """Revierte la transformación Min-Max.
        
        Args:
            dataset: Dataset en escala transformada.
            inplace: Si True, modifica el dataset original.

        Returns:
            Dataset en escala original.
        """
        self._validate_schema(dataset)
        
        if not self._numeric_indices:
            logger.warning("No hay atributos numéricos que revertir")
            return dataset

        if self._data_min is None or self._data_range is None:
            raise RuntimeError("MinMaxScaler no fue entrenado correctamente. Llama a fit() primero.")
        
        logger.info(f"Revirtiendo transformación Min-Max en '{dataset.name}'")
        
        min_range, max_range = self._feature_range
        range_diff = max_range - min_range

        # Capturamos los valores para evitar problemas de tipo
        data_min = self._data_min
        data_range = self._data_range

        def inverse_transform_value(idx_in_numeric: int, value: float) -> float:
            """Función inversa de Min-Max."""
            # De rango objetivo a [0, 1]
            normalized = (value - min_range) / range_diff
            # De [0, 1] a escala original
            return normalized * data_range[idx_in_numeric] + data_min[idx_in_numeric]
        
        # Inicialmente esto estará capado, peligroso modificar directamente el dataset
        if inplace:
            for bag in dataset:
                for instance in bag:
                    for i, idx in enumerate(self._numeric_indices):
                        try:
                            old_val = float(instance.get_value(idx))
                            instance.set_value(idx, inverse_transform_value(i, old_val))
                        except (ValueError, TypeError):
                            continue
            return dataset
        else:
            return self._create_transformed_dataset(
                dataset,
                lambda idx, val: inverse_transform_value(
                    self._numeric_to_position[idx], val
                )
            )        

class StandardScaler(BaseScaler):
    """
    Estandariza atributos eliminando la media y escalando a varianza unitaria.
    
    Formula: X_scaled = (X - mean) / std
    
    donde mean es la media y std es la desviacion estandar.
    
    """
    
    def __init__(self):
        """Constructor del StandardScaler."""
        super().__init__()
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        
        logger.debug("StandardScaler creado")
    
    @property
    def mean(self) -> Optional[np.ndarray]:
        """Media aprendida (solo lectura)."""
        return self._mean.copy() if self._mean is not None else None
    
    @property
    def std(self) -> Optional[np.ndarray]:
        """Desviación estándar aprendida (solo lectura)."""
        return self._std.copy() if self._std is not None else None
    
    def fit(self, dataset: MIData) -> 'StandardScaler':
        """Aprende la media y desviación estándar del dataset.
        
        Args:
            dataset: Dataset de entrenamiento.

        Returns:
            Self para el encadenamiento.
            
        Raises:
            ValueError: Si el dataset está vacío.
        """
        logger.info(f"Entrenando StandardScaler en dataset '{dataset.name}'")
        
        # Extraemos el esquema e identificamos los atributos numéricos
        self._schema = self._extract_schema(dataset)
        self._numeric_indices = self._identify_numeric_indices(self._schema)
        
        if not self._numeric_indices:
            logger.warning("No se encontraron atributos numéricos para escalar")
            self._fitted = True
            return self
        
        # Recolectar datos numéricos
        numeric_matrix = self._collect_numeric_data(dataset)
        
        # Calcular estadísticos
        self._mean = np.mean(numeric_matrix, axis=0)
        self._std = np.std(numeric_matrix, axis=0)
        
        # Capturamos para evitar problemas de tipo
        std_val = self._std if self._std is not None else np.array([])
        
        # Evitar división por cero
        zero_std_mask = std_val == 0
        if zero_std_mask.any():
            num_constant = zero_std_mask.sum()
            logger.warning(f"{num_constant} atributos tienen desviación estándar = 0")
            self._std[zero_std_mask] = 1.0
        
        self._fitted = True
        # Usamos variables locales para evitar problemas de tipo en logging
        mean = self._mean if self._mean is not None else np.array([])
        logger.info(f"StandardScaler entrenado: mean={mean[:3]}..., "
                   f"std={std_val[:3]}...")
        
        return self
    
    def transform(self, dataset: MIData, inplace: bool = False) -> MIData:
        """Aplica la estandarización (Z-score) al dataset.
        
        Args:
            dataset: Dataset en escala transformada.
            inplace: Si True, modifica el dataset original.

        Returns:
            Dataset en escala original.
            
        Raises:
            RuntimeError: Si no está entrenado.
            ValueError: Si el esquema no coincide.
        """
        self._validate_schema(dataset)
        
        if not self._numeric_indices:
            logger.warning("No hay atributos numéricos que transformar")
            return dataset

        if self._mean is None or self._std is None:
            raise RuntimeError("StandardScaler no fue entrenado correctamente. Llama a fit() primero.")
        
        logger.info(f"Transformando dataset '{dataset.name}' con StandardScaler")
        
        # Capturamos los valores para evitar problemas de tipo
        mean = self._mean
        std = self._std
        
        def transform_value(idx_in_numeric: int, value: float) -> float:
            """Función de estandarización Z-score."""
            return (value - mean[idx_in_numeric]) / std[idx_in_numeric]
        
        if inplace:
            logger.warning("Transformación inplace: modificando dataset original")
            for bag in dataset:
                for instance in bag:
                    for i, idx in enumerate(self._numeric_indices):
                        try:
                            old_val = float(instance.get_value(idx))
                            instance.set_value(idx, transform_value(i, old_val))
                        except (ValueError, TypeError):
                            continue
            return dataset
        else:
            return self._create_transformed_dataset(
                dataset,
                lambda idx, val: transform_value(
                    self._numeric_to_position[idx], val
                )
            )
    
    def inverse_transform(self, dataset: MIData, inplace: bool = False) -> MIData:
        """Revierte la estandarización.
        
        Args:
            dataset: Dataset en escala estandarizado.
            inplace: Si True, modifica el dataset original.

        Returns:
            Dataset sin estandarizar original.
        """
        self._validate_schema(dataset)
        
        if not self._numeric_indices:
            logger.warning("No hay atributos numéricos que revertir")
            return dataset

        if self._mean is None or self._std is None:
            raise RuntimeError("StandardScaler no fue entrenado correctamente. Llama a fit() primero.")
        
        logger.info(f"Revirtiendo estandarización en '{dataset.name}'")
        
        # Capturamos los valores para evitar problemas de tipo
        mean = self._mean
        std = self._std
        
        def inverse_transform_value(idx_in_numeric: int, value: float) -> float:
            """Función inversa de Z-score."""
            return value * std[idx_in_numeric] + mean[idx_in_numeric]
        
        if inplace:
            for bag in dataset:
                for instance in bag:
                    for i, idx in enumerate(self._numeric_indices):
                        try:
                            old_val = float(instance.get_value(idx))
                            instance.set_value(idx, inverse_transform_value(i, old_val))
                        except (ValueError, TypeError):
                            continue
            return dataset
        else:
            return self._create_transformed_dataset(
                dataset,
                lambda idx, val: inverse_transform_value(
                    self._numeric_to_position[idx], val
                )
            )