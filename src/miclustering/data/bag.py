from miclustering.data.instance import Instance
from typing import Any, Optional, List
import numpy as np
import weakref


class Bag:
    """Clase Bag.

    Attributes:
        bag_id: identificador único de la bolsa.
        instances: lista de objetos Instance.
        label: etiqueta asociada a la bolsa.
    """
    _instances: List['Instance']
    _matrix_cache: Optional[np.ndarray]

    def __init__(self, bag_id: Any, label: Any, instances: Optional[List['Instance']] = None):
        self._bag_id = bag_id
        self._label = label
        self._instances = instances if instances is not None else []
        self._matrix_cache = None

        if self._instances and not all(isinstance(i, Instance) for i in self._instances):
            raise TypeError("Todos los elementos deben ser instancias de Instance")

        for inst in self._instances:
            inst._bag_ref = weakref.ref(self)

    def __getstate__(self):
        return {
            '_bag_id': self._bag_id,
            '_label': self._label,
            '_instances': self._instances,
            '_matrix_cache': self._matrix_cache,
        }

    def __setstate__(self, state):
        self._bag_id = state['_bag_id']
        self._label = state['_label']
        self._instances = state['_instances']
        self._matrix_cache = state.get('_matrix_cache')
        for inst in self._instances:
            inst._bag_ref = weakref.ref(self)

    def invalidate_cache(self) -> None:
        """Invalida la caché interna de la matriz NumPy."""
        self._matrix_cache = None

    def __iter__(self):
        """Permite iterar sobre las instancias: for inst in bag."""
        return iter(self._instances)
    
    def __len__(self) -> int:
        """Permite usar len(bag) para obtener el número de instancias."""
        return len(self._instances)

    def __contains__(self, instance):
        """Permite usar 'in': instance in bag."""
        return instance in self._instances

    def __getitem__(self, index: int) -> 'Instance':
        """Permite acceder a las instancias con corchetes: bag[0]."""
        return self.get_instance(index)

    def __eq__(self, other):
        if not isinstance(other, Bag):
            return False
        return (self._bag_id == other._bag_id and
                self._label == other._label and
                self._instances == other._instances)

    def __str__(self):
        return f"Bag '{self._bag_id}' ({len(self._instances)} instances)"

    def __repr__(self):
        num_inst = len(self._instances) if isinstance(self._instances, list) else self._instances.shape[0]
        return f"<Bag ID: {self._bag_id} | Label: {self._label} | Instances: {num_inst}>"

    def get_instance(self, i:int) -> 'Instance':
        """Devuelve la instancia i-ésima de la bolsa.

        Args:
            i: Índice de la instancia.

        Returns:
            Objeto Instance.
        """
        if 0 <= i < len(self._instances):
            return self._instances[i]
        raise IndexError(f"Índice {i} fuera de rango para la bolsa {self._bag_id}.")
    
    def get_num_instances(self) -> int:
        """Devuelve el número de instancias en la bolsa.

        Returns:
            Número de instancias (int).
        """
        return len(self._instances)
    
    def add_instance(self, instance: 'Instance'):
        """Añade una instancia a la bolsa e invalida la caché de matriz.

        Args:
            instance: Objeto Instance a añadir.
        """
        if not isinstance(instance, Instance):
            raise TypeError("El elemento debe ser una instancia de Instance")
        instance._bag_ref = weakref.ref(self)
        self._instances.append(instance)
        self.invalidate_cache()

    def as_matrix(self) -> np.ndarray:
        """Devuelve las instancias como una matriz NumPy. (n_instancias * n_atributos)
        Usa caché interna para evitar conversiones repetidas de listas de Python a NumPy.

        Returns:
            Matriz NumPy con las instancias.
        """
        if self._matrix_cache is not None:
            return self._matrix_cache

        if not self._instances:
            self._matrix_cache = np.empty((0,), dtype=np.float64)
        else:
            matrix = [instance.values for instance in self._instances]
            self._matrix_cache = np.array(matrix, dtype=np.float64)
        return self._matrix_cache
    
    @property
    def bag_id(self) -> Any:
        return self._bag_id

    @property
    def label(self) -> Any:
        return self._label

    @label.setter
    def label(self, value: Any):
        self._label = value

    @property
    def instances(self) -> List['Instance']:
        return self._instances.copy()