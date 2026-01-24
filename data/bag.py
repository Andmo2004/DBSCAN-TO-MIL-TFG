from data.instance import Instance
from typing import Any, Optional, List
import numpy as np


class Bag:
    '''
    - Clase Bag: 
        atributos: 
            bag_id: identificador único de la bolsa, instances: lista de objetos Instance. 
            label o class: etiqueta asociada a la bolsa. 
        Métodos: 
            get_instance(i): devuelve la instancia i, 
            get_num_instances(): número de instancias en la bolsa, 
            add_instance(instance): añade una instancia, 
            as_matrix(): devuelve las instancias como matriz NumPy (n_instancias * n_atributos).
    '''
    def __init__(self, bag_id: Any, label: Any, instances: Optional[List['Instance']] = None):
        self._bag_id = bag_id
        self._label = label
        self._instances = instances if instances is not None else []
        
        # Validación
        if self._instances and not all(isinstance(i, Instance) for i in self._instances):
            raise TypeError("Todos los elementos deben ser instancias de Instance")

    def __iter__(self):
        """
        Permite iterar sobre las instancias: for inst in bag.
        """
        return iter(self._instances)

    def __contains__(self, instance):
        """
        Permite usar 'in': instance in bag.
        """
        return instance in self._instances

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
        """
        Devuelve la instancia i-ésima de la bolsa.
        :param i: Índice de la instancia.
        :return: Objeto Instance.
        """
        if 0 <= i < len(self._instances):
            return self._instances[i]
        raise IndexError(f"Índice {i} fuera de rango para la bolsa {self._bag_id}.")
    
    def get_num_instances(self) -> int:
        """
        Devuelve el número de instancias en la bolsa.
        :return: Número de instancias (int).
        """
        return len(self._instances)
    
    def add_instance(self, instance: 'Instance'):
        """
        Añade una instancia a la bolsa.
        :param instance: Objeto Instance a añadir.
        """
        self._instances.append(instance)

    def as_matrix(self) -> np.ndarray:
        """
        Devuelve las instancias como una matriz NumPy. (n_instancias * n_atributos)
        :return: Matriz NumPy con las instancias.
        """
        if not self._instances:
            return np.array([]).reshape(0, len(self._instances[0].values) if self._instances else 0)
        matrix = []
        for instance in self._instances:
            matrix.append(instance.values)
        return np.array(matrix)
    
    
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