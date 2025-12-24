from data.instance import Instance
from typing import Any, Optional, List
import numpy as np

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

class Bag:
    def __init__(self, bag_id: Any, label: Any, instances: Optional[List['Instance']] = None):
        """
        Constructor de la Bolsa.
        :param bag_id: Identificador único de la bolsa (string o int).
        :param label: Etiqueta de la bolsa (clase), si se conoce.
        """
        self.bag_id = bag_id
        self.label = label
        self.instances = instances if instances is not None else []

    def get_instance(self, i:int) -> 'Instance':
        """
        Devuelve la instancia i-ésima de la bolsa.
        :param i: Índice de la instancia.
        :return: Objeto Instance.
        """
        if 0 <= i < len(self.instances):
            return self.instances[i]
        raise IndexError(f"Índice {i} fuera de rango para la bolsa {self.bag_id}.")
    
    def get_num_instances(self) -> int:
        """
        Devuelve el número de instancias en la bolsa.
        :return: Número de instancias (int).
        """
        return len(self.instances)
    
    def add_instance(self, instance: 'Instance'):
        """
        Añade una instancia a la bolsa.
        :param instance: Objeto Instance a añadir.
        """
        self.instances.append(instance)

    def as_matrix(self) -> np.ndarray:
        """
        Devuelve las instancias como una matriz NumPy. (n_instancias * n_atributos)
        :return: Matriz NumPy con las instancias.
        """
        if not self.instances:
            return np.array([])
        
        matrix = []
        for instance in self.instances:
            matrix.append(instance.values)

        return np.array(matrix)
    
    def __repr__(self):
        num_inst = len(self.instances) if isinstance(self.instances, list) else self.instances.shape[0]
        return f"<Bag ID: {self.bag_id} | Label: {self.label} | Instances: {num_inst}>"