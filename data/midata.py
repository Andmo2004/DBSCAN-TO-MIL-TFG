from typing import List, Tuple
from data.bag import Bag
import random


class MIData:
    '''
        - Clase MIData: 
            atributos: 
                bags: lista de objetos Bag, 
                name: nombre del dataset. 
            Métodos: 
                get_bag(i): devuelve la bolsa i, 
                get_num_bags(): número total de bolsas, 
                split_data(percentage_train, seed): divide el dataset en entrenamiento y prueba 
                    (podemos implementar diferentes opciones).
    '''
    def __init__(self, bags: List['Bag'], name: str):
        """
        Constructor del dataset multi-instancia.        
        :param bags: lista de objetos Bag.
        :param name: Nombre del dataset.
        """
        self._bags = bags
        self._name = name

    def __iter__(self):
        """
        Permite iterar sobre las bolsas: for bag in dataset.
        """
        return iter(self._bags)

    def __contains__(self, bag):
        """
        Permite usar 'in': bag in dataset.
        """
        return bag in self._bags

    def __eq__(self, other):
        if not isinstance(other, MIData):
            return False
        return self._bags == other._bags and self._name == other._name

    def __repr__(self):
        return f"<MIData name={self._name} bags={len(self._bags)} >"

    def __str__(self):
        return f"MIData '{self._name}' ({len(self._bags)} bags)"

    def get_bag(self, i: int) -> 'Bag':
        """
        Devuelve la bolsa i-ésima del dataset.
        :param i: Índice de la bolsa.
        :return: Objeto Bag.
        """
        if 0 <= i < len(self.bags):
            return self.bags[i]
        raise IndexError(f"Índice {i} fuera de rango para el dataset {self.name}.")
    
    def get_num_bags(self) -> int:
        """
        Devuelve el número total de bolsas en el dataset.
        :return: Número de bolsas (int).
        """
        return len(self.bags)
    
    def split_data(self, percentage_train: float, seed: int = 1234) -> Tuple['MIData', 'MIData']:
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.
        :param percentage_train: Porcentaje de bolsas para el conjunto de entrenamiento (0-100).
        :param seed: Semilla para la aleatorización.
        :return: Tupla (MIData_train, MIData_test).
        """
        random.seed(seed)
        
        bags_copy = self.bags[:]
        random.shuffle(bags_copy)
        
        split_index = int(len(bags_copy) * (percentage_train / 100.0))
        
        train_bags = bags_copy[:split_index]
        test_bags = bags_copy[split_index:]
        
        mi_data_train = MIData(train_bags, self.name + "_train")
        mi_data_test = MIData(test_bags, self.name + "_test")
        
        return mi_data_train, mi_data_test
    
    @property
    def bags(self) -> List['Bag']:
        return self._bags.copy()

    @property
    def name(self) -> str:
        return self._name