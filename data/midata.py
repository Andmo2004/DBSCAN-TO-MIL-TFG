from typing import List, Tuple, Optional
from data.bag import Bag
import random
import logging

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_arff(cls, 
                  file_path: str, 
                  dataset_name: Optional[str] = None,
                  bag_column: str = 'bag',
                  class_column: str = 'class') -> 'MIData':
        """
        Carga un dataset MIL desde un archivo ARFF.
        
        Este método de clase (factory method) permite crear una instancia de MIData
        directamente desde un archivo ARFF sin necesidad de instanciar manualmente
        el loader.
        
        Args:
            file_path: Ruta al archivo ARFF.
            dataset_name: Nombre del dataset (si es None, usa el nombre del archivo).
            bag_column: Nombre de la columna con estructura relacional (default: 'bag').
            class_column: Nombre de la columna con etiquetas (default: 'class').
            
        :returns: Objeto MIData con el dataset cargado.
            
        :raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el formato es inválido o faltan columnas requeridas.
            
        """
        # Importación local para evitar dependencias circulares
        from data.arff_reader import ArffToMIData
        
        logger.info(f"Cargando dataset desde ARFF: {file_path}")
        
        # Crear el loader con la configuración especificada
        loader = ArffToMIData(
            bag_column=bag_column,
            class_column=class_column
        )
        
        # Cargar y retornar el dataset
        return loader.load(file_path, dataset_name)
    
    @property
    def bags(self) -> List['Bag']:
        return self._bags.copy()

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
            """
            Permite usar len(dataset) para obtener el número de bolsas.
            """
            return len(self._bags)

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

    def __getitem__(self, index: int) -> 'Bag':
        """Permite acceso tipo array: dataset[i]"""
        return self.get_bag(index)

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
    
    def get_labels(self) -> List:
        """
        Obtiene todas las etiquetas del dataset.
        :return: Lista con las etiquetas de todas las bolsas.
        """
        return [bag.label for bag in self._bags]

    def get_positive_bags(self) -> List['Bag']:
        """
        Obtiene todas las bolsas con etiqueta positiva.
        :return: Lista de bolsas con etiqueta positiva (típicamente '1' o 'positive').
        """
        positive_labels = {'1', 1, 'positive', 'pos', True}
        return [bag for bag in self._bags if bag.label in positive_labels]

    def get_negative_bags(self) -> List['Bag']:
        """
        Obtiene todas las bolsas con etiqueta negativa.
        :return: Lista de bolsas con etiqueta negativa (típicamente '0' o 'negative').
        """
        negative_labels = {'0', 0, 'negative', 'neg', False}
        return [bag for bag in self._bags if bag.label in negative_labels]