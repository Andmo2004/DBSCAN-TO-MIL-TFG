import numpy as np
from typing import List, Tuple, Optional, Any
'''
- Clase Attribute: 
    atributos: 
        name: nombre del atributo, 
        type: tipo de dato (cadena, nominal, entero, real, fecha). 
        Según el tipo: 
            values: lista de valores posibles (solo si es nominal), 
            data_format: formato (por ejemplo, para fechas), intervalo de valores enteros.
    Propósito: definir el esquema de cada columna del dataset.
'''
class Attribute:
    def __init__(self, 
                 name: str, 
                 attr_type: str, 
                 values: Optional[List[Any]] = None, 
                 data_format: Optional[str] = None, 
                 val_range: Optional[Tuple[float, float]] = None):
        """
        Constructor del Atributo.
        :param name: Nombre del atributo (string).
        :param attr_type: Tipo de dato (string): 'string', 'nominal', 'integer', 'real', 'date'.
        :param values: Lista de valores posibles (solo si es nominal).
        :param data_format: Formato específico (por ejemplo, para fechas).
        :param int_range: Tupla (min, max) si el tipo es entero y se quiere definir un rango.
        """
        self.name = name
        self.type = attr_type
        self.values = values
        self.data_format = data_format
        self.val_range = val_range

    def __repr__(self):
        details = ""
        if self.type == 'nominal':
            details = f" | Values: {self.values}"
        elif self.val_range:
            details = f" | Range: {self.val_range}"
            
        return f"<Attribute '{self.name}' ({self.type}){details}>"
    

