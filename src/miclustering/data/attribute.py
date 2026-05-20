import numpy as np
from typing import List, Tuple, Optional, Any

class Attribute:
    """Clase Attribute.
    
    Propósito: definir el esquema de cada columna del dataset.
    
    Attributes:
        name: nombre del atributo.
        type: tipo de dato (cadena, nominal, entero, real, fecha).
        values: lista de valores posibles (solo si es nominal).
        data_format: formato (por ejemplo, para fechas).
        val_range: intervalo de valores enteros o reales.
    """
    __slots__ = ('_name', '_type', '_values', '_data_format', '_val_range')
    def __init__(self, 
                 name: str, 
                 attr_type: str, 
                 values: Optional[List[Any]] = None, 
                 data_format: Optional[str] = None, 
                 val_range: Optional[Tuple[float, float]] = None):
        """Constructor del Atributo.

        Args:
            name: Nombre del atributo (string).
            attr_type: Tipo de dato (string): 'string', 'nominal', 'integer', 'real', 'date'.
            values: Lista de valores posibles (solo si es nominal).
            data_format: Formato específico (por ejemplo, para fechas).
            val_range: Tupla (min, max) si se quiere definir un rango.
        """
        self._name = name
        self._type = attr_type
        self._values = values
        self._data_format = data_format
        self._val_range = val_range

    def __repr__(self):
        details = ""
        if self.type == 'nominal':
            details = f" | Values: {self.values}"
        elif self.val_range:
            details = f" | Range: {self.val_range}"
            
        return f"<Attribute '{self.name}' ({self.type}){details}>"
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def values(self) -> Optional[List[Any]]:
        return self._values

    @property
    def data_format(self) -> Optional[str]:
        return self._data_format

    @property
    def val_range(self) -> Optional[Tuple[float, float]]:
        return self._val_range
