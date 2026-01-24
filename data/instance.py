from data.attribute import Attribute
from typing import List, Any

class Instance:
    '''
    - Clase Instance: 
        atributos: 
            values: lista o array con los valores de la instancia, 
            weight: peso (por defecto 1.0). 
        Métodos: 
            get_value(index): devuelve el valor en la posición indicada, 
            set_value(index, value): asigna un valor validando el tipo según el esquema, 
            num_attributes(): devuelve el número de atributos.
        Nota: cada instancia debe conocer el esquema (lista de Attribute) para validar tipos y procesar correctamente.
    '''

    def __init__(self, values: List[Any], schema: List['Attribute'], weight: float = 1.0):
        """
        Constructor de la Instancia.
        :param values: Lista o array con los valores de la instancia.
        :param schema: Esquema (lista de Attribute) para validar tipos.
        :param weight: Peso de la instancia (float), por defecto 1.0.
        """
        self._values = values
        self._schema = schema
        self._weight = weight

    def __eq__(self, other):
        if not isinstance(other, Instance):
            return False
        return self._values == other._values and self._schema == other._schema and self._weight == other._weight

    def __repr__(self):
        return f"<Instance values={self._values} weight={self._weight}>"

    def __str__(self):
        return f"Instance({self._values}, weight={self._weight})"

    def get_value(self, index: int) -> Any:
        """
        Devuelve el valor en la posición indicada.
        :param index: Índice del atributo.
        :return: Valor del atributo.
        """
        return self.values[index]
    
    def set_value(self, index: int, value: Any):
        """
        Asigna un valor en la posición indicada, según el esquema.
        :param index: Índice del atributo.
        :param value: Nuevo valor a asignar.
        """
        if index < 0 or index >= len(self.values):
            raise IndexError("Índice fuera de rango.")

        attribute_def = self.schema[index]

        if not self._validate_type(value, attribute_def):
            raise TypeError(
                f"El valor '{value}' no es válido para el atributo '{attribute_def.name}' "
                f"de tipo '{attribute_def.type}'."
            )

        self.values[index] = value
    
    def _validate_type(self, value: Any, attribute: 'Attribute') -> bool:
        """
        Método interno auxiliar para validar un valor contra un Atributo.
        Devuelve True si es válido, False si no.
        """
        attr_type = attribute.type

        if attr_type == 'integer':
            return isinstance(value, int)
        
        elif attr_type == 'real':
            return isinstance(value, (float, int))
        
        elif attr_type == 'string':
            return isinstance(value, str)
        
        elif attr_type == 'nominal':
            return isinstance(value, str) and (value in attribute.values)
        
        return True
    
    def num_attributes(self) -> int:
        """
        Devuelve el número de atributos en la instancia.
        :return: Número de atributos (int).
        """
        return len(self.values)
    
    @property
    def values(self) -> List[Any]:
        return self._values

    @property
    def schema(self) -> List['Attribute']:
        return self._schema

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = value