from miclustering.data.attribute import Attribute
from typing import List, Any

class Instance:
    """Clase Instance.

    Nota: cada instancia debe conocer el esquema (lista de Attribute) para validar tipos y procesar correctamente.

    Attributes:
        values: lista o array con los valores de la instancia.
        weight: peso (por defecto 1.0).
    """
    __slots__ = ('_values', '_schema', '_weight', '_bag_ref')

    def __init__(self, values: List[Any], schema: List['Attribute'], weight: float = 1.0):
        """Constructor de la Instancia.

        Args:
            values: Lista o array con los valores de la instancia.
            schema: Esquema (lista de Attribute) para validar tipos.
            weight: Peso de la instancia (float), por defecto 1.0.
        """
        self._values = values
        self._schema = schema
        self._weight = weight
        self._bag_ref = None

    def __getstate__(self):
        return {
            '_values': self._values,
            '_schema': self._schema,
            '_weight': self._weight,
        }

    def __setstate__(self, state):
        self._values = state['_values']
        self._schema = state['_schema']
        self._weight = state['_weight']
        self._bag_ref = None

    def __eq__(self, other):
        if not isinstance(other, Instance):
            return False
        return self._values == other._values and self._schema == other._schema and self._weight == other._weight

    def __repr__(self):
        return f"<Instance values={self._values} weight={self._weight}>"

    def __str__(self):
        return f"Instance({self._values}, weight={self._weight})"

    def get_value(self, index: int) -> Any:
        """Devuelve el valor en la posición indicada.

        Args:
            index: Índice del atributo.

        Returns:
            Valor del atributo.
        """
        return self.values[index]
    
    def set_value(self, index: int, value: Any):
        """Asigna un valor en la posición indicada, según el esquema.

        Args:
            index: Índice del atributo.
            value: Nuevo valor a asignar.
        """
        if not (0 <= index < len(self._values)):
            raise IndexError("Índice fuera de rango.")

        attribute_def = self.schema[index]

        if not self._validate_type(value, attribute_def):
            raise TypeError(
                f"El valor '{value}' no es válido para el atributo '{attribute_def.name}' "
                f"de tipo '{attribute_def.type}'."
            )

        self.values[index] = value

        # Invalidar la caché de la bolsa contenedora si existe (tras mutación exitosa)
        ref = self._bag_ref
        bag = ref() if ref is not None else None
        if bag is not None:
            bag.invalidate_cache()
    
    def _validate_type(self, value: Any, attribute: 'Attribute') -> bool:
        """Método interno auxiliar para validar un valor contra un Atributo.

        Returns:
            True si es válido, False si no.
        """
        attr_type = attribute.type

        if attr_type == 'integer':
            return isinstance(value, int)
        
        elif attr_type == 'real':
            return isinstance(value, (float, int))
        
        elif attr_type == 'string':
            return isinstance(value, str)
        
        elif attr_type == 'nominal':
            if not isinstance(value, str):
                return False
            return True if attribute.values is None else value in attribute.values
        
        return True
    
    def num_attributes(self) -> int:
        """Devuelve el número de atributos en la instancia.

        Returns:
            Número de atributos (int).
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