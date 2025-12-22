import numpy as np

class Bag:
    def __init__(self, bag_id, label=None):
        """
        Constructor de la Bolsa.
        :param bag_id: Identificador único de la bolsa (string o int).
        :param label: Etiqueta de la bolsa (clase), si se conoce.
        """
        self.bag_id = bag_id
        self.label = label
        self.instances = [] # Lista temporal para ir añadiendo filas
    
    def add_instance(self, instance_features):
        """
        Añade una nueva instancia (vector de características) a la bolsa.
        :param instance_features: Lista o array con los valores numéricos.
        """
        self.instances.append(instance_features)

    def to_numpy(self):
        """
        Convierte la lista de instancias a una matriz NumPy para eficiencia.
        Debe llamarse una vez se han añadido todas las instancias.
        """
        self.instances = np.array(self.instances, dtype=float)

    def __repr__(self):
        # Esto ayuda a visualizar el objeto al imprimirlo
        num_inst = len(self.instances) if isinstance(self.instances, list) else self.instances.shape[0]
        return f"<Bag ID: {self.bag_id} | Label: {self.label} | Instances: {num_inst}>"