from scipy.io import arff
import pandas as pd
from typing import List

from data.attribute import Attribute
from data.instance import Instance
from data.bag import Bag
from data.midata import MIData

# Al inicio del archivo
DEFAULT_BAG_COLUMN = 'bag'
DEFAULT_CLASS_COLUMN = 'class'

def extract_bag_schema(dataset_path: str, bag_attr_name: str = "bag") -> List[Attribute]:
    """
    Lee el archivo ARFF como texto para extraer los nombres reales de los atributos
    que están dentro de la estructura relacional (entre @attribute bag relational y @end bag).
    """
    attributes = []
    inside_bag_def = False
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Ignorar comentarios y líneas vacías
            if not line or line.startswith('%'):
                continue
            
            # Detectar inicio de la bolsa relacional
            # Buscar algo como: @attribute bag relational
            if line.lower().startswith(f"@attribute {bag_attr_name} relational"):
                inside_bag_def = True
                continue
            
            # Detectar fin de la bolsa
            # Busca algo como: @end bag
            if line.lower().startswith(f"@end {bag_attr_name}"):
                break # Ya tenemos lo que queríamos, dejamos de leer
            
            # Si estamos dentro del bloque, capturamos los atributos
            if inside_bag_def and line.lower().startswith("@attribute"):
                # Ejemplo de linea: @attribute f1 numeric
                parts = line.split()
                if len(parts) >= 3:
                    attr_name = parts[1] # nombre de la carcteristica

                    # Creamos el objeto Attribute (asumimos real/numeric para instancias)
                    attributes.append(Attribute(attr_name, "real"))
                    
    if not attributes:
        print(f"Advertencia: No se encontraron atributos dentro de '{bag_attr_name}'.")
        
    return attributes

def load_arff_dataset(dataset_path: str, dataset_name: str = "dataset") -> MIData:
    """
    Carga un dataset MIL asumiendo estructura estándar:
    Col 0: ID (variable), Col 'bag': Relacional, Col 'class': Etiqueta.
    """

    print(f"Cargando dataset: {dataset_path} ...")

    # 1. Cargar datos crudos
    try:
        data, meta = arff.loadarff(dataset_path)
    except Exception as e:
        raise ValueError(f"Error cargando ARFF: {e}")

    df = pd.DataFrame(data)
    
    # El ID es siempre la primera columna (índice 0), tenga el nombre que tenga
    id_col_name = df.columns[0] 
    
    # Validación básica por si acaso
    if DEFAULT_BAG_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna '{DEFAULT_BAG_COLUMN}' en el dataset.")

    # Obtener el esquema interno (nombres reales de f1, f2...)
    instance_schema = extract_bag_schema(dataset_path)
    print(f"Esquema interno detectado: {len(instance_schema)} atributos.")

    # Construir lista de Bolsas
    bags_list = []

    for index, row in df.iterrows():
        
        # Extraemos ID (primera columna)
        
        b_id = row[id_col_name]
        _decode_if_bytes(b_id)

        # Extraemos Etiqueta (columna 'class')
        label = row[DEFAULT_CLASS_COLUMN]
        _decode_if_bytes(label)
        # Extraemos Instancias (columna 'bag')
        # raw_bag_data es un numpy array estructurado
        raw_bag_data = row[DEFAULT_BAG_COLUMN]
        
        instances_in_bag = []
        for raw_inst in raw_bag_data:
            # Convertimos la fila numpy a lista estándar de python
            values = list(raw_inst)
            
            # Creamos el objeto Instance pasándole el esquema con nombres reales
            new_inst = Instance(values, instance_schema)
            instances_in_bag.append(new_inst)

        # Crear la Bolsa
        new_bag = Bag(bag_id=b_id, label=label, instances=instances_in_bag)
        bags_list.append(new_bag)


    def _decode_if_bytes(value):
        return value.decode('utf-8') if isinstance(value, bytes) else value
    
    print(f"-> Carga finalizada: {len(bags_list)} bolsas procesadas.")
    return MIData(bags_list, dataset_name)

