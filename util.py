from scipy.io import arff
import pandas as pd
import numpy as np
from mil import Bag

def load_arff_dataset(dataset_path):

    # Cargar el archivo ARFF
    data, meta = arff.loadarff('data/musk1.arff')
    df = pd.DataFrame(data)

    id_column_name = df.columns[0]
    dataset_mil = []

    for index, row in df.iterrows():

        # A. Extraer ID 

        b_id = row[id_column_name].decode('utf-8') if isinstance(row[id_column_name], bytes) else row[id_column_name]

        
        # B. Extraer Etiqueta (La columna se llama 'class' en este dataset)

        label = row['class'].decode('utf-8') if isinstance(row['class'], bytes) else row['class']
        
        # C. Extraer Instancias (La columna 'bag' ya tiene los datos anidados)
        # Los datos vienen como una estructura compleja, los convertimos a una matriz numérica limpia

        bag_data = row['bag'] 
        
        # Convertir la estructura interna a una lista de listas y luego a numpy array

        instances_matrix = np.array(bag_data.tolist(), dtype=float)
        
        # Crear el objeto y guardar

        new_bag = Bag(b_id, label)
        new_bag.instances = instances_matrix
        dataset_mil.append(new_bag)

    print(f"Se cargaron {len(dataset_mil)} bolsas.")
    print("Ejemplo de la primera bolsa:")
    print(dataset_mil[0])
    print("Dimensiones de sus instancias:", dataset_mil[0].instances.shape)