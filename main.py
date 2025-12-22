from scipy.io import arff
import pandas as pd
import numpy as np
from mil import Bag

# Cargar el archivo ARFF
data, meta = arff.loadarff('data/musk2.arff')
df = pd.DataFrame(data)

dataset_mil = []

for index, row in df.iterrows():

    # A. Extraer ID (La columna se llama 'molecule_name' en este dataset)
    # Decodificamos porque a veces viene como bytes (b'MUSK-188')

    b_id = row['molecule_name'].decode('utf-8') if isinstance(row['molecule_name'], bytes) else row['molecule_name']
    
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