import random
import numpy as np

def generate_simple_arff(filename="simple_dummy.arff"):
    # Configuración
    n_features = 2  # Solo 2 dimensiones (x, y) para que sea leíble
    n_bags_pos = 15 # 15 bolsas positivas
    n_bags_neg = 15 # 15 bolsas negativas
    n_bags_noise = 5 # 5 bolsas de ruido puro
    
    total_bags = n_bags_pos + n_bags_neg + n_bags_noise
    
    # Generar nombres de bolsas: Bolsa_0, Bolsa_1, ...
    bag_names = [f"Bolsa_{i}" for i in range(total_bags)]
    bag_names_str = ",".join(bag_names)

    # Crear Header ARFF
    header = f"""% DATASET SIMPLE PARA DUMMIES (TEST MIL)
% Estructura idéntica a Musk pero con valores sencillos.
@relation simple_mil_test

@attribute bag_id {{{bag_names_str}}}
@attribute bag relational
  @attribute x numeric
  @attribute y numeric
@end bag
@attribute class {{0,1}}

@data
"""
    data_lines = []

    def create_instances(center_x, center_y, n_instances, noise=False):
        """Crea strings de instancias: '1.2,3.4\\n1.1,3.5'"""
        lines = []
        for _ in range(n_instances):
            if noise:
                # Ruido: valores dispersos entre -50 y 50
                val_x = round(random.uniform(-50, 50), 2)
                val_y = round(random.uniform(-50, 50), 2)
            else:
                # Cluster: valores cerca del centro con pequeña variación (+/- 2)
                val_x = round(random.gauss(center_x, 1.0), 2)
                val_y = round(random.gauss(center_y, 1.0), 2)
            
            lines.append(f"{val_x},{val_y}")
        return "\\n".join(lines)

    bag_idx = 0

    # 1. Generar CLASE 1 (Zona Positiva ~ 10,10)
    for _ in range(n_bags_pos):
        name = bag_names[bag_idx]
        # Crear entre 2 y 4 puntos por bolsa
        instances_str = create_instances(10, 10, random.randint(2, 4))
        # Formato: Nombre, "instancias", clase
        line = f'{name},"{instances_str}",1'
        data_lines.append(line)
        bag_idx += 1

    # 2. Generar CLASE 0 (Zona Negativa ~ -10,-10)
    for _ in range(n_bags_neg):
        name = bag_names[bag_idx]
        instances_str = create_instances(-10, -10, random.randint(2, 4))
        line = f'{name},"{instances_str}",0'
        data_lines.append(line)
        bag_idx += 1

    # 3. Generar RUIDO (Clase aleatoria, valores locos)
    for _ in range(n_bags_noise):
        name = bag_names[bag_idx]
        label = random.choice(["0", "1"])
        instances_str = create_instances(0, 0, random.randint(1, 3), noise=True)
        line = f'{name},"{instances_str}",{label}'
        data_lines.append(line)
        bag_idx += 1

    # Escribir archivo
    with open(filename, "w") as f:
        f.write(header)
        f.write("\n".join(data_lines))
    
    print(f"Generado '{filename}' con éxito.")
    print(f"- Total Bolsas: {total_bags}")
    print(f"- Dimensiones: {n_features} (x, y)")
    print("- Clusters: (10,10) y (-10,-10)")

if __name__ == "__main__":
    generate_simple_arff()