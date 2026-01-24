from scipy.io import arff
import pandas as pd
from typing import List, Optional
from pathlib import Path
import logging

from data.attribute import Attribute
from data.instance import Instance
from data.bag import Bag
from data.midata import MIData

logger = logging.getLogger(__name__)


class ArffToMIData:
    """
    Transformador que convierte archivos ARFF a objetos MIData.
    
    Esta clase maneja la carga y conversión de datasets Multi-Instance Learning
    desde el formato ARFF estándar.
    
    Attributes:
        DEFAULT_BAG_COLUMN: Nombre por defecto de la columna relacional.
        DEFAULT_CLASS_COLUMN: Nombre por defecto de la columna de etiquetas.
        DEFAULT_ENCODING: Codificación por defecto para leer archivos.
    """
    
    DEFAULT_BAG_COLUMN = 'bag'
    DEFAULT_CLASS_COLUMN = 'class'
    DEFAULT_ENCODING = 'utf-8'
    
    def __init__(self, 
                 bag_column: str = DEFAULT_BAG_COLUMN,
                 class_column: str = DEFAULT_CLASS_COLUMN,
                 encoding: str = DEFAULT_ENCODING):
        """
        Constructor del transformador ARFF a MIData.
        
        Args:
            bag_column: Nombre de la columna con la estructura relacional.
            class_column: Nombre de la columna con las etiquetas.
            encoding: Codificación del archivo.
        """
        self._bag_column = bag_column
        self._class_column = class_column
        self._encoding = encoding
        
        logger.debug(f"ArffToMIData inicializado: bag='{bag_column}', class='{class_column}'")
    
    @property
    def bag_column(self) -> str:
        """Nombre de la columna con estructura relacional."""
        return self._bag_column
    
    @property
    def class_column(self) -> str:
        """Nombre de la columna con etiquetas."""
        return self._class_column
    
    @property
    def encoding(self) -> str:
        """Codificación del archivo."""
        return self._encoding
    
    def load(self, file_path: str, dataset_name: Optional[str] = None) -> MIData:
        """
        Carga un archivo ARFF y lo convierte a MIData.
        
        Args:
            file_path: Ruta al archivo ARFF.
            dataset_name: Nombre del dataset (si es None, usa el nombre del archivo).
            
        Returns:
            Objeto MIData con el dataset cargado.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el formato es inválido o faltan columnas requeridas.
        """

        # Validamos que el archivo existe
        path = Path(file_path)
        if not path.exists():
            error_msg = f"Archivo no encontrado: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Usamos el nombre del archivo si no se proporciona dataset_name
        if dataset_name is None:
            dataset_name = path.stem
        
        logger.info(f"Iniciando carga de '{file_path}' como dataset '{dataset_name}'")
        
        # Cargar y validar estructura
        df, meta = self._load_arff_file(file_path)
        self._validate_structure(df, file_path)
        
        # Extraer esquema de instancias
        instance_schema = self._extract_instance_schema(file_path)
        
        # Construir bolsas
        bags = self._build_bags(df, instance_schema)
        
        # Crear y retornar MIData
        logger.info(f"Carga completada: {len(bags)} bolsas procesadas")
        return MIData(bags, dataset_name)
    
    def _load_arff_file(self, file_path: str) -> tuple:
        """
        Carga el archivo ARFF usando scipy.
        
        Args:
            file_path: Ruta al archivo.
            
        Returns:
            Tupla (DataFrame, metadata).
            
        Raises:
            ValueError: Si hay error al parsear el archivo.
        """
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            logger.debug(f"ARFF parseado: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df, meta
        except Exception as e:
            error_msg = f"Error parseando archivo ARFF: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_structure(self, df: pd.DataFrame, file_path: str):
        """
        Valida que el DataFrame tenga la estructura MIL requerida.
        
        Args:
            df: DataFrame con los datos.
            file_path: Ruta del archivo (para mensajes de error).
            
        Raises:
            ValueError: Si falta alguna columna requerida o la estructura es inválida.
        """
        # Validamos la columna de ID (primera columna)
        if len(df.columns) < 3:
            error_msg = (f"Estructura inválida en '{file_path}': "
                        f"se esperan al menos 3 columnas (ID, {self._bag_column}, {self._class_column})")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validamos columna de bolsa relacional
        if self._bag_column not in df.columns:
            error_msg = (f"Columna relacional '{self._bag_column}' no encontrada en '{file_path}'. "
                        f"Columnas disponibles: {list(df.columns)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validamos columna de clase (la última)
        if self._class_column not in df.columns:
            error_msg = (f"Columna de etiquetas '{self._class_column}' no encontrada en '{file_path}'. "
                        f"Columnas disponibles: {list(df.columns)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("Validación de estructura completada exitosamente")
    
    def _extract_instance_schema(self, file_path: str) -> List[Attribute]:
        """
        Extrae el esquema de atributos de las instancias desde el ARFF.
        
        Args:
            file_path: Ruta al archivo ARFF.
            
        Returns:
            Lista de objetos Attribute con el esquema.
            
        Raises:
            ValueError: Si no se encuentra la estructura relacional.
        """
        logger.info(f"Extrayendo esquema de instancias desde '{file_path}'")
        
        attributes = []
        inside_bag_def = False
        
        try:
            with open(file_path, 'r', encoding=self._encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Ignorar comentarios y líneas vacías
                    if not line or line.startswith('%'):
                        continue
                    
                    # Detectar inicio de estructura relacional
                    if line.lower().startswith(f"@attribute {self._bag_column.lower()} relational"):
                        inside_bag_def = True
                        logger.debug(f"Inicio de estructura relacional en línea {line_num}")
                        continue
                    
                    # Detectar fin de estructura relacional
                    if line.lower().startswith(f"@end {self._bag_column.lower()}"):
                        logger.debug(f"Fin de estructura relacional en línea {line_num}")
                        break
                    
                    # Capturar atributos dentro de la estructura
                    if inside_bag_def and line.lower().startswith("@attribute"):
                        attr = self._parse_attribute_line(line)
                        if attr:
                            attributes.append(attr)
                            logger.debug(f"Atributo detectado: {attr.name} ({attr.type})")
        
        except FileNotFoundError:
            error_msg = f"Archivo no encontrado: {file_path}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error leyendo esquema del archivo: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validar que se encontraron atributos
        if not attributes:
            error_msg = (f"No se encontraron atributos en la estructura relacional '{self._bag_column}' "
                        f"del archivo '{file_path}'")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Esquema extraído: {len(attributes)} atributos")
        return attributes
    
    def _parse_attribute_line(self, line: str) -> Optional[Attribute]:
        """
        Parsea una línea de definición de atributo ARFF.
        
        Args:
            line: Línea con formato "@attribute nombre tipo".
            
        Returns:
            Objeto Attribute o None si no se puede parsear.
        """
        parts = line.split()
        if len(parts) < 3:
            logger.warning(f"Línea de atributo mal formada: {line}")
            return None
        
        attr_name = parts[1]
        attr_type_raw = parts[2].lower()
        
        # Mapear tipos ARFF a tipos internos
        if attr_type_raw in ['numeric', 'real', 'float', 'double']:
            attr_type = 'real'
        elif attr_type_raw in ['integer', 'int']:
            attr_type = 'integer'
        elif attr_type_raw == 'string':
            attr_type = 'string'
        elif attr_type_raw.startswith('{'):
            # Tipo nominal: {valor1, valor2, ...}
            attr_type = 'nominal'
            # Extraer valores nominales (simplificado)
            values_str = ' '.join(parts[2:])
            values = [v.strip() for v in values_str.strip('{}').split(',')]
            return Attribute(attr_name, attr_type, values=values)
        else:
            attr_type = 'real'  # Por defecto
        
        return Attribute(attr_name, attr_type)
    
    def _build_bags(self, df: pd.DataFrame, instance_schema: List[Attribute]) -> List[Bag]:
        """
        Construye la lista de bolsas desde el DataFrame.
        
        Args:
            df: DataFrame con los datos.
            instance_schema: Esquema de las instancias.
            
        Returns:
            Lista de objetos Bag.
            
        Raises:
            ValueError: Si hay error construyendo las bolsas.
        """
        logger.info("Construyendo bolsas...")
        
        id_column = df.columns[0]
        bags = []
        
        for index, row in df.iterrows():
            try:
                # Extraer ID
                bag_id = self._decode_if_bytes(row[id_column])
                
                # Extraer etiqueta
                label = self._decode_if_bytes(row[self._class_column])
                
                # Extraer instancias
                raw_instances = row[self._bag_column]
                instances = self._build_instances(raw_instances, instance_schema)
                
                # Crear bolsa
                bag = Bag(bag_id=bag_id, label=label, instances=instances)
                bags.append(bag)
                
            except Exception as e:
                error_msg = f"Error construyendo bolsa en fila {index}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Estadísticas
        total_instances = sum(len(bag) for bag in bags)
        logger.info(f"Bolsas construidas: {len(bags)} bolsas, {total_instances} instancias totales")
        
        return bags
    
    def _build_instances(self, raw_instances, schema: List[Attribute]) -> List[Instance]:
        """
        Construye lista de instancias desde datos crudos.
        
        Args:
            raw_instances: Array numpy con las instancias crudas.
            schema: Esquema de los atributos.
            
        Returns:
            Lista de objetos Instance.
        """
        instances = []
        for raw_inst in raw_instances:
            values = list(raw_inst)
            instance = Instance(values, schema)
            instances.append(instance)
        return instances
    
    def _decode_if_bytes(self, value):
        """
        Decodifica un valor si es bytes.
        
        Args:
            value: Valor a decodificar.
            
        Returns:
            Valor decodificado o valor original.
        """
        return value.decode(self._encoding) if isinstance(value, bytes) else value
    
    def __repr__(self) -> str:
        """Representación técnica del transformador."""
        return (f"<ArffToMIData(bag_column='{self._bag_column}', "
                f"class_column='{self._class_column}', encoding='{self._encoding}')>")
    
    def __str__(self) -> str:
        """Representación legible del transformador."""
        return f"ARFF to MIData Transformer"


# Función de conveniencia para mantener compatibilidad
def load_arff_dataset(file_path: str, 
                     dataset_name: Optional[str] = None,
                     bag_column: str = ArffToMIData.DEFAULT_BAG_COLUMN,
                     class_column: str = ArffToMIData.DEFAULT_CLASS_COLUMN) -> MIData:
    """
    Función de conveniencia para cargar un dataset ARFF.
    
    Args:
        file_path: Ruta al archivo ARFF.
        dataset_name: Nombre del dataset.
        bag_column: Nombre de la columna relacional.
        class_column: Nombre de la columna de etiquetas.
        
    Returns:
        Objeto MIData.
    """
    loader = ArffToMIData(bag_column=bag_column, class_column=class_column)
    return loader.load(file_path, dataset_name)


if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Pruebas
    try:
        # Método 1: Usando la clase directamente
        loader = ArffToMIData()
        dataset1 = loader.load("datasets/musk1.arff")
        print(f"\n{dataset1}")
        
        # Método 2: Usando la función de conveniencia
        dataset2 = load_arff_dataset("datasets/musk1.arff", "musk1_test")
        print(f"\n{dataset2}")
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")