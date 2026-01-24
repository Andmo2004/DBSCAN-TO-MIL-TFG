from data.midata import MIData
from data.arff_reader import ArffToMIData, load_arff_dataset
import logging

# Configuración del sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mil_execution.log'),  # Log a archivo
        logging.StreamHandler()  # Log a consola
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Función principal que demuestra las diferentes formas de cargar datasets.
    """
    logger.info("=" * 70)
    logger.info("Iniciando aplicación MIL - Demostración de carga de datasets")
    logger.info("=" * 70)
    
    # ==========================================
    # MÉTODO 1: Usando MIData.from_arff() - RECOMENDADO
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("MÉTODO 1: Usando MIData.from_arff() (Factory Method)")
    logger.info("="*70)
    
    try:
        dataset1 = MIData.from_arff('datasets/musk1.arff')
        
        logger.info(f"Dataset cargado: {dataset1}")
        logger.info(f"Número de bolsas: {len(dataset1)}")
        logger.info(f"Primera bolsa: {dataset1[0]}")
        
        # Estadísticas
        positive_bags = dataset1.get_positive_bags()
        negative_bags = dataset1.get_negative_bags()
        logger.info(f"Bolsas positivas: {len(positive_bags)}")
        logger.info(f"Bolsas negativas: {len(negative_bags)}")
        
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
    except ValueError as e:
        logger.error(f"Error en formato: {e}")
    
    # ==========================================
    # MÉTODO 2: Usando MIData.from_arff() con configuración personalizada
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("MÉTODO 2: Carga con configuración personalizada")
    logger.info("="*70)
    
    try:
        # Para datasets con nombres de columnas diferentes
        # (este ejemplo usa los estándar, pero muestra cómo personalizarlo)
        dataset2 = MIData.from_arff(
            file_path='datasets/musk1.arff',
            dataset_name='musk1_custom',
            bag_column='bag',      # Puedes cambiar esto si tu ARFF usa otro nombre
            class_column='class'   # Puedes cambiar esto si tu ARFF usa otro nombre
        )
        
        logger.info(f"Dataset personalizado: {dataset2}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # ==========================================
    # MÉTODO 3: Usando la clase ArffToMIData directamente
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("MÉTODO 3: Usando ArffToMIData directamente (más control)")
    logger.info("="*70)
    
    try:
        # Crear el loader con configuración específica
        loader = ArffToMIData(
            bag_column='bag',
            class_column='class',
            encoding='utf-8'
        )
        
        logger.info(f"Loader creado: {loader}")
        
        # Cargar el dataset
        dataset3 = loader.load('datasets/musk1.arff', 'musk1_loader')
        
        logger.info(f"Dataset cargado: {dataset3}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # ==========================================
    # MÉTODO 4: Usando la función de conveniencia (retrocompatibilidad)
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("MÉTODO 4: Función de conveniencia (retrocompatibilidad)")
    logger.info("="*70)
    
    try:
        dataset4 = load_arff_dataset('datasets/musk1.arff', 'musk1_legacy')
        logger.info(f"Dataset cargado: {dataset4}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # ==========================================
    # DEMOSTRACIÓN: Split y operaciones
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("DEMOSTRACIÓN: Operaciones con el dataset")
    logger.info("="*70)
    
    try:
        # Usar el primer dataset
        logger.info("Dividiendo dataset en train/test (70/30)...")
        train_data, test_data = dataset1.split_data(70, seed=42)
        
        logger.info(f"Train: {train_data}")
        logger.info(f"Test: {test_data}")
        
        # Iterar sobre primeras 3 bolsas
        logger.info("\nPrimeras 3 bolsas del conjunto de entrenamiento:")
        for i, bag in enumerate(train_data):
            if i >= 3:
                break
            logger.info(f"  {i+1}. {bag}")
        
        # Acceso pythónico
        logger.info("\nAcceso pythónico a datos:")
        logger.info(f"  Total de bolsas: {len(train_data)}")
        logger.info(f"  Primera bolsa: {train_data[0]}")
        logger.info(f"  Primera instancia de primera bolsa: {train_data[0][0]}")
        
        # Verificar si una bolsa está en el dataset
        first_bag = train_data[0]
        logger.info(f"  ¿Primera bolsa está en train? {first_bag in train_data}")
        
    except Exception as e:
        logger.error(f"Error en operaciones: {e}")
    
    # ==========================================
    # MANEJO DE ERRORES: Ejemplos de validación
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("DEMOSTRACIÓN: Validación y manejo de errores")
    logger.info("="*70)
    
    # Error 1: Archivo no existe
    try:
        logger.info("Intentando cargar archivo inexistente...")
        MIData.from_arff('datasets/no_existe.arff')
    except FileNotFoundError as e:
        logger.info(f"  ✓ Error capturado correctamente: {type(e).__name__}")
    
    # Error 2: Columna faltante (simulación)
    try:
        logger.info("Intentando cargar con columna incorrecta...")
        MIData.from_arff(
            'datasets/musk1.arff',
            bag_column='columna_inexistente'
        )
    except ValueError as e:
        logger.info(f"  ✓ Error capturado correctamente: {type(e).__name__}")
    
    logger.info("\n" + "="*70)
    logger.info("Aplicación finalizada exitosamente")
    logger.info("="*70)


if __name__ == "__main__":
    main()