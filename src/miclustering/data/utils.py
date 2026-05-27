# src/miclustering/data/utils.py

from typing import Any, Dict, Optional

_NOMINAL_MAP: Dict[str, int] = {
    "positive": 1, "pos": 1, "yes": 1, "true": 1,
    "negative": 0, "neg": 0, "no": 0,  "false": 0,
}

def parse_label(raw: Any, nominal_map: Optional[Dict[str, int]] = None) -> int:
    """Convierte una etiqueta raw a int de forma robusta.

    Soporta: int, float, str numérico ('1.0'), bytes, nominal ('positive').

    Args:
        raw: Etiqueta en formato crudo.
        nominal_map: Mapa personalizado para etiquetas nominales.
                     Si None usa el mapa por defecto.

    Returns:
        Etiqueta como int.

    Raises:
        ValueError: Si la etiqueta no puede interpretarse.
    """
    if raw is None:
        raise ValueError("parse_label recibió None; se esperaba int, float, str o bytes.")
    
    mapping = nominal_map if nominal_map is not None else _NOMINAL_MAP
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw_lower = raw.strip().lower()
        if raw_lower in mapping:
            return mapping[raw_lower]
        return int(float(raw_lower))
    return int(float(raw))