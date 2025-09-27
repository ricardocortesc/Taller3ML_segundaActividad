"""
settings.py — Configuración para Workshop PCA.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Parámetros por defecto del flujo
N_COMPONENTS = 2           # número de componentes principales
STANDARDIZE = True         # estandarizar antes de PCA (recomendado)
SEED = 42                  # para reproducibilidad

# Si quieres cargar desde archivo CSV, pon aquí la ruta (o déjalo en None y usamos iris)
CSV_PATH = None            # DATA_DIR / "tu_archivo.csv"  # <- ajusta si quieres
TARGET_COL = None          # si tu CSV tiene columna objetivo, nómbrala; si no, deja None
