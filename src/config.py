"""
Project configuration settings.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Images directory
IMAGES_DIR = PROJECT_ROOT / "images"

# Target variable
TARGET_COLUMN = "SalePrice"

# Reproducibility
RANDOM_STATE = 42

# Test size for validation
TEST_SIZE = 0.2

# Random seed
RANDOM_STATE = 42