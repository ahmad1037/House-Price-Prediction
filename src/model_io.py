"""
Utilities for saving and loading models.
"""

from pathlib import Path
import joblib

from src.config import MODELS_DIR
def save_object(
    obj,
    filename: str,
):
    """
    Save any Python object using Joblib.

    Parameters
    ----------
    obj
        Object to save.

    filename : str
        Output filename.
    """

    MODELS_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    filepath = MODELS_DIR / filename

    joblib.dump(obj, filepath)

    print(f"Saved: {filepath}")

def load_object(
    filename: str,
):
    """
    Load a saved Joblib object.
    """

    filepath = MODELS_DIR / filename

    return joblib.load(filepath)