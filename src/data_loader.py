"""
Functions for loading and saving datasets.
"""

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

def load_train_data():
    """
    Load the training dataset.

    Returns
    -------
    pd.DataFrame
        Training dataset.
    """

    file_path = RAW_DATA_DIR / "train.csv"

    return pd.read_csv(file_path)

def load_test_data():
    """
    Load the test dataset.

    Returns
    -------
    pd.DataFrame
        Test dataset.
    """

    file_path = RAW_DATA_DIR / "test.csv"

    return pd.read_csv(file_path)