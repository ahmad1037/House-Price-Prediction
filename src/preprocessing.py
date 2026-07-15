"""
Data preprocessing utilities.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import TARGET_COLUMN
def split_features_target(
    data: pd.DataFrame,
):
    """
    Split the dataset into features and target.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix (X) and target vector (y).
    """

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return X, y

def identify_feature_types(
    X: pd.DataFrame,
):
    """
    Identify numerical and categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    tuple[list, list]
        Numerical columns and categorical columns.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return (
        numerical_features,
        categorical_features,
    )

def create_preprocessor(
    numerical_features,
    categorical_features,
):
    """
    Create preprocessing pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "num",

                numerical_pipeline,

                numerical_features,

            ),

            (

                "cat",

                categorical_pipeline,

                categorical_features,

            ),

        ]

    )

    return preprocessor

def get_feature_names(
    preprocessor,
):
    """
    Return feature names after preprocessing.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A fitted preprocessing pipeline.

    Returns
    -------
    list[str]
        Feature names after transformation.
    """

    return preprocessor.get_feature_names_out().tolist()