import pandas as pd

from src.preprocessing import (
    identify_feature_types,
    create_preprocessor,
)


def test_preprocessor_can_be_created():

    X = pd.DataFrame({
        "Age": [20, 30, 40],
        "Area": [1000, 1500, 2000],
        "Neighborhood": [
            "NAmes",
            "CollgCr",
            "NAmes",
        ],
    })

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features,
    )

    assert preprocessor is not None