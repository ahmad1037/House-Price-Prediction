import pandas as pd
import pytest

from src.preprocessing import create_preprocessor

@pytest.fixture
def sample_house():

    return pd.DataFrame({
        "OverallQual": [7],
        "GrLivArea": [1710],
        "GarageCars": [2],
        "TotalBsmtSF": [856],
        "YearBuilt": [2003],
    })

def test_preprocessor_can_fit(sample_house):

    preprocessor = create_preprocessor()

    preprocessor.fit(sample_house)

    transformed = preprocessor.transform(sample_house)

    assert transformed.shape[0] == 1