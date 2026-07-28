from src.models import (
    create_linear_regression,
    create_random_forest,
    create_gradient_boosting,
)
from sklearn.ensemble import RandomForestRegressor

def test_random_forest_creation():

    model = create_random_forest()

    assert isinstance(
        model,
        RandomForestRegressor,
    )