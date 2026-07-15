"""
Machine Learning models.
"""
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
def create_linear_regression():
    """
    Create a Linear Regression model.

    Returns
    -------
    LinearRegression
        Untrained Linear Regression model.
    """

    return LinearRegression()

def create_random_forest():
    """
    Create a Random Forest Regressor.
    """

    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

def create_gradient_boosting():
    """
    Create a Gradient Boosting Regressor.
    """

    return GradientBoostingRegressor(
        random_state=42,
    )

def get_available_models():
    """
    Return all available models.
    """

    return {
        "Linear Regression": create_linear_regression(),
        "Random Forest": create_random_forest(),
        "Gradient Boosting": create_gradient_boosting(),
    }