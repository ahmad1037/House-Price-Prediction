"""
Hyperparameter tuning utilities.
"""

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
def grid_search_tuning(
    model,
    param_grid,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=5,
):
    """
    Perform Grid Search with cross-validation.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    return grid_search