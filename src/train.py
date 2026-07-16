"""
Training script for the House Price Prediction project.
"""
from matplotlib.pyplot import grid
import pandas as pd
from xml.parsers.expat import model
from src.tuning import grid_search_tuning
from src.model_configs import RANDOM_FOREST_PARAMS
from src.models import create_random_forest
from sklearn import metrics
from sklearn.model_selection import train_test_split
from src.explainability import (
    get_feature_importance,
    plot_feature_importance,
    calculate_residuals,
    plot_predictions,
    plot_residuals
)
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
)
from src.models import create_linear_regression
from src.preprocessing import get_feature_names
from src.evaluation import (
    evaluate_regression_model,
)

from src.models import get_available_models

from src.data_loader import load_train_data

from src.preprocessing import (
    split_features_target,
    identify_feature_types,
    create_preprocessor,
)
from src.model_io import (
    save_object,
)

def main():
    """
    Main training workflow.
    """

    print("Loading dataset...")

    train_df = load_train_data()

    print("Dataset loaded.")

    X, y = split_features_target(train_df)

    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    )
    print("\nTraining Set")

    print(X_train.shape)

    print(y_train.shape)

    print("\nValidation Set")

    print(X_valid.shape)

    print(y_valid.shape)

    numerical_features, categorical_features = (
    identify_feature_types(X_train)
)
    preprocessor = create_preprocessor(
    numerical_features,
    categorical_features,
)
    X_train_processed = preprocessor.fit_transform(
    X_train
)
    X_valid_processed = preprocessor.transform(
    X_valid
)
    print("\nProcessed Training Data")

    print(X_train_processed.shape)

    print("\nProcessed Validation Data")

    print(X_valid_processed.shape)

    models = get_available_models()

    best_rmse = float("inf")
    best_model = None
    best_model_name = None
    results = []
    metrics = {}
    for model_name, model in models.items():

        model.fit(
            X_train_processed,
            y_train,
        )

        predictions = model.predict(
            X_valid_processed,
        )

        metrics = evaluate_regression_model(
            y_valid,
            predictions,
        )

        results.append({
            "Model": model_name,
            **metrics,
        })
        if metrics["RMSE"] < best_rmse:

            best_rmse = metrics["RMSE"]

            best_model = model

            best_model_name = model_name
        

    results_df = pd.DataFrame(results)
    print(results_df)
    """save_object(
    best_model,
    "best_model.joblib",
)"""

    print(f"Best model: {best_model_name}")
    results_df.to_csv(
    "reports/model_comparison.csv",
    index=False,
)
    rf = create_random_forest()
    """
    grid = grid_search_tuning(
        model=rf,
        param_grid=RANDOM_FOREST_PARAMS,
        X_train=X_train_processed,
        y_train=y_train,
)
    print("Best Parameters:")
    print(grid.best_params_)
    print("Best CV RMSE:")
    print(-grid.best_score_)
    best_rf = grid.best_estimator_

    predictions = best_rf.predict(
        X_valid_processed
    )

    metrics = evaluate_regression_model(
        y_valid,
        predictions,
    )

    print(metrics)
    save_object(
    best_rf,
    "best_model.joblib",
)"""
    preprocessor.fit(X_train)


    feature_names = get_feature_names(preprocessor)
    importance_df = get_feature_importance(
        best_model,
        feature_names,
    )

    importance_df.head()
    print(importance_df.head())

    importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False,
)
    residuals = calculate_residuals(
    y_valid,
    predictions,
)
    #plot_feature_importance(importance_df)
    print(residuals[:10])
    residual_df = pd.DataFrame({

    "Actual": y_valid,

    "Predicted": predictions,

    "Residual": residuals,

    }
)   
    print(residual_df)

    residual_df.to_csv(
    "reports/residuals.csv",
    index=False,
)
    largest_errors = residual_df.copy()

    largest_errors["Absolute Error"] = (
        largest_errors["Residual"].abs()
    )

    largest_errors = largest_errors.sort_values(
        by="Absolute Error",
        ascending=False,
    )

    largest_errors.head(10)
    #plot_residuals(residual_df)
    plot_predictions(y_valid, predictions)
    




if __name__ == "__main__":
    main()