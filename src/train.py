"""
Training script for the House Price Prediction project.
"""
import pandas as pd
from xml.parsers.expat import model

from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
)
from src.models import create_linear_regression

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
    save_object(
    best_model,
    "best_model.joblib",
)

    print(f"Best model: {best_model_name}")
    results_df.to_csv(
    "reports/model_comparison.csv",
    index=False,
)


if __name__ == "__main__":
    main()