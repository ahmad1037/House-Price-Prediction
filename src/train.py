"""
Training script for the House Price Prediction project.
"""

from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
)

from src.data_loader import load_train_data

from src.preprocessing import (
    split_features_target,
    identify_feature_types,
    create_preprocessor,
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

if __name__ == "__main__":
    main()