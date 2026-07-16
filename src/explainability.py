"""
Model explainability utilities.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def get_feature_importance(
    model,
    feature_names,
):
    """
    Return feature importance as a DataFrame.
    """

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )
    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    return importance  

def plot_feature_importance(
    importance_df,
    top_n=20,
):
    """
    Plot the top feature importances.
    """

    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.gca().invert_yaxis()

    plt.title(f"Top {top_n} Feature Importances")
    plt.show()

def calculate_residuals(
    y_true,
    y_pred,
):
    """
    Calculate prediction residuals.
    """

    residuals = y_true - y_pred

    return residuals

def plot_residuals(
    residual_df,
):
    """
    Plot residuals against predicted values.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        residual_df["Predicted"],
        residual_df["Residual"],
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel("Predicted Price")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        "reports/residual_plot.png"
    )

    plt.show()

def plot_predictions(
    y_true,
    y_pred,
):
    """
    Plot actual vs predicted values.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
    )

    min_value = min(
        y_true.min(),
        y_pred.min(),
    )

    max_value = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
    )

    plt.xlabel("Actual Price")

    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(
        "reports/prediction_vs_actual.png"
    )

    plt.show()