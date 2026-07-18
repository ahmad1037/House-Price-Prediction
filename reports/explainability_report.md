# Model Explainability Report

## Feature Importance

The model relied most heavily on:

- OverallQual
- GrLivArea
- TotalBsmtSF
- GarageCars
- YearBuilt

---

## Residual Analysis

Residuals were generally centered around zero.

Some larger errors occurred for high-value properties, suggesting the model may struggle with luxury homes.

---

## Prediction Accuracy

The Actual vs Predicted plot showed a strong linear relationship, indicating good overall predictive performance.

---

## Future Improvements

- Engineer additional features.
- Tune hyperparameters further.
- Experiment with XGBoost.
- Analyze high-value outliers separately.

## SHAP Analysis

SHAP values were computed for the best-performing tree-based model.

### Key Insights

- OverallQual consistently increased predicted prices.
- Larger living areas contributed positively.
- Premium neighborhoods increased predictions.
- Lower kitchen quality reduced predicted prices.

### Benefits

SHAP provides both global and local interpretability, allowing individual predictions to be explained in terms of feature contributions.