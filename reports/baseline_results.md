    # Linear Regression Baseline

## Validation Results

| Metric | Value |
|---------|------:|
| MAE | 16,034.11 |
| RMSE | 29,635.36 |
| R² | 0.89 |

## Notes

- Baseline model created using Linear Regression.
- Preprocessing pipeline includes:
  - Median imputation
  - Most frequent categorical imputation
  - Standard scaling
  - One-hot encoding

## Next Steps

Compare against:

- Random Forest
- Gradient Boosting
- XGBoost