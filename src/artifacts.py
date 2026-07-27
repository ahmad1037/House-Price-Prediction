import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
def load_feature_columns():

    with open(
        ARTIFACTS_DIR / "feature_columns.json"
    ) as f:

        return json.load(f)

def load_numerical_columns():
    with open(
        ARTIFACTS_DIR / "numerical_columns.json"
    ) as f:

        return json.load(f)

def load_categorical_columns():
    with open(
        ARTIFACTS_DIR / "categorical_columns.json"
    ) as f:

        return json.load(f)

