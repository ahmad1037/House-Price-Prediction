from pathlib import Path

from src.model_io import (
    save_object,
    load_object,
)

def test_save_and_load_object(tmp_path):

    obj = {"message": "hello"}

    file_path = tmp_path / "test.joblib"

    save_object(
        obj,
        file_path,
    )

    loaded = load_object(
        file_path,
    )

    assert loaded == obj