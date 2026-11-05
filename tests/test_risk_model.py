from pathlib import Path

import joblib

from src.risk_model import run_training

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.risk_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    train_models,
    evaluate_models,
)


def make_training_data(rows=100):
    """Create a deterministic, synthetic binary-classification dataset."""
    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "Age": rng.integers(18, 75, size=rows),
        "Biomarker_Level": rng.normal(50, 10, size=rows),
        "Pre_Existing_Conditions": rng.integers(0, 4, size=rows),
        "Vaccine_Dose_mcg": rng.choice([25, 50, 75, 100], size=rows),
        "Prior_Reaction_History": rng.choice(
            ["None", "Mild", "Moderate"],
            size=rows,
        ),
    })

    # Balanced target so stratified splitting can work reliably.
    data[TARGET_COLUMN] = np.tile([0, 1], rows // 2)
    if rows % 2:
        data.loc[rows - 1, TARGET_COLUMN] = 0

    return data


def test_feature_configuration_has_no_duplicates():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_train_models_returns_two_fitted_pipelines():
    data = make_training_data()

    models, X_test, y_test = train_models(data)

    assert set(models) == {"logistic_regression", "random_forest"}
    assert len(X_test) == 20
    assert len(y_test) == 20

    for model in models.values():
        assert isinstance(model, Pipeline)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        assert len(predictions) == len(y_test)
        assert probabilities.shape == (len(y_test), 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_train_models_rejects_missing_columns():
    data = make_training_data().drop(columns=["Age"])

    with pytest.raises(ValueError, match="Age"):
        train_models(data)


def test_train_models_rejects_single_target_class():
    data = make_training_data()
    data[TARGET_COLUMN] = 0

    with pytest.raises(ValueError, match="both binary target classes"):
        train_models(data)


def test_evaluate_models_writes_metrics_and_confusion_matrices(tmp_path):
    data = make_training_data()
    models, X_test, y_test = train_models(data)

    figures_dir = tmp_path / "outputs" / "figures"

    best_name, comparison = evaluate_models(
        models,
        X_test,
        y_test,
        figures_dir,
    )

    assert best_name in models
    assert len(comparison) == 2
    assert {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }.issubset(comparison.columns)

    assert comparison["f1"].between(0, 1).all()
    assert comparison["roc_auc"].between(0, 1).all()

    for model_name in models:
        assert (
            figures_dir / f"{model_name}_classification_report.json"
        ).exists()

        assert (
            figures_dir / f"{model_name}_confusion_matrix.png"
        ).exists()

    assert (
        tmp_path / "outputs" / "reports" / "model_comparison.csv"
    ).exists()
    assert (figures_dir / "model_comparison.csv").exists()


def test_saved_risk_model_predictions_match_after_reload(tmp_path):
    """Saved model predictions must remain identical after reloading."""
    data = make_training_data()

    models, X_test, _ = train_models(data)
    model = models["random_forest"]

    model_path = tmp_path / "risk_model.joblib"

    original_predictions = model.predict(X_test)

    joblib.dump(model, model_path)
    reloaded_model = joblib.load(model_path)

    reloaded_predictions = reloaded_model.predict(X_test)

    np.testing.assert_array_equal(
        original_predictions,
        reloaded_predictions,
    )