"""Baseline machine-learning models for the mRNA-VIP learning project."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "patient_records_clean.csv"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TARGET_COLUMN = "Reaction_Score"
ID_COLUMNS = ["Patient_ID"]
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    """Train, evaluate, compare, and save baseline classifiers."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("This baseline script requires a binary target.")

    # Remove target and identifier columns from model inputs.
    excluded_columns = [TARGET_COLUMN] + [
        col for col in ID_COLUMNS if col in df.columns
    ]

    X = df.drop(columns=excluded_columns)
    y = df[TARGET_COLUMN]

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after excluding identifiers.")

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Reaction_Score must be encoded as numeric binary labels.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Reaction_Score must contain the labels 0 and 1.")

    # Determine feature types from the actual dataset.
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found.")

    print("Starting baseline model training...")
    print(f"Dataset shape: {df.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Stratification preserves the target proportions in each split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Fit imputers, scaling and encoding only on training data.
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    roc_data = []

    for model_name, classifier in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        results.append(metrics)

        # Save detailed classification report.
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Reaction_Score_0", "Reaction_Score_1"],
            zero_division=0,
        )

        report_path = REPORTS_DIR / f"{model_name}_report.txt"
        with report_path.open("w", encoding="utf-8") as file:
            file.write(f"Model: {model_name}\n")
            file.write(f"Training records: {len(X_train)}\n")
            file.write(f"Testing records: {len(X_test)}\n\n")
            file.write("Evaluation metrics:\n")
            file.write(
                pd.Series(metrics).to_string()
            )
            file.write("\n\nClassification report:\n")
            file.write(report)
            file.write("\n\nConfusion matrix:\n")
            file.write(str(confusion_matrix(y_test, predictions)))

        # Save the complete fitted pipeline, including preprocessing.
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        roc_data.append((model_name, pipeline))

        print(pd.Series(metrics).to_string())
        print(f"Report saved: {report_path}")
        print(f"Model saved: {model_path}")

    # Save side-by-side model comparison.
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORTS_DIR / "baseline_model_comparison.csv",
        index=False,
    )

    print("\nBaseline model comparison:")
    print(results_df.round(4).to_string(index=False))

    # Plot ROC curves on the same held-out test set.
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, pipeline in roc_data:
        RocCurveDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            name=model_name.replace("_", " ").title(),
            ax=ax,
        )

    ax.set_title("Baseline Model ROC Curve Comparison")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "baseline_roc_curves.png",
        dpi=300,
    )
    plt.close(fig)


    # Recreate predictions using the saved fitted pipelines.
    # Reloading is unnecessary; use the fitted pipelines retained above.
    prediction_columns = {
        "actual_reaction_score": y_test.to_numpy()
    }

    for model_name, pipeline in roc_data:
        prediction_columns[f"{model_name}_prediction"] = (
            pipeline.predict(X_test)
        )
        prediction_columns[f"{model_name}_probability"] = (
            pipeline.predict_proba(X_test)[:, 1]
        )

    predictions_df = pd.DataFrame(prediction_columns)
    predictions_df.to_csv(
        REPORTS_DIR / "baseline_test_predictions.csv",
        index=False,
    )

    summary = {
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target_column": TARGET_COLUMN,
        "excluded_identifier_columns": [
            col for col in ID_COLUMNS if col in df.columns
        ],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "models": results,
        "interpretation_note": (
            "These results are from a single held-out split of a "
            "learning dataset. They do not establish clinical validity."
        ),
    }

    with (REPORTS_DIR / "baseline_training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nBaseline model training completed successfully.")
    print(f"Comparison: {REPORTS_DIR / 'baseline_model_comparison.csv'}")
    print(f"ROC curves: {FIGURES_DIR / 'baseline_roc_curves.png'}")
    print(f"Test predictions: {REPORTS_DIR / 'baseline_test_predictions.csv'}")


if __name__ == "__main__":
    main()
