"""Train, evaluate, and save the mRNA VIP educational risk classifier."""

from pathlib import Path
import json

import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_pipeline import load_and_clean


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42
TARGET_COLUMN = "Reaction_Score"

# Every feature appears exactly once.
NUMERIC_FEATURES = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
]

CATEGORICAL_FEATURES = [
    "Prior_Reaction_History",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------
# 2. Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Scale numerical fields and encode categorical fields."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------
# 3. Train candidate models
# ---------------------------------------------------------

def train_models(
    dataframe: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Train Logistic Regression and Random Forest on a stratified split."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing training columns: " + ", ".join(missing_columns)
        )

    if len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS)):
        raise ValueError("Duplicate feature names detected.")

    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    if target.nunique() != 2:
        raise ValueError(
            "Reaction_Score must contain both binary target classes."
        )

    train_features, test_features, train_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=target,
        )
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
    }

    pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(train_features, train_target)
        pipelines[name] = pipeline

        print(f"Trained: {name}")

    return pipelines, test_features, test_target


# ---------------------------------------------------------
# 4. Evaluate candidate models
# ---------------------------------------------------------

def evaluate_models(
    models: dict,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    output_dir: Path,
) -> tuple[str, pd.DataFrame]:
    """Evaluate models and save reports and confusion matrices."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIX: Create the reports directory before saving CSV files.
    reports_dir = output_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, model in models.items():
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)[:, 1]

        report = classification_report(
            test_target,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        report_path = (
            output_dir / f"{name}_classification_report.json"
        )

        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        display = ConfusionMatrixDisplay.from_predictions(
            test_target,
            predictions,
            cmap="Blues",
            colorbar=False,
        )

        display.ax_.set_title(
            f"{name.replace('_', ' ').title()} Confusion Matrix"
        )

        display.figure_.tight_layout()

        display.figure_.savefig(
            output_dir / f"{name}_confusion_matrix.png",
            dpi=180,
        )

        plt.close(display.figure_)

        rows.append(
            {
                "model": name,
                "accuracy": float(report["accuracy"]),
                "precision": float(report["1"]["precision"]),
                "recall": float(report["1"]["recall"]),
                "f1": float(
                    f1_score(
                        test_target,
                        predictions,
                        zero_division=0,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(test_target, probabilities)
                ),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    # Save the comparison in the reports directory.
    comparison.to_csv(
        reports_dir / "model_comparison.csv",
        index=False,
    )

    # Preserve the comparison CSV in the figures directory too.
    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    best_name = str(comparison.iloc[0]["model"])

    return best_name, comparison


# ---------------------------------------------------------
# 5. Run training and persist artifacts
# ---------------------------------------------------------

def run_training(project_root: Path) -> dict:
    """Train models, evaluate them, and save the selected pipeline."""

    project_root = Path(project_root)

    print("Loading and cleaning dataset...")

    dataframe = load_and_clean(
        project_root / "data" / "patient_records.csv"
    )

    print(f"Dataset shape: {dataframe.shape}")
    print(f"Unique model features: {FEATURE_COLUMNS}")

    # Save the cleaned dataset at the standard path.
    cleaned_path = (
        project_root / "data" / "patient_records_clean.csv"
    )

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(cleaned_path, index=False)

    print("\nTraining candidate models...")

    models, test_features, test_target = train_models(dataframe)

    figures_dir = project_root / "outputs" / "figures"
    reports_dir = project_root / "outputs" / "reports"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating models...")

    best_name, comparison = evaluate_models(
        models,
        test_features,
        test_target,
        figures_dir,
    )

    selected_model = models[best_name]

    # Preserve the artifact path expected by report_generator.py.
    model_path = project_root / "outputs" / "risk_model.joblib"

    joblib.dump(selected_model, model_path)

    probabilities = selected_model.predict_proba(
        test_features
    )[:, 1]

    # Preserve the artifact keys expected by dashboard.py.
    evaluation_path = (
        project_root / "outputs" / "risk_evaluation.joblib"
    )

    joblib.dump(
        {
            "X_test": test_features,
            "y_test": test_target,
            "scores": probabilities,
            "model_name": best_name,
        },
        evaluation_path,
    )

    # Save the final comparison in the reports directory.
    comparison.to_csv(
        reports_dir / "model_comparison.csv",
        index=False,
    )

    best_row = comparison.loc[
        comparison["model"] == best_name
    ].iloc[0]

    summary = (
        "# Educational Model Selection Summary\n\n"
        f"Selected model: **{best_name.replace('_', ' ').title()}**\n\n"
        f"- Held-out accuracy: {best_row['accuracy']:.3f}\n"
        f"- Held-out precision: {best_row['precision']:.3f}\n"
        f"- Held-out recall: {best_row['recall']:.3f}\n"
        f"- Held-out F1: {best_row['f1']:.3f}\n"
        f"- Held-out ROC-AUC: {best_row['roc_auc']:.3f}\n\n"
        "The model was selected using F1 on the same held-out split "
        "used for reporting these metrics. Therefore, these results "
        "are exploratory and the test split is not an independent "
        "final evaluation after model selection.\n\n"
        "This is an educational model trained on the supplied dataset. "
        "It is not clinically validated and must not be used for "
        "real patient-care decisions.\n"
    )

    summary_path = (
        project_root / "outputs" / "model_selection_summary.md"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation data: {evaluation_path}")
    print(f"Saved comparison: {reports_dir / 'model_comparison.csv'}")

    return {
        "model_name": best_name,
        "comparison": comparison.to_dict(orient="records"),
        "model_path": str(model_path),
        "evaluation_path": str(evaluation_path),
    }


# ---------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    result = run_training(root)

    print("\nTraining completed successfully.")
    print(result)
