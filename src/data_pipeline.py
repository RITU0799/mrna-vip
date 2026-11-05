"""Data validation and preprocessing for the mRNA VIP learning project."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------
# 1. Project paths and required columns
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

RAW_DATA_PATH = DATA_DIR / "patient_records.csv"
CLEAN_DATA_PATH = DATA_DIR / "patient_records_clean.csv"
VALIDATION_REPORT_PATH = OUTPUT_REPORT_DIR / "data_validation.txt"

REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score",
]

NUMERIC_COLUMNS = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Reaction_Score",
]


# ---------------------------------------------------------
# 2. Custom validation exception
# ---------------------------------------------------------

class MissingRequiredColumnError(ValueError):
    """Raised when the input dataset lacks required columns."""


# ---------------------------------------------------------
# 3. Load and validate the raw dataset
# ---------------------------------------------------------

def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the CSV and validate its required fields and values."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            "Please ensure patient_records.csv exists in the data folder."
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise MissingRequiredColumnError(
            "Missing required column(s): "
            + ", ".join(missing_columns)
        )

    # Keep only the seven columns specified in the project requirements.
    dataframe = dataframe[REQUIRED_COLUMNS].copy()

    # Patient identifiers must be present and unique.
    if dataframe["Patient_ID"].isna().any():
        raise ValueError("Patient_ID contains missing values.")

    if dataframe["Patient_ID"].duplicated().any():
        raise ValueError("Duplicate Patient_ID values were found.")

    # The target must be available for supervised learning.
    if dataframe["Reaction_Score"].isna().any():
        raise ValueError("Reaction_Score contains missing values.")

    # Validate numeric fields.
    for column in NUMERIC_COLUMNS:
        try:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{column}' contains non-numeric values."
            ) from exc

        if dataframe[column].isna().any() and column != "Biomarker_Level":
            raise ValueError(
                f"Column '{column}' contains unexpected missing values."
            )

        if not dataframe[column].dropna().map(
            lambda value: pd.notna(value)
            and abs(value) != float("inf")
        ).all():
            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

    if not dataframe["Reaction_Score"].isin([0, 1]).all():
        raise ValueError("Reaction_Score must contain only 0 and 1.")

    # Validate prior-reaction categories.
    valid_reactions = {"Mild", "Moderate", "None"}

    observed_reactions = set(
        dataframe["Prior_Reaction_History"].dropna().unique()
    )

    unexpected_reactions = observed_reactions - valid_reactions

    if unexpected_reactions:
        raise ValueError(
            "Unexpected Prior_Reaction_History value(s): "
            + ", ".join(map(str, sorted(unexpected_reactions)))
        )

    return dataframe


# ---------------------------------------------------------
# 4. Impute missing values and create an audit
# ---------------------------------------------------------

def clean_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Impute missing fields and return cleaned data plus audit details."""

    cleaned = dataframe.copy()

    missing_before = cleaned.isna().sum().to_dict()

    # Median imputation for the numerical biomarker field.
    biomarker_imputer = SimpleImputer(strategy="median")

    cleaned[["Biomarker_Level"]] = biomarker_imputer.fit_transform(
        cleaned[["Biomarker_Level"]]
    )

    biomarker_median = float(
        biomarker_imputer.statistics_[0]
    )

    # Mode imputation for prior reaction history.
    reaction_imputer = SimpleImputer(strategy="most_frequent")

    cleaned[["Prior_Reaction_History"]] = (
        reaction_imputer.fit_transform(
            cleaned[["Prior_Reaction_History"]]
        )
    )

    reaction_mode = str(reaction_imputer.statistics_[0])

    # Fail explicitly if any missing values remain.
    remaining_missing = cleaned.columns[
        cleaned.isna().any()
    ].tolist()

    if remaining_missing:
        raise ValueError(
            "Cleaning left missing values in: "
            + ", ".join(remaining_missing)
        )

    # Preserve record count and check for duplicate rows.
    if cleaned.duplicated().any():
        raise ValueError("Duplicate rows were found in the cleaned dataset.")

    audit = {
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
        "missing_before": missing_before,
        "missing_after": cleaned.isna().sum().to_dict(),
        "biomarker_median_used": biomarker_median,
        "reaction_history_mode_used": reaction_mode,
        "target_distribution": (
            cleaned["Reaction_Score"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "duplicate_patient_ids": int(
            cleaned["Patient_ID"].duplicated().sum()
        ),
    }

    return cleaned, audit


# ---------------------------------------------------------
# 5. Save the data validation report
# ---------------------------------------------------------

def save_validation_report(audit: dict) -> None:
    """Write validation and cleaning results to a text report."""

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "mRNA Vaccine Clinical Safety (mRNA VIP)",
        "DATA VALIDATION AND CLEANING REPORT",
        "=" * 55,
        "",
        "IMPORTANT: Educational course project only.",
        "Not intended for real clinical or medical decision-making.",
        "",
        f"Number of records: {audit['rows']}",
        f"Number of columns: {audit['columns']}",
        f"Biomarker median used: {audit['biomarker_median_used']:.4f}",
        (
            "Prior reaction history mode used: "
            f"{audit['reaction_history_mode_used']}"
        ),
        f"Duplicate patient IDs: {audit['duplicate_patient_ids']}",
        "",
        "Missing values BEFORE cleaning:",
    ]

    for column, count in audit["missing_before"].items():
        lines.append(f"  {column}: {count}")

    lines.extend([
        "",
        "Missing values AFTER cleaning:",
    ])

    for column, count in audit["missing_after"].items():
        lines.append(f"  {column}: {count}")

    lines.extend([
        "",
        "Reaction_Score distribution:",
    ])

    for target, count in audit["target_distribution"].items():
        lines.append(f"  {target}: {count}")

    lines.extend([
        "",
        "DATA QUALITY NOTES",
        "-" * 55,
        (
            "Prior_Reaction_History may contain missing values in "
            "the supplied dataset."
        ),
        (
            "Mode imputation follows the project specification, but "
            "imputation can introduce bias."
        ),
        (
            "Missingness and the imputation approach should be discussed "
            "when interpreting model results."
        ),
        (
            "Feature standardization is performed in the model pipeline, "
            "where the scaler is fitted on training data."
        ),
        "",
        "This report documents preprocessing only; it does not establish "
        "clinical safety or vaccine eligibility.",
    ])

    VALIDATION_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# 6. Combined interface for model training
# ---------------------------------------------------------

def load_and_clean(
    path: str | Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Load, validate, clean, save, and return data for model training.

    This interface is used by src.risk_model.
    """

    raw_data = load_data(path)
    cleaned_data, audit = clean_data(raw_data)

    # Save the cleaned dataset in the standard project location.
    cleaned_data.to_csv(CLEAN_DATA_PATH, index=False)

    # Save the validation audit.
    save_validation_report(audit)

    return cleaned_data


# ---------------------------------------------------------
# 7. Run the complete pipeline
# ---------------------------------------------------------

def main() -> None:
    """Validate, clean, save, and report on the input dataset."""

    print("Starting data validation and cleaning...")

    raw_data = load_data()
    print(f"Raw dataset shape: {raw_data.shape}")

    cleaned_data, audit = clean_data(raw_data)

    cleaned_data.to_csv(CLEAN_DATA_PATH, index=False)
    save_validation_report(audit)

    print("\nData validation and cleaning completed successfully.")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    print("\nMissing values after cleaning:")
    print(cleaned_data.isna().sum().to_string())

    print(f"\nCleaned CSV saved to:\n{CLEAN_DATA_PATH}")
    print(f"\nValidation report saved to:\n{VALIDATION_REPORT_PATH}")

    print("\nTarget distribution:")
    print(cleaned_data["Reaction_Score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
