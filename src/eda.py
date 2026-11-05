"""Exploratory data analysis for the mRNA-VIP learning project."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "patient_records_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

TARGET_COLUMN = "Reaction_Score"


def run_eda():
    """Generate summary statistics, plots, and an EDA report."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run the data validation pipeline first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    print("Starting exploratory data analysis...")
    print(f"Dataset shape: {df.shape}")

    # Summary statistics
    summary = df.describe(include="all").transpose()
    summary.to_csv(REPORTS_DIR / "eda_summary_statistics.csv")

    # Target distribution
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=target_counts.index.astype(str),
        y=target_counts.values,
    )
    plt.title("Reaction Score Distribution")
    plt.xlabel("Reaction Score")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reaction_score_distribution.png", dpi=300)
    plt.close()

    # Numeric feature distributions
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        df[numeric_columns].hist(
            figsize=(12, 8),
            bins=20,
            edgecolor="black",
        )
        plt.suptitle("Numeric Feature Distributions")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "numeric_feature_distributions.png", dpi=300)
        plt.close()

    # Correlation heatmap
    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            center=0,
        )
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=300)
        plt.close()

        correlation.to_csv(REPORTS_DIR / "feature_correlations.csv")

    # Reaction score by categorical and binary features
    categorical_columns = [
        "Pre_Existing_Conditions",
        "Prior_Reaction_History",
    ]

    group_summaries = []

    for column in categorical_columns:
        if column in df.columns:
            grouped = (
                df.groupby(column, dropna=False)[TARGET_COLUMN]
                .agg(["count", "mean"])
                .reset_index()
            )
            grouped["feature"] = column
            group_summaries.append(grouped)

            plt.figure(figsize=(9, 5))
            sns.barplot(
                data=df,
                x=column,
                y=TARGET_COLUMN,
                errorbar=None,
            )
            plt.title(f"Reaction Score Rate by {column}")
            plt.xlabel(column)
            plt.ylabel("Mean Reaction Score")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            safe_name = column.lower().replace(" ", "_")
            plt.savefig(
                FIGURES_DIR / f"reaction_rate_by_{safe_name}.png",
                dpi=300,
            )
            plt.close()

    if group_summaries:
        pd.concat(group_summaries, ignore_index=True).to_csv(
            REPORTS_DIR / "categorical_reaction_summary.csv",
            index=False,
        )

    # Text report
    report_path = REPORTS_DIR / "eda_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("mRNA-VIP Exploratory Data Analysis Report\n")
        report.write("=" * 45 + "\n\n")
        report.write(f"Dataset shape: {df.shape}\n\n")
        report.write("Missing values:\n")
        report.write(df.isnull().sum().to_string())
        report.write("\n\nTarget distribution:\n")
        report.write(target_counts.to_string())
        report.write("\n\nTarget percentages:\n")
        report.write(
            (df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100)
            .round(2)
            .to_string()
        )
        report.write("\n\nSummary statistics:\n")
        report.write(summary.to_string())

    print("\nEDA completed successfully.")
    print(f"Summary statistics: {REPORTS_DIR / 'eda_summary_statistics.csv'}")
    print(f"EDA report: {report_path}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
