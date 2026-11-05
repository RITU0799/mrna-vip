"""Build the mRNA-VIP educational project dashboard."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from src.attention import encode_sequence, self_attention


DISCLAIMER = (
    "Student engineering project output only. "
    "Not real clinical, medical, or regulatory guidance."
)


def build_dashboard(project_root: Path) -> Path:
    """Build and save a three-panel educational project dashboard."""

    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load saved model evaluation results.
    evaluation_path = outputs_dir / "risk_evaluation.joblib"

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Model evaluation file not found: {evaluation_path}. "
            "Run the risk model module first."
        )

    evaluation = joblib.load(evaluation_path)

    required_evaluation_keys = ["y_test", "scores", "model_name"]

    for key in required_evaluation_keys:
        if key not in evaluation:
            raise KeyError(
                f"Missing '{key}' in {evaluation_path}"
            )

    model_name = str(evaluation["model_name"])

    fpr, tpr, _ = roc_curve(
        evaluation["y_test"],
        evaluation["scores"],
    )
    roc_auc = auc(fpr, tpr)

    # Generate the educational self-attention demonstration.
    sequence = "ACGUACGUACGGCUA"
    attention, _, _ = self_attention(encode_sequence(sequence))

    # Load recorded pipeline timing results.
    latency_path = outputs_dir / "report_generation_latency.json"

    if not latency_path.exists():
        raise FileNotFoundError(
            f"Latency file not found: {latency_path}. "
            "Run the retrieval and report-generation steps first."
        )

    latency_data = json.loads(
        latency_path.read_text(encoding="utf-8")
    )

    required_latency_keys = [
        "retrieval_stage_seconds",
        "model_scoring_seconds",
        "report_generation_seconds",
        "total_pipeline_seconds",
        "report_generation_mode",
    ]

    for key in required_latency_keys:
        if key not in latency_data:
            raise KeyError(
                f"Missing '{key}' in {latency_path}"
            )

    # Prepare pipeline performance measurements.
    names = [
        "RAG Retrieval",
        "Model Scoring",
        "Report Generation",
        "Total Pipeline",
    ]

    values = [
        float(latency_data["retrieval_stage_seconds"]),
        float(latency_data["model_scoring_seconds"]),
        float(latency_data["report_generation_seconds"]),
        float(latency_data["total_pipeline_seconds"]),
    ]

    if any(value < 0 for value in values):
        raise ValueError("Latency measurements cannot be negative.")

    # Professional three-panel layout.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 7),
    )

    figure.patch.set_facecolor("#F5F7FA")

    for axis in axes:
        axis.set_facecolor("#FFFFFF")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors="#334155")
        axis.title.set_color("#0F172A")
        axis.xaxis.label.set_color("#334155")
        axis.yaxis.label.set_color("#334155")

    # Panel 1: Model evaluation.
    axes[0].plot(
        fpr,
        tpr,
        color="#0F766E",
        linewidth=2.5,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )

    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94A3B8",
        linewidth=1.5,
        label="Random baseline",
    )

    axes[0].set(
        title="1. Risk Classifier Evaluation",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )

    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(alpha=0.2)

    axes[0].text(
        0.5,
        -0.19,
        f"Model: {model_name}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
        color="#475569",
    )

    # Panel 2: Computational self-attention demonstration.
    image = axes[1].imshow(
        attention,
        cmap="viridis",
        aspect="auto",
    )

    axes[1].set(
        title="2. Computational Self-Attention",
        xlabel="Attended Position",
        ylabel="Query Position",
    )

    axes[1].set_xticks(range(len(sequence)), list(sequence))
    axes[1].set_yticks(range(len(sequence)), list(sequence))
    axes[1].tick_params(axis="both", labelsize=8)

    colorbar = figure.colorbar(
        image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label("Attention Weight")

    axes[1].text(
        0.5,
        -0.19,
        "Illustrative computational demonstration only",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8,
        color="#475569",
    )

    # Panel 3: Complete pipeline performance.
    bars = axes[2].bar(
        names,
        values,
        color=["#1D4ED8", "#0F766E", "#D97706", "#7C3AED"],
        width=0.62,
    )

    axes[2].set(
        title="3. Pipeline Performance",
        ylabel="Recorded Time (seconds)",
    )

    axes[2].tick_params(
        axis="x",
        labelrotation=18,
        labelsize=8,
    )

    axes[2].grid(axis="y", alpha=0.2)
    axes[2].set_axisbelow(True)

    axes[2].bar_label(
        bars,
        labels=[f"{value:.3f} s" for value in values],
        padding=4,
        fontsize=8,
    )

    axes[2].set_ylim(
        0,
        max(values) * 1.18 if max(values) > 0 else 1,
    )

    # Overall heading and educational disclaimer.
    figure.suptitle(
        "mRNA-VIP | Educational ML & RAG Demonstration",
        fontsize=19,
        fontweight="bold",
        color="#0F172A",
        y=0.97,
    )

    figure.text(
        0.5,
        0.915,
        DISCLAIMER,
        ha="center",
        va="center",
        fontsize=9,
        color="#9A3412",
        fontweight="bold",
        wrap=True,
    )

    figure.text(
        0.5,
        0.035,
        (
            f"Generation mode: {latency_data['report_generation_mode']} | "
            "Illustrative project outputs only. Model performance and "
            "attention weights do not establish clinical safety, "
            "vaccine eligibility, or treatment guidance."
        ),
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
        wrap=True,
    )

    figure.tight_layout(
        rect=[0.025, 0.11, 0.975, 0.87]
    )

    destination = figures_dir / "dashboard.png"

    figure.savefig(
        destination,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )

    plt.close(figure)

    print(f"Dashboard saved to: {destination}")
    print(f"Model: {model_name}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Total pipeline time: {values[3]:.4f} seconds")

    return destination


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    build_dashboard(project_root)
