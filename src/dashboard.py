"""Build the mRNA-VIP educational project dashboard."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from src.attention import encode_sequence, self_attention


DISCLAIMER = (
    "EDUCATIONAL PROJECT ONLY | Not validated for clinical use. "
    "Not intended for medical or clinical decision-making."
)


def build_dashboard(project_root: Path) -> Path:
    """Build and save a three-panel project dashboard."""

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
        "retrieval_seconds",
        "report_generation_seconds",
    ]

    for key in required_latency_keys:
        if key not in latency_data:
            raise KeyError(
                f"Missing '{key}' in {latency_path}"
            )

    # Professional layout: heading and disclaimer above three panels.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
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

    # Panel 2: Computational self-attention demonstration.
    image = axes[1].imshow(
        attention,
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set(
        title="2. Self-Attention Demonstration",
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

    # Panel 3: Recorded pipeline timing.
    names = ["RAG Retrieval", "Offline Report Generation"]
    values = [
        float(latency_data["retrieval_seconds"]),
        float(latency_data["report_generation_seconds"]),
    ]

    bars = axes[2].bar(
        names,
        values,
        color=["#1D4ED8", "#D97706"],
        width=0.58,
    )
    axes[2].set(
        title="3. Pipeline Stage Latency",
        ylabel="Recorded Time (seconds)",
    )
    axes[2].tick_params(axis="x", labelrotation=10)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].bar_label(
        bars,
        fmt="%.4f s",
        padding=4,
        fontsize=9,
    )

    # Overall title and visible educational disclaimer.
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
        fontsize=10,
        color="#9A3412",
        fontweight="bold",
        wrap=True,
    )

    figure.text(
        0.5,
        0.035,
        (
            "Illustrative project outputs only. "
            "Model performance and attention weights do not establish "
            "clinical safety, vaccine eligibility, or treatment guidance."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
        wrap=True,
    )

    figure.tight_layout(rect=[0.025, 0.09, 0.975, 0.87])

    destination = figures_dir / "dashboard.png"

    figure.savefig(
        destination,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )

    plt.close(figure)

    print(f"Dashboard saved to: {destination}")
    return destination


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    build_dashboard(project_root)
