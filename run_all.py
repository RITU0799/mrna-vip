"""Run the complete mRNA VIP educational project pipeline."""

from pathlib import Path
import traceback

from src.data_pipeline import load_and_clean
from src.risk_model import run_training
from src.attention import encode_sequence, self_attention, plot_attention
from src.rag_index import run_retrieval_demo
from src.report_generator import run_sample_report
from src.dashboard import build_dashboard


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "outputs" / "figures"


def run_pipeline() -> None:
    """Execute the project modules in dependency order."""

    steps = [
        ("Data validation and cleaning", lambda: load_and_clean()),
        ("Risk model training and evaluation", lambda: run_training(ROOT)),
        (
            "Self-attention analysis",
            lambda: plot_attention(
                self_attention(encode_sequence("ACGUACGUACGGCUA"))[0],
                "ACGUACGUACGGCUA",
                FIGURES_DIR / "attention_heatmap.png",
            ),
        ),
        ("RAG retrieval evaluation", lambda: run_retrieval_demo(ROOT)),
        ("Structured report generation", lambda: run_sample_report(ROOT)),
        ("Dashboard generation", lambda: build_dashboard(ROOT)),
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, action in steps:
        print(f"\n{'=' * 60}\nSTARTING: {name}\n{'=' * 60}")

        try:
            action()
        except Exception as exc:
            print(f"\nFAILED: {name}")
            print(f"Error: {exc}")
            traceback.print_exc()
            raise RuntimeError(
                f"Pipeline stopped because '{name}' failed."
            ) from exc

        print(f"COMPLETED: {name}")

    print("\nAll pipeline modules completed successfully.")
    print(f"Project outputs: {ROOT / 'outputs'}")
    print(
        "Educational project only; results are not validated "
        "for clinical or medical decision-making."
    )


if __name__ == "__main__":
    run_pipeline()
