"""From-scratch scaled dot-product self-attention for a toy mRNA sequence."""

from pathlib import Path
import numpy as np

VOCABULARY = ("A", "C", "G", "U")
EMBED_DIM = 8
RANDOM_SEED = 42
_generator = np.random.default_rng(RANDOM_SEED)
EMBEDDINGS = {base: _generator.normal(size=EMBED_DIM) for base in VOCABULARY}
WEIGHTS = tuple(_generator.normal(scale=0.1, size=(EMBED_DIM, EMBED_DIM)) for _ in range(3))

def encode_sequence(sequence: str) -> np.ndarray:
    sequence = sequence.upper()
    if not 5 <= len(sequence) <= 50:
        raise ValueError("Sequence length must be between 5 and 50 bases.")
    invalid = sorted(set(sequence) - set(VOCABULARY))
    if invalid:
        raise ValueError(f"Invalid nucleotide(s): {', '.join(invalid)}. Allowed bases: A, C, G, U.")
    return np.stack([EMBEDDINGS[base] for base in sequence])

def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)

def self_attention(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    query_weights, key_weights, value_weights = WEIGHTS
    queries, keys, values = embeddings @ query_weights, embeddings @ key_weights, embeddings @ value_weights
    scores = (queries @ keys.T) / np.sqrt(keys.shape[1])
    attention_weights = softmax(scores)
    if not np.allclose(attention_weights.sum(axis=1), 1.0):
        raise AssertionError("Softmax rows must sum to 1.")
    return attention_weights, attention_weights @ values, (queries, keys, values)

def plot_attention(attention_weights: np.ndarray, sequence: str, save_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    save = Path(save_path)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(attention_weights, cmap="viridis")
    labels = [f"{index + 1}:{base}" for index, base in enumerate(sequence)]
    axis.set_xticks(range(len(sequence)), labels, rotation=90)
    axis.set_yticks(range(len(sequence)), labels)
    axis.set_xlabel("Attending to sequence position")
    axis.set_ylabel("Attending from sequence position")
    axis.set_title("mRNA Sequence Self Attention Weights")
    figure.colorbar(image, ax=axis, label="Attention weight")
    figure.tight_layout()
    figure.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(figure)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_sequence = "ACGUACGUACGGCUA"
    matrix = encode_sequence(sample_sequence)
    weights, _, _ = self_attention(matrix)
    plot_attention(weights, sample_sequence, project_root / "outputs" / "figures" / "attention_heatmap.png")
