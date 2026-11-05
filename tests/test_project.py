from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attention import encode_sequence, self_attention, plot_attention
from src.data_pipeline import MissingRequiredColumnError, load_and_clean
from src.rag_index import chunk_text, retrieve_scored
from src.schemas import ClinicalSafetyReport, DISCLAIMER

ROOT = Path(__file__).resolve().parents[1]


def test_cleaning_removes_all_missing_values():
    cleaned = load_and_clean(ROOT / "data" / "patient_records.csv")
    assert not cleaned.isna().any().any()


def test_missing_required_column_is_named(tmp_path):
    original = pd.read_csv(
        ROOT / "data" / "patient_records.csv"
    ).drop(columns=["Age"])

    path = tmp_path / "missing_age.csv"
    original.to_csv(path, index=False)

    with pytest.raises(MissingRequiredColumnError, match="Age"):
        load_and_clean(path)


def test_attention_validates_and_normalizes_rows():
    embeddings = encode_sequence("ACGUACGU")
    weights, _, _ = self_attention(embeddings)

    assert weights.shape == (8, 8)
    assert np.allclose(weights.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="Invalid nucleotide"):
        encode_sequence("ACGXU")


def test_schema_accepts_required_disclaimer():
    report = ClinicalSafetyReport(
        Eligible_For_Vaccine=False,
        Recommended_Dose_mg=0.1,
        Reasoning_Steps=["Insufficient evidence."],
        Retrieved_Safety_Reference_Excerpts=["Source excerpt."],
        Disclaimer=DISCLAIMER,
    )

    assert report.Disclaimer == DISCLAIMER


def test_schema_rejects_missing_disclaimer():
    with pytest.raises(ValueError, match="Disclaimer"):
        ClinicalSafetyReport(
            Eligible_For_Vaccine=False,
            Recommended_Dose_mg=0.1,
            Reasoning_Steps=["Insufficient evidence."],
            Retrieved_Safety_Reference_Excerpts=["Source excerpt."],
            Disclaimer="",
        )


def test_chunk_configuration_is_valid():
    from src.rag_index import CHUNK_SIZE, CHUNK_OVERLAP

    assert CHUNK_SIZE == 400
    assert CHUNK_OVERLAP == 50
    assert 0 < CHUNK_OVERLAP < CHUNK_SIZE


def test_chunk_text_returns_content():
    chunks = chunk_text(
        "Section 1 Storage\n\n"
        + "Keep the product refrigerated. " * 100
    )

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)
    assert any("refrigerated" in chunk.lower() for chunk in chunks)


def test_retrieval_rejects_empty_question():
    with pytest.raises(ValueError, match="Question"):
        retrieve_scored(
            "   ",
            index={},
        )


def test_retrieval_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        retrieve_scored(
            "What is storage?",
            index={},
            k=0,
        )



def test_attention_accepts_minimum_and_maximum_sequence_lengths():
    assert encode_sequence("A" * 5).shape == (5, 8)
    assert encode_sequence("A" * 50).shape == (50, 8)


@pytest.mark.parametrize("sequence", ["ACGU", "A" * 51])
def test_attention_rejects_out_of_range_sequence_lengths(sequence):
    with pytest.raises(ValueError, match="length"):
        encode_sequence(sequence)


def test_attention_outputs_have_expected_shapes_and_finite_values():
    embeddings = encode_sequence("ACGUACGU")
    weights, output, (queries, keys, values) = self_attention(embeddings)

    assert weights.shape == (8, 8)
    assert output.shape == (8, 8)
    assert queries.shape == keys.shape == values.shape == (8, 8)
    assert np.isfinite(weights).all()
    assert np.isfinite(output).all()
    assert np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_attention_plot_is_created(tmp_path):
    sequence = "ACGUACGU"
    embeddings = encode_sequence(sequence)
    weights, _, _ = self_attention(embeddings)
    image_path = tmp_path / "attention_heatmap.png"

    plot_attention(weights, sequence, image_path)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
