from pathlib import Path

from src.rag_index import build_index, load_index, retrieve_scored

def test_temperature_excursion_retrieves_both_required_actions():
    """Both excursion review and prolonged-deviation quarantine should be retrieved."""
    root = Path(__file__).resolve().parents[1]

    index = build_index(
        root / "data" / "mrna_safety_guidelines.txt",
        root / "outputs" / "chroma_store",
    )

    results, _ = retrieve_scored(
        "What happens after a temperature excursion?",
        index,
        k=3,
    )

    top_two_text = " ".join(
        result["chunk"].lower()
        for result in results[:2]
    )

    assert "documented and reviewed" in top_two_text
    assert "quarantined pending quality review" in top_two_text

def test_rag_index_can_be_reloaded_from_disk(tmp_path):
    """A persisted ChromaDB index should remain searchable after reload."""
    root = Path(__file__).resolve().parents[1]
    source_path = root / "data" / "mrna_safety_guidelines.txt"
    persist_dir = tmp_path / "chroma_store"

    # Build and persist a fresh index in a temporary directory.
    build_index(source_path, persist_dir)

    # Reload the saved index instead of rebuilding it.
    reloaded_index = load_index(persist_dir)

    results, _ = retrieve_scored(
        "What happens after a temperature excursion?",
        reloaded_index,
        k=3,
    )

    assert results
    retrieved_text = " ".join(
        result["chunk"].lower() for result in results
    )
    assert "temperature" in retrieved_text