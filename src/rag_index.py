"""Persistent, section-aware retrieval for the mRNA-VIP educational project.

Retrieved text is not clinical or regulatory advice.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import time

import chromadb
from sentence_transformers import SentenceTransformer


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MIN_SIMILARITY = 0.01

LEXICAL_WEIGHT = 0.25

EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
COLLECTION_NAME = "mrna_safety_guidelines"

SECTION_PATTERN = re.compile(
    r"(?=^Section\s+\d+\b)",
    flags=re.MULTILINE | re.IGNORECASE,
)
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does",
    "for", "from", "happens", "how", "in", "is", "it", "of",
    "on", "or", "the", "this", "to", "what", "when", "which",
    "who", "why", "with", "within", "after", "require", "requires",
    "required", "longer",
}

QUERY_EXPANSIONS = {
    "monitoring": {
        "monitor", "monitored", "monitoring", "minutes",
        "allergic", "allergy", "reaction", "patients",
        "history", "15", "30",
    },
    "temperature": {
        "temperature", "excursion", "cold", "chain",
        "documented", "reviewed", "quarantine", "deviation",
    },
    "excursion": {
        "excursion", "deviation", "temperature", "quarantine",
        "quality", "review", "batch", "documented", "hour",
    },
    "quarantine": {
        "quarantine", "deviation", "excursion", "batch",
        "quality", "review", "temperature", "hour",
    },
    "allergic": {
        "allergic", "allergy", "reaction", "monitoring",
        "minutes", "patients", "history", "monitored",
    },
    "allergy": {
        "allergic", "allergy", "reaction", "monitoring",
        "minutes", "patients", "history", "monitored",
    },
    "dose": {
        "dose", "dosing", "micrograms", "secondary",
        "review", "candidate", "exceeding", "clinician",
    },
    "documented": {
        "documented", "documentation", "onset", "duration",
        "severity", "treatment", "adverse", "event",
    },
}


def _get_embedding_model() -> SentenceTransformer:
    """Load the sentence-embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _token_count(text: str, tokenizer) -> int:
    """Return token count without special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_token_ids(
    token_ids: list[int],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    """Return balanced token windows with the requested overlap."""
    total_tokens = len(token_ids)

    if total_tokens == 0:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    if total_tokens <= chunk_size:
        return [(0, total_tokens)]

    # Distribute tokens evenly rather than leaving a short final chunk.
    n_chunks = (total_tokens - overlap + (chunk_size - overlap) - 1) // (
        chunk_size - overlap
    )

    total_window_tokens = total_tokens + overlap * (n_chunks - 1)
    base_size, remainder = divmod(total_window_tokens, n_chunks)

    if base_size < 300 or base_size + (1 if remainder else 0) > chunk_size:
        raise ValueError(
            "Cannot divide this document into compliant chunks "
            "with the configured chunk size and overlap."
        )

    spans = []
    start = 0

    for index in range(n_chunks):
        current_size = base_size + (1 if index < remainder else 0)
        end = min(start + current_size, total_tokens)
        spans.append((start, end))
        start = end - overlap

    return spans


def chunk_text(text: str) -> list[str]:
    """Create balanced, overlapping chunks using the embedding tokenizer."""
    if not text or not text.strip():
        return []

    model = _get_embedding_model()
    tokenizer = model.tokenizer

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    spans = _split_token_ids(
        token_ids,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for index, (start, end) in enumerate(spans):
        chunk = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        actual_count = len(
            tokenizer.encode(chunk, add_special_tokens=False)
        )

        if not 300 <= actual_count <= CHUNK_SIZE:
            raise ValueError(
                f"Chunk {index + 1} contains {actual_count} tokens "
                f"after decoding. Expected 300 to {CHUNK_SIZE} tokens."
            )

        chunks.append(chunk)

    return chunks


def _make_client(persist_dir: str | Path):
    """Create a persistent ChromaDB client."""
    destination = Path(persist_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(destination))


def build_index(
    doc_path: str | Path,
    persist_dir: str | Path,
) -> dict:
    """Build and persist the vector index."""
    source = Path(doc_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Guideline source file not found: {source}"
        )

    raw = source.read_text(encoding="utf-8").strip()

    if not raw:
        raise ValueError("Guideline source file is empty.")

    chunks = chunk_text(raw)

    if not chunks:
        raise ValueError("No searchable text chunks were created.")

    model = _get_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    client = _make_client(persist_dir)

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as exc:
        message = str(exc).lower()

        if "does not exist" not in message and "not found" not in message:
            raise

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"guideline_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "chunk_index": i,
                "source_file": source.name,
            }
            for i in range(len(chunks))
        ],
    )

    destination = Path(persist_dir)

    metadata = {
        "source_file": source.name,
        "chunk_count": len(chunks),
        "chunk_size_tokens": CHUNK_SIZE,
        "overlap_tokens": CHUNK_OVERLAP,
        "backend": "ChromaDB persistent vector store",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": model.get_embedding_dimension(),
        "distance_metric": "cosine",
        "section_aware": False,
        "sentence_aware": False,
        "retrieval_ranking": (
            "cosine similarity, lexical relevance, and targeted intent bonus"
        ),
        "warning": (
            "Educational course-project reference only; "
            "not clinical or regulatory guidance."
        ),
    }

    (destination / "index_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "client": client,
        "collection": collection,
        "model": model,
        "chunks": chunks,
        "persist_dir": str(destination),
    }


def load_index(persist_dir: str | Path) -> dict:
    """Load an existing persistent index."""
    destination = Path(persist_dir)
    metadata_path = destination / "index_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Index metadata not found: {metadata_path}. Build the index first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
        raise ValueError(
            "The saved index uses a different embedding model. "
            "Rebuild the index before querying."
        )

    client = _make_client(destination)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found. "
            "Build the index first."
        ) from exc

    model = _get_embedding_model()
    stored = collection.get(include=["documents"])

    return {
        "client": client,
        "collection": collection,
        "model": model,
        "chunks": stored.get("documents") or [],
        "persist_dir": str(destination),
    }


def _terms(text: str) -> set[str]:
    """Normalize text into meaningful lowercase terms."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


def _lexical_relevance(question: str, document: str) -> float:
    """Return a bounded lexical relevance score between 0 and 1."""
    query_terms = _terms(question)
    document_terms = _terms(document)

    if not query_terms or not document_terms:
        return 0.0

    expanded_terms = set(query_terms)

    for term in query_terms:
        expanded_terms.update(QUERY_EXPANSIONS.get(term, set()))

    direct_matches = query_terms & document_terms
    expanded_matches = expanded_terms & document_terms

    direct_score = len(direct_matches) / len(query_terms)
    expanded_score = len(expanded_matches) / len(expanded_terms)

    return min(
        1.0,
        0.75 * direct_score + 0.25 * expanded_score,
    )


def _intent_bonus(question: str, document: str) -> float:
    """Boost passages that directly contain the action asked about."""
    q = question.lower()
    doc = document.lower()
    bonus = 0.0

    if (
        ("monitoring" in q or "monitor" in q)
        and "patients should be monitored" in doc
        and "minutes after administration" in doc
    ):
        bonus += 0.08

    if (
        "excursion" in q
        and "excursion" in doc
        and "documented and reviewed" in doc
    ):
        bonus += 0.05

    if (
        "excursion" in q
        and "quarantined pending quality review" in doc
    ):
        bonus += 0.12

    return bonus


def retrieve_scored(
    question: str,
    index: dict,
    k: int = 3,
    min_similarity: float = MIN_SIMILARITY,
) -> tuple[list[dict], float]:
    """Retrieve and rank chunks using semantic, lexical, and intent scores."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1.")

    started = time.perf_counter()

    collection = index["collection"]
    model = index["model"]
    count = collection.count()

    if count == 0:
        return [], time.perf_counter() - started

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    response = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=count,
        include=["documents", "distances", "metadatas"],
    )

    results = []

    documents = response.get("documents", [[]])[0]
    distances = response.get("distances", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas,
    ):
        similarity = max(
            0.0,
            min(1.0, 1.0 - float(distance)),
        )

        if similarity < min_similarity:
            continue

        lexical_score = _lexical_relevance(question, document)
        intent_bonus = _intent_bonus(question, document)

        ranking_score = (
            (1.0 - LEXICAL_WEIGHT) * similarity
            + LEXICAL_WEIGHT * lexical_score
            + intent_bonus
        )

        results.append({
            "chunk": document,
            "similarity": similarity,
            "lexical_score": lexical_score,
            "intent_bonus": intent_bonus,
            "ranking_score": ranking_score,
            "metadata": metadata or {},
        })

    results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    latency = time.perf_counter() - started

    return results[:k], latency


def query_guidelines(
    question: str,
    index: dict | None = None,
    persist_dir: str | Path | None = None,
    k: int = 3,
) -> tuple[list[str], float]:
    """Return retrieved chunks, preserving the existing interface."""
    if index is None:
        if persist_dir is None:
            raise ValueError("Provide an in-memory index or persist_dir.")

        index = load_index(persist_dir)

    results, latency = retrieve_scored(
        question,
        index,
        k=k,
    )

    return [item["chunk"] for item in results], latency


def run_retrieval_demo(project_root: Path) -> list[dict]:
    """Build the index, test five queries, and write the results report."""
    store = project_root / "outputs" / "chroma_store"
    source = project_root / "data" / "mrna_safety_guidelines.txt"

    index = build_index(source, store)

    questions = [
        "What refrigerated storage temperature is required?",
        "When does a high dose require secondary review?",
        "Which patients require a longer monitoring period?",
        "What adverse events must be documented within 24 hours?",
        "What happens after a temperature excursion?",
    ]

    results = []

    for question in questions:
        matches, latency = retrieve_scored(
            question,
            index,
            k=3,
        )

        results.append({
            "question": question,
            "matches": matches,
            "latency_seconds": latency,
        })

    lines = [
        "# Guideline Retrieval Tests",
        "",
        "> Educational project reference only. Not clinical or regulatory guidance.",
        "",
        "| Question | Rank | Cosine similarity | Lexical score | Intent bonus | Ranking score | Retrieved text | Latency (seconds) |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]

    for item in results:
        if not item["matches"]:
            lines.append(
                f"| {item['question']} | - | - | - | - | - | "
                f"No match above threshold | {item['latency_seconds']:.6f} |"
            )
            continue

        for rank, match in enumerate(item["matches"], start=1):
            preview = (
                match["chunk"]
                .replace("\n", " ")
                .replace("|", "/")
            )

            if len(preview) > 220:
                preview = preview[:220].rsplit(" ", 1)[0] + "..."

            lines.append(
                f"| {item['question']} | {rank} | "
                f"{match['similarity']:.4f} | "
                f"{match['lexical_score']:.4f} | "
                f"{match['intent_bonus']:.4f} | "
                f"{match['ranking_score']:.4f} | "
                f"{preview} | {item['latency_seconds']:.6f} |"
            )

    lines.extend([
        "",
        "## Chunking configuration",
        "",
        f"- Embedding model: `{EMBEDDING_MODEL_NAME}`",
        f"- Target chunk size: {CHUNK_SIZE} tokens",
        f"- Target overlap: {CHUNK_OVERLAP} tokens",
        f"- Actual indexed chunks: {len(index['chunks'])}",
        "",
        "## Interpretation and limitations",
        "",
        "- Cosine similarity measures semantic closeness, not factual correctness.",
        "- Ranking score combines cosine similarity, lexical relevance, and targeted intent bonuses.",
        "- Intent bonuses are explicit heuristics and require evaluation against test queries.",
        "- Inspect retrieved passages against the original educational source.",
        "- A missing result does not prove that the source lacks an answer.",
        "- Token-based chunk boundaries can occur within sentences.",
        "",
        "## Failure analysis",
        "",
        "### Monitoring period",
        "",
        "- Source check: Section 3 specifies at least 15 minutes of monitoring, "
        "or 30 minutes for patients with a documented history of any allergic reaction.",
        "- Evaluation: Check whether the retrieved passages contain this instruction.",
        "",
        "### Temperature excursion",
        "",
        "- Source check: Section 1 says excursions above -15C for formulations "
        "requiring ultra-low storage must be documented and reviewed. Deviations "
        "lasting longer than one hour require quarantine pending quality review.",
        "- Evaluation: Check whether the retrieved passages contain both actions.",
        "",
        "### Adverse-event documentation",
        "",
        "- Source check: Section 4 says moderate or severe events must be "
        "documented within 24 hours, including onset time, duration, severity, "
        "and treatment administered.",
        "",
        "These tests evaluate an educational retrieval prototype only. "
        "They do not establish clinical validity or support real-world medical decisions.",
    ])

    report_path = project_root / "outputs" / "retrieval_tests.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Indexed {len(index['chunks'])} chunks.")
    print(f"Report saved to: {report_path}")

    for item in results:
        best = item["matches"][0] if item["matches"] else None

        if best:
            print(
                f"\nQuestion: {item['question']}\n"
                f"Cosine similarity: {best['similarity']:.4f}\n"
                f"Lexical score: {best['lexical_score']:.4f}\n"
                f"Intent bonus: {best['intent_bonus']:.4f}\n"
                f"Ranking score: {best['ranking_score']:.4f}\n"
                f"Retrieved text: {best['chunk'][:180]}..."
            )
        else:
            print(
                f"\nQuestion: {item['question']}\n"
                "No sufficiently similar text found."
            )

    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_retrieval_demo(root)
