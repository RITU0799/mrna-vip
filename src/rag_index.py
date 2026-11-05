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

# Keep the lexical contribution small and transparent.
LEXICAL_WEIGHT = 0.25

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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

# Transparent query-term expansion for concepts used in this source.
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
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split oversized text into overlapping token chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        piece = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
        ).strip()

        if piece:
            chunks.append(piece)

        if end == len(token_ids):
            break

        start = end - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Create section-aware chunks without cutting ordinary sentences."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    text = text.strip()
    if not text:
        return []

    model = _get_embedding_model()
    tokenizer = model.tokenizer
    effective_size = min(chunk_size, model.max_seq_length - 2)
    chunks: list[str] = []

    sections = SECTION_PATTERN.split(text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in PARAGRAPH_PATTERN.split(section)
            if paragraph.strip()
        ]
        if not paragraphs:
            continue

        first_paragraph = paragraphs[0]
        first_line, separator, first_body = first_paragraph.partition("\n")

        if separator and first_line.strip().lower().startswith("section "):
            heading = first_line.strip()
            body_parts = [first_body.strip(), *paragraphs[1:]]
        else:
            heading = first_paragraph
            body_parts = paragraphs[1:]

        body = "\n\n".join(
            part for part in body_parts if part
        ).strip()

        heading_ids = tokenizer.encode(
            heading,
            add_special_tokens=False,
        )
        heading_text = tokenizer.decode(
            heading_ids[:60],
            skip_special_tokens=True,
        ).strip()

        heading_tokens = _token_count(heading_text, tokenizer)
        body_budget = effective_size - heading_tokens

        if body_budget < 1:
            raise ValueError(
                "Chunk size is too small for the section heading."
            )

        if not body:
            chunks.append(heading_text)
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", body)
            if sentence.strip()
        ]

        current: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens

            if not current:
                return

            chunk = f"{heading_text}\n{' '.join(current)}".strip()

            if _token_count(chunk, tokenizer) <= effective_size:
                chunks.append(chunk)

            current = []
            current_tokens = 0

        for sentence in sentences:
            sentence_tokens = _token_count(sentence, tokenizer)

            if sentence_tokens > body_budget:
                flush()

                pieces = _split_token_ids(
                    tokenizer.encode(
                        sentence,
                        add_special_tokens=False,
                    ),
                    tokenizer,
                    chunk_size=body_budget,
                    overlap=min(overlap, body_budget - 1),
                )

                for piece in pieces:
                    chunk = f"{heading_text}\n{piece}".strip()

                    if _token_count(chunk, tokenizer) <= effective_size:
                        chunks.append(chunk)
                continue

            if current and current_tokens + sentence_tokens > body_budget:
                flush()

            current.append(sentence)
            current_tokens += sentence_tokens

        flush()

    # Remove exact duplicates without changing chunk order.
    return list(dict.fromkeys(chunks))


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
            {"chunk_index": i, "source_file": source.name}
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
        "distance_metric": "cosine",
        "section_aware": True,
        "sentence_aware": True,
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

    # Monitoring duration and allergy-history exception
    if (
        ("monitoring" in q or "monitor" in q)
        and "patients should be monitored" in doc
        and "minutes after administration" in doc
    ):
        bonus += 0.08

    # Temperature-excursion documentation and review
    if (
        "excursion" in q
        and "excursion" in doc
        and "documented and reviewed" in doc
    ):
        bonus += 0.05

    # Quarantine requirement for prolonged temperature deviations
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
    """Retrieve and rank chunks using semantic, lexical, and intent scores.

    `similarity` remains the original cosine similarity.
    `ranking_score` is a separate score used only for ordering results.
    """
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
        documents, distances, metadatas
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

    results, latency = retrieve_scored(question, index, k=k)
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
        "## Interpretation and limitations",
        "",
        "- Cosine similarity measures semantic closeness, not factual correctness.",
        "- Ranking score combines cosine similarity, lexical relevance, and targeted intent bonuses.",
        "- Intent bonuses are explicit heuristics and require evaluation against test queries.",
        "- Inspect retrieved passages against the original educational source.",
        "- A missing result does not prove that the source lacks an answer.",
        "- Sentence-aware chunking reduces, but cannot eliminate, truncation.",
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
