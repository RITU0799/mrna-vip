"""Generate structured educational safety summaries from model output and retrieved text."""

from pathlib import Path
import json
import re
import time

import joblib
import pandas as pd
from pydantic import ValidationError

import os

from dotenv import load_dotenv

from src.rag_index import load_index, query_guidelines
from src.schemas import ClinicalSafetyReport, DISCLAIMER


def score_patient(
    patient: dict,
    model_path: str | Path,
) -> tuple[str, float]:
    """Return the experimental model label and probability."""

    model = joblib.load(model_path)
    frame = pd.DataFrame([patient])
    probability = float(model.predict_proba(frame)[0, 1])

    label = "High" if probability >= 0.5 else "Low"
    return label, probability


def extract_explicit_max_dose_mcg(
    chunks: list[str],
) -> float | None:
    """Extract a maximum dose only when explicitly stated in retrieved text.

    A secondary-review threshold is not treated as a maximum safe dose.
    """

    patterns = [
        (
            r"\bmaximum\s+(?:safe\s+)?dose\s*"
            r"(?:is|of|:|=)?\s*(\d+(?:\.\d+)?)\s*"
            r"(micrograms?|mcg|µg)\b"
        ),
        (
            r"\bmax(?:imum)?\s+dose\s*"
            r"(?:is|of|:|=)?\s*(\d+(?:\.\d+)?)\s*"
            r"(micrograms?|mcg|µg)\b"
        ),
    ]

    for chunk in chunks:
        for pattern in patterns:
            match = re.search(
                pattern,
                chunk,
                flags=re.IGNORECASE,
            )
            if match:
                return float(match.group(1))

    return None


def grounded_report(
    patient: dict,
    risk_label: str,
    risk_probability: float,
    chunks: list[str],
) -> ClinicalSafetyReport:
    """Create a conservative summary using the supplied course reference."""

    dose_mcg = float(patient["Vaccine_Dose_mcg"])
    biomarker = float(patient["Biomarker_Level"])
    conditions = int(patient["Pre_Existing_Conditions"])
    age = int(patient["Age"])

    prior_reaction = str(
        patient["Prior_Reaction_History"]
    ).strip().title()

    source = " ".join(chunks).lower()

    # These flags follow the course-project reference.
    # They are not clinically validated decision rules.
    conditions_review = conditions >= 3
    prior_reaction_review = prior_reaction == "Moderate"
    older_patient_review = age >= 65 and conditions >= 2
    young_patient_review = age < 5 and conditions >= 1

    # A source-defined maximum is checked only when explicitly stated.
    # The 100 mcg secondary-review threshold is not a maximum safe dose.
    explicit_max_dose_mcg = extract_explicit_max_dose_mcg(chunks)

    exceeds_explicit_maximum = (
        explicit_max_dose_mcg is not None
        and dose_mcg > explicit_max_dose_mcg
    )

    dose_review = (
        dose_mcg > 100
        and (
            "biomarker" in source
            or "inflammatory" in source
        )
    )

    # The supplied source does not specify a numerical biomarker threshold.
    biomarker_threshold_unavailable = True

    review_reasons = []

    if conditions_review:
        review_reasons.append(
            "The supplied course-project reference flags three or more "
            "pre-existing conditions for secondary review."
        )

    if prior_reaction_review:
        review_reasons.append(
            "The supplied course-project reference flags a prior moderate "
            "reaction to an mRNA-based product for secondary review."
        )

    if older_patient_review:
        review_reasons.append(
            "The supplied course-project reference flags patients aged "
            "65 or older with two or more relevant pre-existing conditions."
        )

    if young_patient_review:
        review_reasons.append(
            "The supplied course-project reference flags patients under "
            "age 5 with any pre-existing condition for secondary review."
        )

    if exceeds_explicit_maximum:
        review_reasons.append(
            f"The candidate dose of {dose_mcg:.1f} micrograms exceeds "
            f"the explicitly stated maximum of "
            f"{explicit_max_dose_mcg:.1f} micrograms in the retrieved "
            "text. Eligibility remains false in this educational prototype. "
            "This source-based check is not clinical advice."
        )

    if dose_review:
        review_reasons.append(
            "The candidate dose exceeds 100 micrograms and the retrieved "
            "text mentions biomarker-related information. The source's "
            "numerical threshold for an elevated biomarker is unspecified, "
            "so the combined condition cannot be verified."
        )
    elif dose_mcg > 100:
        review_reasons.append(
            "The candidate dose exceeds 100 micrograms. The supplied "
            "reference calls for secondary review for doses above this "
            "level. This is a secondary-review threshold, not a universal "
            "maximum safe dose."
        )

    if biomarker_threshold_unavailable:
        review_reasons.append(
            f"The recorded biomarker level is {biomarker:.3f}, but the "
            "supplied reference does not define a numerical threshold for "
            "an elevated biomarker. Its status cannot be classified "
            "reliably by this prototype."
        )

    reasoning = [
        (
            f"The experimental classifier returned a '{risk_label}' label "
            f"with a model score of {risk_probability:.3f}. This score is "
            "not a clinically validated estimate of individual risk."
        ),
        (
            f"The source record lists a candidate dose of {dose_mcg:.1f} "
            f"micrograms ({dose_mcg / 1000:.4f} mg). This is a unit "
            "conversion only, not a dose recommendation."
        ),
    ]

    if review_reasons:
        reasoning.extend(review_reasons)
    else:
        reasoning.append(
            "No matching review condition was identified from the "
            "available record fields. This does not establish eligibility "
            "or safety."
        )

    reasoning.append(
        "The supplied course-project reference is not a verified clinical "
        "protocol. A qualified clinician must use current, product-specific "
        "official guidance for real medical decisions."
    )

    # This prototype cannot establish clinical eligibility.
    # Any explicit maximum-dose exceedance is also recorded above.
    eligible = False

    # Required schema compatibility; this is not a recommendation.
    dose_mg = round(dose_mcg / 1000, 4)

    # Extract unique sentences from every retrieved chunk.
    # Preserve all source warnings, even when a sentence lacks a
    # predefined keyword, so relevant evidence is not silently discarded.
    warnings = []
    seen = set()

    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        normalized = re.sub(r"\s+-\s+", "-", normalized)

        sentences = re.split(r"(?<=[.!?])\s+", normalized)

        for sentence in sentences:
            sentence = sentence.strip(" \t\n>-")
            if not sentence:
                continue

            key = sentence.casefold()
            if key in seen:
                continue

            seen.add(key)
            warnings.append(sentence)

    # Keep the report concise while retaining source order.
    warnings = warnings[:6]

    if not warnings:
        warnings = [
            "No relevant guideline excerpt was retrieved. "
            "Source verification is required."
        ]

    return ClinicalSafetyReport(
        Eligible_For_Vaccine=eligible,
        Recommended_Dose_mg=dose_mg,
        Reasoning_Steps=reasoning,
        FDA_Safety_Warnings_Cited=warnings,
        Disclaimer=DISCLAIMER,
    )


def build_report_with_retry(
    patient: dict,
    risk_label: str,
    risk_probability: float,
    chunks: list[str],
) -> ClinicalSafetyReport:
    """Validate the report, retrying once if schema validation fails."""

    last_error = None

    for attempt in range(2):
        try:
            return grounded_report(
                patient=patient,
                risk_label=risk_label,
                risk_probability=risk_probability,
                chunks=chunks,
            )
        except ValidationError as exc:
            last_error = exc

            if attempt == 0:
                continue

    raise RuntimeError(
        "Report validation failed after one retry; no report was saved."
    ) from last_error



def _invoke_llm_structured(
    patient: dict,
    risk_label: str,
    risk_probability: float,
    chunks: list[str],
    retry_error: str | None = None,
) -> ClinicalSafetyReport:
    """Call an OpenAI-compatible chat model through LangChain.

    The LLM is constrained to summarize supplied inputs. Deterministic
    post-processing below prevents it from making an eligibility decision
    or changing the recorded candidate dose.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM mode requires langchain-openai. Install requirements.txt "
            "and configure OPENAI_API_KEY."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to a local .env file; "
            "never commit the key."
        )

    model_name = os.getenv("MRNA_VIP_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    ).with_structured_output(ClinicalSafetyReport)

    context = "\n\n--- RETRIEVED EDUCATIONAL SOURCE ---\n\n".join(chunks)
    prompt = f"""
You are producing a student-project engineering demonstration, NOT medical
advice. Use ONLY the patient record, experimental model output, and retrieved
educational excerpts below. Do not use outside knowledge. Do not invent
citations, thresholds, contraindications, or recommendations.

Return a ClinicalSafetyReport with exactly these fields:
Eligible_For_Vaccine (boolean), Recommended_Dose_mg (number),
Reasoning_Steps (non-empty list of strings),
FDA_Safety_Warnings_Cited (non-empty list of strings), Disclaimer (string).

Mandatory constraints:
- Set Eligible_For_Vaccine to false. This prototype cannot establish eligibility.
- Recommended_Dose_mg is only the recorded Vaccine_Dose_mcg divided by 1000;
  it is a unit conversion, NOT a recommendation.
- State that the classifier score is experimental and not clinically validated.
- Do not describe a secondary-review threshold as a maximum safe dose.
- If the retrieved text does not support a claim, say that it is unspecified.
- Keep warning citations limited to verbatim or near-verbatim information in
  the retrieved excerpts. If none are relevant, say no relevant excerpt was
  retrieved and source verification is required.
- Disclaimer must be exactly: {DISCLAIMER}

Patient record (JSON-like): {json.dumps(patient, default=str)}
Experimental classifier label: {risk_label}
Experimental classifier probability: {risk_probability:.6f}
Retrieved excerpts:
{context}
"""
    if retry_error:
        prompt += (
            "\n\nYour previous response failed schema validation. Correct "
            "the validation problem while preserving every mandatory constraint. "
            f"Validation feedback: {retry_error[:1200]}"
        )

    result = llm.invoke(prompt)
    if isinstance(result, ClinicalSafetyReport):
        return result
    return ClinicalSafetyReport.model_validate(result)


def build_llm_report_with_retry(
    patient: dict,
    risk_label: str,
    risk_probability: float,
    chunks: list[str],
) -> ClinicalSafetyReport:
    """Generate an LLM report and retry exactly once on schema validation."""
    last_error = None
    for attempt in range(2):
        try:
            candidate = _invoke_llm_structured(
                patient=patient,
                risk_label=risk_label,
                risk_probability=risk_probability,
                chunks=chunks,
                retry_error=str(last_error) if last_error else None,
            )
            # Deterministic guardrails: the LLM cannot determine eligibility
            # or alter the candidate dose recorded in the input dataset.
            dose_mcg = float(patient["Vaccine_Dose_mcg"])
            safe_data = candidate.model_dump()
            safe_data["Eligible_For_Vaccine"] = False
            safe_data["Recommended_Dose_mg"] = round(dose_mcg / 1000, 4)
            safe_data["Disclaimer"] = DISCLAIMER

            # Only retain warning text supported by the retrieved source.
            source = " ".join(chunks).casefold()
            grounded_warnings = [
                warning for warning in candidate.FDA_Safety_Warnings_Cited
                if warning.casefold() in source
            ]
            safe_data["FDA_Safety_Warnings_Cited"] = (
                grounded_warnings[:6] or [
                    "No directly matching warning excerpt was verified "
                    "in the retrieved educational source."
                ]
            )
            safe_data["Reasoning_Steps"] = (
                candidate.Reasoning_Steps[:8]
                + [
                    "This is an educational software output only. "
                    "It does not establish clinical eligibility or provide "
                    "a dose recommendation."
                ]
            )
            return ClinicalSafetyReport.model_validate(safe_data)
        except ValidationError as exc:
            last_error = exc
            if attempt == 0:
                continue
    raise RuntimeError(
        "LLM report validation failed after one retry; no report was saved."
    ) from last_error


def save_report(
    report: ClinicalSafetyReport,
    report_dir: str | Path,
    stem: str = "sample_report",
) -> tuple[Path, Path]:
    """Save the validated report as JSON and Markdown."""

    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / f"{stem}.json"
    markdown_path = destination / f"{stem}.md"

    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = [
        "# Educational Safety Summary",
        "",
        "**Clinical eligibility established:** No - this prototype "
        "cannot determine eligibility.",
        (
            f"**Candidate dose recorded in source data:** "
            f"{report.Recommended_Dose_mg:.4f} mg "
            "(unit conversion only; not a recommendation)"
        ),
        "",
        "## Reasoning and limitations",
    ]

    markdown.extend(
        f"- {item}" for item in report.Reasoning_Steps
    )

    markdown.extend(
        ["", "## Retrieved course-project reference excerpts"]
    )

    markdown.extend(
        f"> {item}" for item in report.FDA_Safety_Warnings_Cited
    )

    markdown.extend(
        ["", "## Disclaimer", report.Disclaimer, ""]
    )

    markdown_path.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    return json_path, markdown_path


def run_sample_report(
    project_root: Path,
    use_llm: bool | None = None,
) -> ClinicalSafetyReport:
    """Generate the sample report and record pipeline timings."""

    project_root = Path(project_root)
    total_started = time.perf_counter()
    load_dotenv(project_root / ".env", override=False)

    patient_path = (
        project_root / "data" / "patient_records_clean.csv"
    )

    patient = pd.read_csv(patient_path).iloc[0].to_dict()

    score_started = time.perf_counter()

    label, probability = score_patient(
        patient,
        project_root / "outputs" / "risk_model.joblib",
    )

    scoring_seconds = time.perf_counter() - score_started

    retrieval_started = time.perf_counter()

    index = load_index(
        project_root / "outputs" / "chroma_store"
    )

    query = (
        "secondary review contraindications dose and biomarker "
        f"{patient['Vaccine_Dose_mcg']} micrograms "
        f"age {patient['Age']} prior reaction monitoring"
    )

    chunks, retrieval_seconds = query_guidelines(
        query,
        index=index,
        k=3,
    )

    retrieval_wall_seconds = time.perf_counter() - retrieval_started

    if use_llm is None:
        use_llm = os.getenv("MRNA_VIP_USE_LLM", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

    report_started = time.perf_counter()
    if use_llm:
        report = build_llm_report_with_retry(
            patient=patient,
            risk_label=label,
            risk_probability=probability,
            chunks=chunks,
        )
        generation_mode = "langchain_openai"
    else:
        report = build_report_with_retry(
            patient=patient,
            risk_label=label,
            risk_probability=probability,
            chunks=chunks,
        )
        generation_mode = "offline_rule_based"

    save_report(
        report,
        project_root / "outputs" / "reports",
    )

    report_build_seconds = time.perf_counter() - report_started
    total_seconds = time.perf_counter() - total_started

    latency = {
        "report_generation_mode": generation_mode,
        "retrieval_seconds": round(retrieval_seconds, 6),
        "retrieval_stage_seconds": round(retrieval_wall_seconds, 6),
        "model_scoring_seconds": round(scoring_seconds, 6),
        "report_generation_seconds": round(report_build_seconds, 6),
        "total_pipeline_seconds": round(total_seconds, 6),
    }

    latency_path = (
        project_root
        / "outputs"
        / "report_generation_latency.json"
    )

    latency_path.parent.mkdir(parents=True, exist_ok=True)

    latency_path.write_text(
        json.dumps(latency, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an educational mRNA-VIP report."
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the LangChain/OpenAI-compatible LLM (requires OPENAI_API_KEY).",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    result = run_sample_report(project_root, use_llm=args.llm)

    print("Sample educational report generated.")
    print(
        "Clinical eligibility established: "
        f"{result.Eligible_For_Vaccine}"
    )
    print(f"Disclaimer: {result.Disclaimer}")
