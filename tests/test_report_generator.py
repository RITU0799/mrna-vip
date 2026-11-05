import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.report_generator import (
    build_report_with_retry,
    save_report,
)
from src.schemas import ClinicalSafetyReport


def make_valid_report():
    """Return a valid educational report for testing."""
    return ClinicalSafetyReport(
        Eligible_For_Vaccine=False,
        Recommended_Dose_mg=0.0,
        Reasoning_Steps=[
            "This is an educational test report.",
            "The output is not a clinical decision.",
        ],
        Retrieved_Safety_Reference_Excerpts=[
            "Verify relevant safety information against authoritative sources."
        ],
        Disclaimer=(
            "Student engineering project output only. "
            "Not real clinical, medical, or regulatory guidance."
        ),
    )


def make_validation_error():
    """Generate a genuine Pydantic validation error."""
    invalid_data = {
        "Eligible_For_Vaccine": False,
        "Recommended_Dose_mg": -1.0,
        "Reasoning_Steps": ["Test reasoning"],
        "Retrieved_Safety_Reference_Excerpts": ["Test warning"],
        "Disclaimer": (
            "Student engineering project output only. "
            "Not real clinical, medical, or regulatory guidance."
        ),
    }

    with pytest.raises(ValidationError) as exc_info:
        ClinicalSafetyReport.model_validate(invalid_data)

    return exc_info.value


def test_report_retry_succeeds_after_first_validation_error():
    """The report builder retries once after validation fails."""
    validation_error = make_validation_error()
    valid_report = make_valid_report()

    with patch(
        "src.report_generator.grounded_report",
        side_effect=[validation_error, valid_report],
    ) as mock_builder:
        result = build_report_with_retry(
            patient={"Patient_ID": "TEST001"},
            risk_label="Low",
            risk_probability=0.25,
            chunks=[],
        )

    assert result == valid_report
    assert mock_builder.call_count == 2


def test_report_retry_stops_after_two_failed_attempts():
    """A second validation failure must raise an error."""
    validation_error = make_validation_error()

    with patch(
        "src.report_generator.grounded_report",
        side_effect=[validation_error, validation_error],
    ) as mock_builder:
        with pytest.raises(RuntimeError, match="after one retry"):
            build_report_with_retry(
                patient={"Patient_ID": "TEST001"},
                risk_label="Low",
                risk_probability=0.25,
                chunks=[],
            )

    assert mock_builder.call_count == 2


def test_save_report_creates_json_and_markdown_files(tmp_path):
    """A valid report should be saved in both required formats."""
    report = make_valid_report()

    json_path, markdown_path = save_report(
        report=report,
        report_dir=tmp_path,
        stem="test_report",
    )

    assert Path(json_path).is_file()
    assert Path(markdown_path).is_file()

    saved_json = json.loads(Path(json_path).read_text(encoding="utf-8"))

    assert saved_json["Eligible_For_Vaccine"] is False
    assert "Disclaimer" in saved_json
    assert "not real clinical" in saved_json["Disclaimer"].lower()

    saved_markdown = Path(markdown_path).read_text(encoding="utf-8")
    assert saved_markdown.strip()
    assert "Disclaimer" in saved_markdown


def test_grounded_report_extracts_warnings_from_all_chunks():
    from src.report_generator import grounded_report

    patient = {
        "Vaccine_Dose_mcg": 50,
        "Biomarker_Level": 40.0,
        "Pre_Existing_Conditions": 0,
        "Age": 30,
        "Prior_Reaction_History": "None",
    }

    chunks = [
        "First source warning: monitor the recorded temperature.",
        "Second source warning: document the adverse event.",
        "Third source warning: verify the source information.",
    ]

    report = grounded_report(
        patient=patient,
        risk_label="Low",
        risk_probability=0.25,
        chunks=chunks,
    )

    warnings_text = " ".join(report.Retrieved_Safety_Reference_Excerpts)

    assert "First source warning" in warnings_text
    assert "Second source warning" in warnings_text
    assert "Third source warning" in warnings_text



def test_llm_report_guardrails_force_educational_values():
    """LLM output cannot set eligibility or change the recorded dose."""
    from unittest.mock import patch
    from src.report_generator import build_llm_report_with_retry

    candidate = ClinicalSafetyReport(
        Eligible_For_Vaccine=True,
        Recommended_Dose_mg=99.0,
        Reasoning_Steps=["The model generated this summary."],
        Retrieved_Safety_Reference_Excerpts=[
            "This exact sentence is present in the source."
        ],
        Disclaimer=(
            "Student engineering project output only. "
            "Not real clinical, medical, or regulatory guidance."
        ),
    )
    patient = {"Vaccine_Dose_mcg": 50}
    chunks = ["This exact sentence is present in the source."]

    with patch(
        "src.report_generator._invoke_llm_structured",
        return_value=candidate,
    ):
        report = build_llm_report_with_retry(
            patient=patient,
            risk_label="Low",
            risk_probability=0.2,
            chunks=chunks,
        )

    assert report.Eligible_For_Vaccine is False
    assert report.Recommended_Dose_mg == 0.05
    assert report.Retrieved_Safety_Reference_Excerpts == chunks
    assert report.Disclaimer == (
        "Student engineering project output only. "
        "Not real clinical, medical, or regulatory guidance."
    )


def test_llm_report_discards_ungrounded_warning():
    from unittest.mock import patch
    from src.report_generator import build_llm_report_with_retry

    candidate = make_valid_report()
    candidate.Retrieved_Safety_Reference_Excerpts = ["Unsupported invented warning"]

    with patch(
        "src.report_generator._invoke_llm_structured",
        return_value=candidate,
    ):
        report = build_llm_report_with_retry(
            patient={"Vaccine_Dose_mcg": 25},
            risk_label="Low",
            risk_probability=0.1,
            chunks=["A source excerpt about storage."],
        )

    assert report.Retrieved_Safety_Reference_Excerpts == [
        "No directly matching warning excerpt was verified "
        "in the retrieved educational source."
    ]
