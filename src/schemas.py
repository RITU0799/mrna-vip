"""Validated output schema for generated student-project safety summaries."""
from typing import List
from pydantic import BaseModel, Field, field_validator

DISCLAIMER = "Student engineering project output only. Not real clinical, medical, or regulatory guidance."

class ClinicalSafetyReport(BaseModel):
    Eligible_For_Vaccine: bool
    Recommended_Dose_mg: float = Field(ge=0)
    Reasoning_Steps: List[str] = Field(min_length=1)
    Retrieved_Safety_Reference_Excerpts: List[str] = Field(min_length=1)
    Disclaimer: str = DISCLAIMER

    @field_validator("Disclaimer")
    @classmethod
    def require_disclaimer(cls, value: str) -> str:
        if "not real clinical" not in value.lower() and "not real medical" not in value.lower():
            raise ValueError("Disclaimer must state that this is not real clinical guidance.")
        return value
