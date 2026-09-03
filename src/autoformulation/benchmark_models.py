"""Versioned artifacts for reproducible autoformulation benchmarks."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformulation.comparison_models import ModelComparison
from autoformulation.schema import ModelSpec, StrictModel

METHODOLOGY_VERSION = "1.0.0"


class VariantKind(StrEnum):
    CANONICAL = "canonical"
    PARAPHRASE = "paraphrase"
    ADVERSARIAL = "adversarial"
    AMBIGUOUS = "ambiguous"


class ExpectedOutcome(StrEnum):
    FORMULATION = "formulation"
    ABSTENTION = "abstention"


class FormulationPassPolicy(StrEnum):
    STRICT = "strict"
    BEHAVIORAL = "behavioral"


class BenchmarkSuiteCase(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1, max_length=128)
    family_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1, max_length=128)
    variant: VariantKind
    statement: str = Field(min_length=1, max_length=100_000)
    expected_outcome: ExpectedOutcome = ExpectedOutcome.FORMULATION
    reference_model_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]+$",
        max_length=128,
    )
    tags: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def expectation_has_reference(self) -> BenchmarkSuiteCase:
        if self.expected_outcome is ExpectedOutcome.FORMULATION and not self.reference_model_id:
            raise ValueError("formulation cases require reference_model_id")
        if (
            self.expected_outcome is ExpectedOutcome.ABSTENTION
            and self.reference_model_id is not None
        ):
            raise ValueError("abstention cases must not declare reference_model_id")
        return self


class BenchmarkSuite(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    reference_models: dict[str, ModelSpec] = Field(default_factory=dict)
    cases: list[BenchmarkSuiteCase] = Field(min_length=1)

    @model_validator(mode="after")
    def references_and_ids_are_consistent(self) -> BenchmarkSuite:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        for case in self.cases:
            if (
                case.reference_model_id is not None
                and case.reference_model_id not in self.reference_models
            ):
                raise ValueError(
                    f"case '{case.id}' references unknown model '{case.reference_model_id}'"
                )
        return self

    def reference_for(self, case: BenchmarkSuiteCase) -> ModelSpec | None:
        reference_model_id = case.reference_model_id
        if reference_model_id is None:
            return None
        return self.reference_models[reference_model_id]


class SystemDescriptor(StrictModel):
    system_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=256)
    prompt_version: str | None = Field(default=None, max_length=128)
    repair_rounds: int = Field(ge=0, le=5)
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoringConfig(StrictModel):
    coefficient_tolerance: float = Field(default=1e-7, ge=0)
    objective_tolerance_percent: float = Field(default=1e-5, ge=0)
    minimum_alignment_score: float = Field(default=0.55, ge=0, le=1)
    solve_time_limit_seconds: float = Field(default=60.0, gt=0)
    formulation_pass_policy: FormulationPassPolicy = FormulationPassPolicy.STRICT


class PredictionRecord(StrictModel):
    case_id: str
    completed: bool
    model: ModelSpec | None = None
    valid: bool = False
    validation_codes: list[str] = Field(default_factory=list)
    repair_attempts: int = Field(default=0, ge=0, le=5)
    latency_seconds: float = Field(ge=0)
    error_message: str | None = None

    @model_validator(mode="after")
    def completion_state_is_consistent(self) -> PredictionRecord:
        if self.completed and self.model is None:
            raise ValueError("completed predictions require a model")
        if not self.completed and self.model is not None:
            raise ValueError("failed predictions must not contain a model")
        if self.model is None and self.valid:
            raise ValueError("a prediction without a model cannot be valid")
        return self


class BenchmarkRun(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    methodology_version: Literal["1.0.0"] = "1.0.0"
    suite_id: str
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system: SystemDescriptor
    predictions: list[PredictionRecord]

    @model_validator(mode="after")
    def prediction_ids_are_unique(self) -> BenchmarkRun:
        ids = [prediction.case_id for prediction in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("prediction case IDs must be unique")
        return self


class BenchmarkIssue(StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str


class BenchmarkCaseScore(StrictModel):
    case_id: str
    family_id: str
    variant: VariantKind
    expected_outcome: ExpectedOutcome
    tags: list[str] = Field(default_factory=list)
    prediction_completed: bool
    candidate_valid: bool
    candidate_solved: bool
    abstained: bool
    expectation_pass: bool
    behavioral_match: bool = False
    structural_match: bool = False
    strict_match: bool = False
    latency_seconds: float | None = Field(default=None, ge=0)
    comparison: ModelComparison | None = None
    issues: list[BenchmarkIssue] = Field(default_factory=list)


class MetricSlice(StrictModel):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0, le=1)


class BenchmarkSummaryV2(StrictModel):
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    formulation_cases: int = Field(ge=0)
    abstention_cases: int = Field(ge=0)
    expectation_passes: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    expectation_pass_rate: float = Field(ge=0, le=1)
    valid_formulation_rate: float | None = Field(default=None, ge=0, le=1)
    solve_rate: float | None = Field(default=None, ge=0, le=1)
    behavioral_match_rate: float | None = Field(default=None, ge=0, le=1)
    structural_match_rate: float | None = Field(default=None, ge=0, le=1)
    strict_match_rate: float | None = Field(default=None, ge=0, le=1)
    abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    family_robustness_rate: float = Field(ge=0, le=1)
    paraphrase_retention_rate: float | None = Field(default=None, ge=0, le=1)
    adversarial_retention_rate: float | None = Field(default=None, ge=0, le=1)
    mean_reference_decision_gap_percent: float | None = Field(default=None, ge=0)
    mean_latency_seconds: float | None = Field(default=None, ge=0)
    by_variant: dict[str, MetricSlice] = Field(default_factory=dict)
    by_tag: dict[str, MetricSlice] = Field(default_factory=dict)
    error_taxonomy: dict[str, int] = Field(default_factory=dict)


class BenchmarkReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    methodology_version: Literal["1.0.0"] = "1.0.0"
    suite_id: str
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scoring: ScoringConfig
    system: SystemDescriptor
    summary: BenchmarkSummaryV2
    cases: list[BenchmarkCaseScore]


def _canonical_sha256(value: StrictModel) -> str:
    canonical = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def suite_fingerprint(suite: BenchmarkSuite) -> str:
    """Return a stable fingerprint over statements, gold models, labels, and metadata."""

    return _canonical_sha256(suite)


def run_fingerprint(run: BenchmarkRun) -> str:
    """Return a stable fingerprint over the raw provider/model run artifact."""

    return _canonical_sha256(run)


def load_suite(path: str | Path) -> BenchmarkSuite:
    source = Path(path)
    return BenchmarkSuite.model_validate_json(source.read_text(encoding="utf-8"))


def load_run(path: str | Path) -> BenchmarkRun:
    source = Path(path)
    return BenchmarkRun.model_validate_json(source.read_text(encoding="utf-8"))


def load_report(path: str | Path) -> BenchmarkReport:
    source = Path(path)
    return BenchmarkReport.model_validate_json(source.read_text(encoding="utf-8"))


def save_artifact(value: StrictModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


__all__ = [
    "BenchmarkCaseScore",
    "BenchmarkIssue",
    "BenchmarkReport",
    "BenchmarkRun",
    "BenchmarkSuite",
    "BenchmarkSuiteCase",
    "BenchmarkSummaryV2",
    "ExpectedOutcome",
    "FormulationPassPolicy",
    "METHODOLOGY_VERSION",
    "MetricSlice",
    "PredictionRecord",
    "ScoringConfig",
    "SystemDescriptor",
    "VariantKind",
    "load_report",
    "load_run",
    "load_suite",
    "run_fingerprint",
    "save_artifact",
    "suite_fingerprint",
]
