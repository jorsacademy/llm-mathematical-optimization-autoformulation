"""Typed comparison results and semantic error taxonomy."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from autoformulation.schema import StrictModel


class SemanticErrorCode(StrEnum):
    STATIC_INVALID = "static_invalid"
    SOLVER_FAILURE = "solver_failure"
    VARIABLE_ALIGNMENT_UNRELIABLE = "variable_alignment_unreliable"
    MISSING_VARIABLE = "missing_variable"
    EXTRA_VARIABLE = "extra_variable"
    VARIABLE_TYPE_MISMATCH = "variable_type_mismatch"
    VARIABLE_BOUND_MISMATCH = "variable_bound_mismatch"
    OBJECTIVE_SENSE_MISMATCH = "objective_sense_mismatch"
    OBJECTIVE_COEFFICIENT_MISMATCH = "objective_coefficient_mismatch"
    OBJECTIVE_CONSTANT_MISMATCH = "objective_constant_mismatch"
    CONSTRAINT_OMISSION = "constraint_omission"
    EXTRA_CONSTRAINT = "extra_constraint"
    CONSTRAINT_MISMATCH = "constraint_mismatch"
    CANDIDATE_DECISION_INFEASIBLE_IN_REFERENCE = (
        "candidate_decision_infeasible_in_reference"
    )
    REFERENCE_DECISION_INFEASIBLE_IN_CANDIDATE = (
        "reference_decision_infeasible_in_candidate"
    )
    DECISION_OBJECTIVE_GAP = "decision_objective_gap"
    UNJUSTIFIED_ASSUMPTION = "unjustified_assumption"
    UNRESOLVED_QUESTION = "unresolved_question"


class SemanticIssue(StrictModel):
    code: SemanticErrorCode
    message: str
    location: str | None = None


class VariablePair(StrictModel):
    candidate: str
    reference: str
    score: float = Field(ge=0, le=1)
    ambiguous: bool = False


class VariableAlignment(StrictModel):
    pairs: list[VariablePair] = Field(default_factory=list)
    unmatched_candidate: list[str] = Field(default_factory=list)
    unmatched_reference: list[str] = Field(default_factory=list)
    mean_score: float = Field(ge=0, le=1)
    minimum_score: float = Field(ge=0, le=1)
    reliable: bool

    def mapping(self) -> dict[str, str]:
        return {pair.candidate: pair.reference for pair in self.pairs}


class VariableComparison(StrictModel):
    matched: int = Field(ge=0)
    reference_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    type_accuracy: float = Field(ge=0, le=1)
    lower_bound_accuracy: float = Field(ge=0, le=1)
    upper_bound_accuracy: float = Field(ge=0, le=1)
    all_domains_match: bool


class ObjectiveComparison(StrictModel):
    sense_match: bool
    coefficient_linf: float | None = Field(default=None, ge=0)
    constant_error: float = Field(ge=0)
    exact_match: bool


class ConstraintPairComparison(StrictModel):
    candidate: str
    reference: str
    distance: float = Field(ge=0)
    relation_match: bool
    coefficient_linf: float = Field(ge=0)
    rhs_error: float = Field(ge=0)
    exact_match: bool


class ConstraintComparison(StrictModel):
    reference_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    pairs: list[ConstraintPairComparison] = Field(default_factory=list)
    omitted_reference: list[str] = Field(default_factory=list)
    extra_candidate: list[str] = Field(default_factory=list)


class ModelComparison(StrictModel):
    exact_fingerprint_match: bool
    alignment: VariableAlignment
    variables: VariableComparison
    objective: ObjectiveComparison
    constraints: ConstraintComparison
    candidate_valid: bool
    candidate_solved: bool
    reference_solved: bool
    candidate_decision_feasible_in_reference: bool | None = None
    reference_decision_feasible_in_candidate: bool | None = None
    reference_objective_value: float | None = None
    candidate_native_objective_value: float | None = None
    candidate_decision_reference_objective: float | None = None
    candidate_decision_in_reference_space: dict[str, float] = Field(default_factory=dict)
    reference_decision_in_candidate_space: dict[str, float] = Field(default_factory=dict)
    reference_decision_gap_percent: float | None = Field(default=None, ge=0)
    native_objective_value_gap_percent: float | None = Field(default=None, ge=0)
    structural_match: bool
    behavioral_match: bool
    strict_match: bool
    issues: list[SemanticIssue] = Field(default_factory=list)


__all__ = [
    "ConstraintComparison",
    "ConstraintPairComparison",
    "ModelComparison",
    "ObjectiveComparison",
    "SemanticErrorCode",
    "SemanticIssue",
    "VariableAlignment",
    "VariableComparison",
    "VariablePair",
]
