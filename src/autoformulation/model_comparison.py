"""Deterministic, solver-grounded comparison of finite LP/MILP formulations."""

from __future__ import annotations

from autoformulation.comparison_alignment import (
    _close,
    _jaccard,
    _number_similarity,
    align_variables,
)
from autoformulation.comparison_constraints import compare_constraints
from autoformulation.comparison_models import (
    ConstraintComparison,
    ConstraintPairComparison,
    ModelComparison,
    ObjectiveComparison,
    SemanticErrorCode,
    SemanticIssue,
    VariableAlignment,
    VariableComparison,
    VariablePair,
)
from autoformulation.pipeline import model_fingerprint
from autoformulation.schema import ModelSpec, ObjectiveSense
from autoformulation.solver import (
    SolveOptions,
    check_solution,
    evaluate_expression,
    solve_model,
)


def _compare_variables(
    reference: ModelSpec,
    candidate: ModelSpec,
    alignment: VariableAlignment,
    *,
    tolerance: float,
) -> VariableComparison:
    reference_by_name = {variable.name: variable for variable in reference.variables}
    candidate_by_name = {variable.name: variable for variable in candidate.variables}
    if not alignment.pairs:
        return VariableComparison(
            matched=0,
            reference_count=len(reference.variables),
            candidate_count=len(candidate.variables),
            type_accuracy=0.0,
            lower_bound_accuracy=0.0,
            upper_bound_accuracy=0.0,
            all_domains_match=False,
        )

    type_matches = 0
    lower_matches = 0
    upper_matches = 0
    for pair in alignment.pairs:
        item = candidate_by_name[pair.candidate]
        gold = reference_by_name[pair.reference]
        type_matches += item.variable_type is gold.variable_type
        lower_matches += _close(item.lower_bound, gold.lower_bound, tolerance)
        upper_matches += _close(item.upper_bound, gold.upper_bound, tolerance)
    count = len(alignment.pairs)
    full = not alignment.unmatched_candidate and not alignment.unmatched_reference
    return VariableComparison(
        matched=count,
        reference_count=len(reference.variables),
        candidate_count=len(candidate.variables),
        type_accuracy=type_matches / count,
        lower_bound_accuracy=lower_matches / count,
        upper_bound_accuracy=upper_matches / count,
        all_domains_match=bool(
            full and type_matches == count and lower_matches == count and upper_matches == count
        ),
    )


def _compare_objective(
    reference: ModelSpec,
    candidate: ModelSpec,
    alignment: VariableAlignment,
    *,
    tolerance: float,
) -> ObjectiveComparison:
    mapping = alignment.mapping()
    reference_coefficients = reference.objective.expression.coefficient_map()
    mapped_candidate: dict[str, float] = {}
    unmapped_magnitude = 0.0
    for name, coefficient in candidate.objective.expression.coefficient_map().items():
        mapped = mapping.get(name)
        if mapped is None:
            unmapped_magnitude += abs(coefficient)
        else:
            mapped_candidate[mapped] = mapped_candidate.get(mapped, 0.0) + coefficient
    differences = [
        abs(mapped_candidate.get(name, 0.0) - reference_coefficients.get(name, 0.0))
        for name in reference.variable_names()
    ]
    coefficient_linf = max([*differences, unmapped_magnitude], default=0.0)
    constant_error = abs(
        candidate.objective.expression.constant - reference.objective.expression.constant
    )
    sense_match = candidate.objective.sense is reference.objective.sense
    exact = bool(
        alignment.reliable
        and sense_match
        and coefficient_linf <= tolerance
        and constant_error <= tolerance
    )
    return ObjectiveComparison(
        sense_match=sense_match,
        coefficient_linf=coefficient_linf,
        constant_error=constant_error,
        exact_match=exact,
    )


def _relative_gap(left: float, right: float) -> float:
    return 100.0 * abs(left - right) / max(abs(right), 1e-12)


def compare_models(
    reference: ModelSpec,
    candidate: ModelSpec,
    *,
    coefficient_tolerance: float = 1e-7,
    objective_tolerance_percent: float = 1e-5,
    minimum_alignment_score: float = 0.55,
    solve_time_limit_seconds: float = 60.0,
) -> ModelComparison:
    if coefficient_tolerance < 0 or objective_tolerance_percent < 0:
        raise ValueError("comparison tolerances must be nonnegative")
    if solve_time_limit_seconds <= 0:
        raise ValueError("solve_time_limit_seconds must be positive")

    alignment = align_variables(
        reference,
        candidate,
        minimum_pair_score=minimum_alignment_score,
    )
    variables = _compare_variables(
        reference,
        candidate,
        alignment,
        tolerance=coefficient_tolerance,
    )
    objective = _compare_objective(
        reference,
        candidate,
        alignment,
        tolerance=coefficient_tolerance,
    )
    constraints = compare_constraints(
        reference,
        candidate,
        alignment,
        tolerance=coefficient_tolerance,
    )

    solve_options = SolveOptions(time_limit=solve_time_limit_seconds)
    reference_solution = solve_model(reference, solve_options)
    candidate_solution = solve_model(candidate, solve_options)
    reference_objective_value = reference_solution.objective_value
    candidate_objective_value = candidate_solution.objective_value
    candidate_valid = candidate_solution.validation.ok
    issues: list[SemanticIssue] = []
    if not candidate_valid:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.STATIC_INVALID,
                message="Candidate failed deterministic static validation.",
            )
        )
    if candidate_valid and not candidate_solution.success:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.SOLVER_FAILURE,
                message=f"Candidate solver status is {candidate_solution.status.value}.",
            )
        )
    if not alignment.reliable:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.VARIABLE_ALIGNMENT_UNRELIABLE,
                message="Deterministic variable alignment was incomplete or ambiguous.",
            )
        )
    if alignment.unmatched_reference:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.MISSING_VARIABLE,
                message="Reference variables were not represented: "
                + ", ".join(alignment.unmatched_reference),
            )
        )
    if alignment.unmatched_candidate:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.EXTRA_VARIABLE,
                message="Candidate contains unmatched variables: "
                + ", ".join(alignment.unmatched_candidate),
            )
        )
    if variables.type_accuracy < 1.0:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.VARIABLE_TYPE_MISMATCH,
                message="At least one aligned variable has the wrong domain type.",
            )
        )
    if variables.lower_bound_accuracy < 1.0 or variables.upper_bound_accuracy < 1.0:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.VARIABLE_BOUND_MISMATCH,
                message="At least one aligned variable has different bounds.",
            )
        )
    if not objective.sense_match:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.OBJECTIVE_SENSE_MISMATCH,
                message="Objective sense differs from the reference model.",
                location="objective.sense",
            )
        )
    if objective.coefficient_linf is None or objective.coefficient_linf > coefficient_tolerance:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.OBJECTIVE_COEFFICIENT_MISMATCH,
                message="Objective coefficients differ after variable alignment.",
                location="objective.expression",
            )
        )
    if objective.constant_error > coefficient_tolerance:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.OBJECTIVE_CONSTANT_MISMATCH,
                message="Objective constant differs from the reference model.",
                location="objective.expression.constant",
            )
        )
    if constraints.omitted_reference:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.CONSTRAINT_OMISSION,
                message="Reference constraints not exactly matched: "
                + ", ".join(constraints.omitted_reference),
            )
        )
    if constraints.extra_candidate:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.EXTRA_CONSTRAINT,
                message="Candidate constraints not exactly matched: "
                + ", ".join(constraints.extra_candidate),
            )
        )
    if any(not pair.exact_match for pair in constraints.pairs):
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.CONSTRAINT_MISMATCH,
                message="At least one structurally similar constraint has different semantics.",
            )
        )
    if candidate.assumptions:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.UNJUSTIFIED_ASSUMPTION,
                message=f"Candidate declares {len(candidate.assumptions)} modeling assumption(s).",
                location="assumptions",
            )
        )
    if candidate.unresolved_questions:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.UNRESOLVED_QUESTION,
                message=(
                    f"Candidate reports {len(candidate.unresolved_questions)} "
                    "unresolved question(s)."
                ),
                location="unresolved_questions",
            )
        )

    candidate_feasible_in_reference: bool | None = None
    reference_feasible_in_candidate: bool | None = None
    candidate_reference_objective: float | None = None
    candidate_as_reference: dict[str, float] = {}
    reference_as_candidate: dict[str, float] = {}
    decision_gap: float | None = None
    native_objective_gap: float | None = None
    if (
        alignment.reliable
        and reference_solution.success
        and candidate_solution.success
        and reference_objective_value is not None
        and candidate_objective_value is not None
    ):
        mapping = alignment.mapping()
        candidate_as_reference = {
            reference_name: candidate_solution.variable_values[candidate_name]
            for candidate_name, reference_name in mapping.items()
        }
        reference_as_candidate = {
            candidate_name: reference_solution.variable_values[reference_name]
            for candidate_name, reference_name in mapping.items()
        }
        candidate_check = check_solution(reference, candidate_as_reference)
        reference_check = check_solution(candidate, reference_as_candidate)
        candidate_feasible_in_reference = candidate_check.feasible
        reference_feasible_in_candidate = reference_check.feasible
        native_objective_gap = _relative_gap(
            candidate_objective_value,
            reference_objective_value,
        )
        if candidate_check.feasible:
            candidate_reference_objective = evaluate_expression(
                reference.objective.expression,
                candidate_as_reference,
            )
            if reference.objective.sense is ObjectiveSense.MAXIMIZE:
                degradation = reference_objective_value - candidate_reference_objective
            else:
                degradation = candidate_reference_objective - reference_objective_value
            decision_gap = (
                100.0
                * max(degradation, 0.0)
                / max(abs(reference_objective_value), 1e-12)
            )

    if candidate_feasible_in_reference is False:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.CANDIDATE_DECISION_INFEASIBLE_IN_REFERENCE,
                message="Candidate optimum violates the reference formulation.",
            )
        )
    if reference_feasible_in_candidate is False:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.REFERENCE_DECISION_INFEASIBLE_IN_CANDIDATE,
                message="Reference optimum violates the candidate formulation.",
            )
        )
    if decision_gap is not None and decision_gap > objective_tolerance_percent:
        issues.append(
            SemanticIssue(
                code=SemanticErrorCode.DECISION_OBJECTIVE_GAP,
                message=(
                    f"Candidate decision has {decision_gap:.6g}% degradation under the "
                    "reference objective."
                ),
            )
        )

    structural_match = bool(
        alignment.reliable
        and variables.all_domains_match
        and objective.exact_match
        and constraints.precision == 1.0
        and constraints.recall == 1.0
    )
    behavioral_match = bool(
        candidate_feasible_in_reference is True
        and reference_feasible_in_candidate is True
        and decision_gap is not None
        and decision_gap <= objective_tolerance_percent
    )
    return ModelComparison(
        exact_fingerprint_match=model_fingerprint(reference) == model_fingerprint(candidate),
        alignment=alignment,
        variables=variables,
        objective=objective,
        constraints=constraints,
        candidate_valid=candidate_valid,
        candidate_solved=candidate_solution.success,
        reference_solved=reference_solution.success,
        candidate_decision_feasible_in_reference=candidate_feasible_in_reference,
        reference_decision_feasible_in_candidate=reference_feasible_in_candidate,
        reference_objective_value=reference_objective_value,
        candidate_native_objective_value=candidate_objective_value,
        candidate_decision_reference_objective=candidate_reference_objective,
        candidate_decision_in_reference_space=candidate_as_reference,
        reference_decision_in_candidate_space=reference_as_candidate,
        reference_decision_gap_percent=decision_gap,
        native_objective_value_gap_percent=native_objective_gap,
        structural_match=structural_match,
        behavioral_match=behavioral_match,
        strict_match=structural_match and behavioral_match,
        issues=issues,
    )


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
    "align_variables",
    "compare_constraints",
    "compare_models",
    "_compare_variables",
    "_jaccard",
    "_number_similarity",
]
