"""Scale-normalized comparison of linear constraints."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from autoformulation.comparison_models import (
    ConstraintComparison,
    ConstraintPairComparison,
    VariableAlignment,
)
from autoformulation.schema import ConstraintSpec, ModelSpec, Relation


def _normalized_constraint(
    constraint: ConstraintSpec,
    *,
    mapping: dict[str, str],
    reference_order: list[str],
) -> tuple[Relation, np.ndarray, float, float]:
    coefficients = np.zeros(len(reference_order), dtype=float)
    reference_index = {name: position for position, name in enumerate(reference_order)}
    unmapped_magnitude = 0.0
    for name, coefficient in constraint.lhs.coefficient_map().items():
        mapped = mapping.get(name)
        if mapped is None or mapped not in reference_index:
            unmapped_magnitude += abs(coefficient)
        else:
            coefficients[reference_index[mapped]] += coefficient

    relation = constraint.relation
    rhs = constraint.rhs - constraint.lhs.constant
    if relation is Relation.GE:
        coefficients *= -1.0
        rhs *= -1.0
        relation = Relation.LE
    elif relation is Relation.EQ:
        signature = [*coefficients.tolist(), rhs]
        first_nonzero = next((value for value in signature if not math.isclose(value, 0.0)), 0.0)
        if first_nonzero < 0:
            coefficients *= -1.0
            rhs *= -1.0

    scale = float(
        max(
            max((abs(value) for value in coefficients), default=0.0),
            abs(rhs),
            unmapped_magnitude,
            1e-12,
        )
    )
    return relation, coefficients / scale, rhs / scale, unmapped_magnitude / scale


def _compare_constraint_pair(
    candidate: ConstraintSpec,
    reference: ConstraintSpec,
    *,
    mapping: dict[str, str],
    reference_order: list[str],
    tolerance: float,
) -> ConstraintPairComparison:
    candidate_relation, candidate_coefficients, candidate_rhs, unmapped = _normalized_constraint(
        candidate,
        mapping=mapping,
        reference_order=reference_order,
    )
    reference_relation, reference_coefficients, reference_rhs, _ = _normalized_constraint(
        reference,
        mapping={name: name for name in reference_order},
        reference_order=reference_order,
    )
    coefficient_linf = float(
        np.max(np.abs(candidate_coefficients - reference_coefficients), initial=0.0)
    )
    rhs_error = abs(candidate_rhs - reference_rhs)
    relation_match = candidate_relation is reference_relation
    distance = max(coefficient_linf, rhs_error, unmapped, 0.0 if relation_match else 1.0)
    return ConstraintPairComparison(
        candidate=candidate.name,
        reference=reference.name,
        distance=distance,
        relation_match=relation_match,
        coefficient_linf=coefficient_linf,
        rhs_error=rhs_error,
        exact_match=distance <= tolerance,
    )


def compare_constraints(
    reference: ModelSpec,
    candidate: ModelSpec,
    alignment: VariableAlignment,
    *,
    tolerance: float = 1e-7,
    diagnostic_pair_cutoff: float = 0.35,
) -> ConstraintComparison:
    if tolerance < 0 or diagnostic_pair_cutoff < 0:
        raise ValueError("constraint tolerances must be nonnegative")

    if not reference.constraints or not candidate.constraints:
        exact_matches = 0
        precision = 1.0 if not candidate.constraints and not reference.constraints else 0.0
        recall = precision
        return ConstraintComparison(
            reference_count=len(reference.constraints),
            candidate_count=len(candidate.constraints),
            exact_matches=exact_matches,
            precision=precision,
            recall=recall,
            f1=precision,
            omitted_reference=[constraint.name for constraint in reference.constraints],
            extra_candidate=[constraint.name for constraint in candidate.constraints],
        )

    mapping = alignment.mapping()
    matrix: list[list[ConstraintPairComparison]] = [
        [
            _compare_constraint_pair(
                item,
                gold,
                mapping=mapping,
                reference_order=reference.variable_names(),
                tolerance=tolerance,
            )
            for gold in reference.constraints
        ]
        for item in candidate.constraints
    ]
    costs = np.asarray([[pair.distance for pair in row] for row in matrix], dtype=float)
    rows, columns = linear_sum_assignment(costs)
    assigned = [matrix[row][column] for row, column in zip(rows, columns, strict=True)]
    exact_pairs = [pair for pair in assigned if pair.exact_match]
    diagnostic_pairs = [pair for pair in assigned if pair.distance <= diagnostic_pair_cutoff]
    exact_candidate = {pair.candidate for pair in exact_pairs}
    exact_reference = {pair.reference for pair in exact_pairs}
    omitted = [
        constraint.name
        for constraint in reference.constraints
        if constraint.name not in exact_reference
    ]
    extra = [
        constraint.name
        for constraint in candidate.constraints
        if constraint.name not in exact_candidate
    ]
    precision = len(exact_pairs) / len(candidate.constraints)
    recall = len(exact_pairs) / len(reference.constraints)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return ConstraintComparison(
        reference_count=len(reference.constraints),
        candidate_count=len(candidate.constraints),
        exact_matches=len(exact_pairs),
        precision=precision,
        recall=recall,
        f1=f1,
        pairs=diagnostic_pairs,
        omitted_reference=omitted,
        extra_candidate=extra,
    )


__all__ = ["compare_constraints"]
