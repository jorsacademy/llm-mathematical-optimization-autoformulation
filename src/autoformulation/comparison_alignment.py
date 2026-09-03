"""Deterministic one-to-one alignment of candidate and reference variables."""

from __future__ import annotations

import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from autoformulation.comparison_models import VariableAlignment, VariablePair
from autoformulation.schema import ModelSpec, VariableSpec

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "units",
    "value",
    "variable",
    "ve",
    "bir",
    "icin",
    "ile",
}


def _tokens(*parts: str | None) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        if not part:
            continue
        normalized = part.casefold().replace("ı", "i")
        tokens.update(token for token in _WORD_RE.findall(normalized) if token not in _STOPWORDS)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _number_similarity(left: float | None, right: float | None) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    scale = max(abs(left), abs(right), 1.0)
    return max(0.0, 1.0 - abs(left - right) / scale)


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _variable_score(
    candidate: VariableSpec,
    reference: VariableSpec,
    candidate_objective: dict[str, float],
    reference_objective: dict[str, float],
) -> float:
    score = 0.0
    score += 0.25 if candidate.name == reference.name else 0.0
    score += 0.20 if candidate.variable_type is reference.variable_type else 0.0
    score += 0.10 * _number_similarity(candidate.lower_bound, reference.lower_bound)
    score += 0.10 * _number_similarity(candidate.upper_bound, reference.upper_bound)
    score += 0.20 * _number_similarity(
        candidate_objective.get(candidate.name, 0.0),
        reference_objective.get(reference.name, 0.0),
    )
    score += 0.10 * _jaccard(
        _tokens(candidate.description),
        _tokens(reference.description),
    )
    score += 0.05 * _jaccard(
        _tokens(candidate.unit, candidate.source_excerpt),
        _tokens(reference.unit, reference.source_excerpt),
    )
    return min(max(score, 0.0), 1.0)


def align_variables(
    reference: ModelSpec,
    candidate: ModelSpec,
    *,
    minimum_pair_score: float = 0.55,
    ambiguity_margin: float = 0.04,
) -> VariableAlignment:
    if not 0 <= minimum_pair_score <= 1:
        raise ValueError("minimum_pair_score must be between 0 and 1")
    if not 0 <= ambiguity_margin <= 1:
        raise ValueError("ambiguity_margin must be between 0 and 1")

    if not reference.variables or not candidate.variables:
        return VariableAlignment(
            unmatched_candidate=candidate.variable_names(),
            unmatched_reference=reference.variable_names(),
            mean_score=0.0,
            minimum_score=0.0,
            reliable=False,
        )

    candidate_objective = candidate.objective.expression.coefficient_map()
    reference_objective = reference.objective.expression.coefficient_map()
    scores = np.asarray(
        [
            [
                _variable_score(item, gold, candidate_objective, reference_objective)
                for gold in reference.variables
            ]
            for item in candidate.variables
        ],
        dtype=float,
    )
    row_indices, column_indices = linear_sum_assignment(1.0 - scores)

    pairs: list[VariablePair] = []
    for row, column in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
        assigned = float(scores[row, column])
        alternatives = [float(value) for idx, value in enumerate(scores[row]) if idx != column]
        ambiguous = bool(
            alternatives
            and candidate.variables[row].name != reference.variables[column].name
            and assigned - max(alternatives) < ambiguity_margin
        )
        pairs.append(
            VariablePair(
                candidate=candidate.variables[row].name,
                reference=reference.variables[column].name,
                score=assigned,
                ambiguous=ambiguous,
            )
        )

    matched_candidate = {pair.candidate for pair in pairs}
    matched_reference = {pair.reference for pair in pairs}
    unmatched_candidate = [
        name for name in candidate.variable_names() if name not in matched_candidate
    ]
    unmatched_reference = [
        name for name in reference.variable_names() if name not in matched_reference
    ]
    pair_scores = [pair.score for pair in pairs]
    mean_score = sum(pair_scores) / len(pair_scores)
    minimum_score = min(pair_scores)
    reliable = bool(
        not unmatched_candidate
        and not unmatched_reference
        and minimum_score >= minimum_pair_score
        and not any(pair.ambiguous for pair in pairs)
    )
    return VariableAlignment(
        pairs=pairs,
        unmatched_candidate=unmatched_candidate,
        unmatched_reference=unmatched_reference,
        mean_score=mean_score,
        minimum_score=minimum_score,
        reliable=reliable,
    )


__all__ = ["align_variables"]
