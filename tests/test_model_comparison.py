from __future__ import annotations

import pytest

from autoformulation.model_comparison import (
    SemanticErrorCode,
    VariableAlignment,
    _compare_variables,
    _jaccard,
    _number_similarity,
    align_variables,
    compare_constraints,
    compare_models,
)
from autoformulation.schema import ModelSpec


def production_model() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "production",
            "problem_summary": "production",
            "source_language": "en",
            "variables": [
                {
                    "name": "tables",
                    "description": "tables produced",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": None,
                    "unit": "items",
                },
                {
                    "name": "chairs",
                    "description": "chairs produced",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": None,
                    "unit": "items",
                },
            ],
            "objective": {
                "sense": "maximize",
                "description": "profit",
                "expression": {
                    "terms": [
                        {"variable": "tables", "coefficient": 40},
                        {"variable": "chairs", "coefficient": 30},
                    ]
                },
            },
            "constraints": [
                {
                    "name": "labor",
                    "description": "labor capacity",
                    "lhs": {
                        "terms": [
                            {"variable": "tables", "coefficient": 2},
                            {"variable": "chairs", "coefficient": 1},
                        ]
                    },
                    "relation": "<=",
                    "rhs": 100,
                },
                {
                    "name": "wood",
                    "description": "wood capacity",
                    "lhs": {
                        "terms": [
                            {"variable": "tables", "coefficient": 1},
                            {"variable": "chairs", "coefficient": 2},
                        ]
                    },
                    "relation": "<=",
                    "rhs": 80,
                },
            ],
            "assumptions": [],
            "unresolved_questions": [],
        }
    )


def mutate(model: ModelSpec, callback: object) -> ModelSpec:
    payload = model.model_dump(mode="json")
    callback(payload)  # type: ignore[operator]
    return ModelSpec.model_validate(payload)


def rename(model: ModelSpec) -> ModelSpec:
    mapping = {"tables": "x", "chairs": "y"}

    def change(payload: dict[str, object]) -> None:
        variables = payload["variables"]
        assert isinstance(variables, list)
        for variable in variables:
            assert isinstance(variable, dict)
            variable["name"] = mapping[str(variable["name"])]
        objective = payload["objective"]
        assert isinstance(objective, dict)
        expression = objective["expression"]
        assert isinstance(expression, dict)
        for term in expression["terms"]:  # type: ignore[index]
            term["variable"] = mapping[term["variable"]]
        constraints = payload["constraints"]
        assert isinstance(constraints, list)
        for constraint in constraints:
            for term in constraint["lhs"]["terms"]:
                term["variable"] = mapping[term["variable"]]

    return mutate(model, change)


def codes(comparison: object) -> set[SemanticErrorCode]:
    return {issue.code for issue in comparison.issues}  # type: ignore[attr-defined]


def test_equivalence_under_renaming_and_positive_scaling() -> None:
    reference = production_model()
    candidate = rename(reference)
    payload = candidate.model_dump(mode="json")
    first = payload["constraints"][0]
    for term in first["lhs"]["terms"]:
        term["coefficient"] *= 2
    first["rhs"] *= 2
    comparison = compare_models(reference, ModelSpec.model_validate(payload))
    assert comparison.structural_match
    assert comparison.behavioral_match
    assert not comparison.exact_fingerprint_match


def test_ge_and_equality_sign_normalization() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    first = payload["constraints"][0]
    first["relation"] = ">="
    first["rhs"] *= -1
    for term in first["lhs"]["terms"]:
        term["coefficient"] *= -1
    payload["constraints"][1]["relation"] = "=="
    reference_with_eq = ModelSpec.model_validate(payload)
    candidate_payload = reference_with_eq.model_dump(mode="json")
    second = candidate_payload["constraints"][1]
    second["rhs"] *= -1
    for term in second["lhs"]["terms"]:
        term["coefficient"] *= -1
    comparison = compare_models(reference_with_eq, ModelSpec.model_validate(candidate_payload))
    assert comparison.constraints.f1 == 1


def test_missing_constraint_and_restrictive_constraint_cross_checks() -> None:
    reference = production_model()
    missing_payload = reference.model_dump(mode="json")
    missing_payload["constraints"] = missing_payload["constraints"][:1]
    missing = compare_models(reference, ModelSpec.model_validate(missing_payload))
    assert SemanticErrorCode.CONSTRAINT_OMISSION in codes(missing)
    assert SemanticErrorCode.CANDIDATE_DECISION_INFEASIBLE_IN_REFERENCE in codes(missing)

    restrictive_payload = reference.model_dump(mode="json")
    restrictive_payload["constraints"].append(
        {
            "name": "no_tables",
            "description": "no tables",
            "lhs": {"terms": [{"variable": "tables", "coefficient": 1}]},
            "relation": "<=",
            "rhs": 0,
        }
    )
    restrictive = compare_models(reference, ModelSpec.model_validate(restrictive_payload))
    assert SemanticErrorCode.EXTRA_CONSTRAINT in codes(restrictive)
    assert SemanticErrorCode.REFERENCE_DECISION_INFEASIBLE_IN_CANDIDATE in codes(restrictive)


def test_objective_gap_and_objective_taxonomy() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    payload["objective"]["expression"]["terms"][1]["coefficient"] = 0
    payload["objective"]["expression"]["constant"] = 5
    candidate = ModelSpec.model_validate(payload)
    comparison = compare_models(reference, candidate)
    assert SemanticErrorCode.OBJECTIVE_COEFFICIENT_MISMATCH in codes(comparison)
    assert SemanticErrorCode.OBJECTIVE_CONSTANT_MISMATCH in codes(comparison)
    assert SemanticErrorCode.DECISION_OBJECTIVE_GAP in codes(comparison)
    assert comparison.reference_decision_gap_percent is not None

    payload["objective"]["sense"] = "minimize"
    sense = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.OBJECTIVE_SENSE_MISMATCH in codes(sense)


def test_domain_alignment_assumption_and_unresolved_taxonomy() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    payload["variables"][0]["variable_type"] = "integer"
    payload["variables"][1]["upper_bound"] = 10
    payload["assumptions"] = ["invented"]
    payload["unresolved_questions"] = ["missing"]
    comparison = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.STATIC_INVALID in codes(comparison)
    assert SemanticErrorCode.VARIABLE_TYPE_MISMATCH in codes(comparison)
    assert SemanticErrorCode.VARIABLE_BOUND_MISMATCH in codes(comparison)
    assert SemanticErrorCode.UNJUSTIFIED_ASSUMPTION in codes(comparison)
    assert SemanticErrorCode.UNRESOLVED_QUESTION in codes(comparison)


def test_extra_and_missing_variables_make_alignment_unreliable() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    payload["variables"].append(
        {
            "name": "unused",
            "description": "extra variable",
            "variable_type": "continuous",
            "lower_bound": None,
            "upper_bound": 10,
        }
    )
    payload["objective"]["expression"]["terms"].append(
        {"variable": "unused", "coefficient": 1}
    )
    extra = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.EXTRA_VARIABLE in codes(extra)
    assert SemanticErrorCode.VARIABLE_ALIGNMENT_UNRELIABLE in codes(extra)

    payload = reference.model_dump(mode="json")
    payload["variables"] = payload["variables"][:1]
    payload["objective"]["expression"]["terms"] = payload["objective"]["expression"][
        "terms"
    ][:1]
    for constraint in payload["constraints"]:
        constraint["lhs"]["terms"] = [
            term for term in constraint["lhs"]["terms"] if term["variable"] == "tables"
        ]
    missing = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.MISSING_VARIABLE in codes(missing)


def test_valid_but_infeasible_candidate_reports_solver_failure() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    payload["constraints"].append(
        {
            "name": "impossible",
            "description": "impossible",
            "lhs": {"terms": [{"variable": "tables", "coefficient": 1}]},
            "relation": "<=",
            "rhs": -1,
        }
    )
    comparison = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.SOLVER_FAILURE in codes(comparison)


def test_constraint_mismatch_and_empty_constraint_sets() -> None:
    reference = production_model()
    payload = reference.model_dump(mode="json")
    payload["constraints"][0]["rhs"] = 90
    mismatch = compare_models(reference, ModelSpec.model_validate(payload))
    assert SemanticErrorCode.CONSTRAINT_MISMATCH in codes(mismatch)

    no_constraints = reference.model_copy(update={"constraints": []})
    comparison = compare_models(no_constraints, no_constraints)
    assert comparison.constraints.precision == 1


def test_helper_validation_and_empty_alignment_paths() -> None:
    reference = production_model()
    with pytest.raises(ValueError):
        align_variables(reference, reference, minimum_pair_score=-1)
    with pytest.raises(ValueError):
        align_variables(reference, reference, ambiguity_margin=2)
    with pytest.raises(ValueError):
        compare_constraints(
            reference,
            reference,
            align_variables(reference, reference),
            tolerance=-1,
        )
    with pytest.raises(ValueError):
        compare_models(reference, reference, coefficient_tolerance=-1)
    with pytest.raises(ValueError):
        compare_models(reference, reference, solve_time_limit_seconds=0)

    assert _jaccard(set(), {"x"}) == 0
    assert _number_similarity(None, 1) == 0
    empty = reference.model_copy(update={"variables": []})
    alignment = align_variables(reference, empty)
    assert not alignment.reliable
    variables = _compare_variables(
        reference,
        empty,
        VariableAlignment(
            mean_score=0,
            minimum_score=0,
            reliable=False,
        ),
        tolerance=1e-7,
    )
    assert variables.matched == 0
