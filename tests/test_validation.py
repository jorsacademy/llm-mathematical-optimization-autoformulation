from __future__ import annotations

from autoformulation.schema import ModelSpec
from autoformulation.validation import ModelValidator, Severity


def _codes(model: ModelSpec) -> set[str]:
    return {issue.code for issue in ModelValidator().validate(model).issues}


def test_valid_model_has_no_errors(production_model: ModelSpec) -> None:
    report = ModelValidator().validate(production_model)
    assert report.ok
    assert report.error_count == 0
    assert report.summary()["ok"] is True


def test_unknown_symbol_is_error(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["objective"]["expression"]["terms"].append(
        {"variable": "ghost", "coefficient": 1}
    )
    report = ModelValidator().validate(ModelSpec.model_validate(payload))
    assert not report.ok
    issue = next(issue for issue in report.issues if issue.code == "UNKNOWN_SYMBOL")
    assert issue.severity is Severity.ERROR


def test_unresolved_question_fails_closed(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unresolved_questions"] = ["Is production required to be integral?"]
    report = ModelValidator().validate(ModelSpec.model_validate(payload))
    assert not report.ok
    assert "UNRESOLVED_QUESTION" in _codes(ModelSpec.model_validate(payload))


def test_assumption_can_be_warning_or_error(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["assumptions"] = ["Demand is unlimited."]
    model = ModelSpec.model_validate(payload)
    assert ModelValidator().validate(model).ok
    assert not ModelValidator(assumptions_as_error=True).validate(model).ok


def test_constant_contradiction_is_error(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["constraints"].append(
        {
            "name": "impossible",
            "description": "An impossible constant constraint.",
            "lhs": {"terms": [], "constant": 1},
            "relation": "<=",
            "rhs": 0,
        }
    )
    assert "CONSTANT_CONTRADICTION" in _codes(ModelSpec.model_validate(payload))


def test_duplicate_names_and_terms_are_reported(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["variables"].append(payload["variables"][0].copy())
    payload["objective"]["expression"]["terms"].append(
        {"variable": "chairs", "coefficient": 0}
    )
    codes = _codes(ModelSpec.model_validate(payload))
    assert "DUPLICATE_NAME" in codes
    assert "DUPLICATE_TERM" in codes
    assert "ZERO_COEFFICIENT" in codes


def test_empty_integer_domain_is_error(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    variable = payload["variables"][0]
    variable["variable_type"] = "integer"
    variable["lower_bound"] = 0.2
    variable["upper_bound"] = 0.8
    assert "EMPTY_INTEGER_DOMAIN" in _codes(ModelSpec.model_validate(payload))


def test_binary_bounds_are_checked(fixed_charge_model: ModelSpec) -> None:
    payload = fixed_charge_model.model_dump(mode="json")
    payload["variables"][1]["lower_bound"] = 0.2
    payload["variables"][1]["upper_bound"] = 1.2
    codes = _codes(ModelSpec.model_validate(payload))
    assert "BINARY_FIXED_BY_BOUNDS" in codes
    assert "BINARY_BOUNDS_NORMALIZED" in codes


def test_no_constraints_unused_and_redundant_constant_are_reported(
    production_model: ModelSpec,
) -> None:
    payload = production_model.model_dump(mode="json")
    payload["objective"]["expression"] = {"terms": [], "constant": 0}
    payload["constraints"] = []
    codes = _codes(ModelSpec.model_validate(payload))
    assert {"NO_CONSTRAINTS", "UNUSED_VARIABLE", "EMPTY_OBJECTIVE"} <= codes

    payload["constraints"] = [
        {
            "name": "always_true",
            "description": "A redundant constant constraint.",
            "lhs": {"terms": [], "constant": 0},
            "relation": "<=",
            "rhs": 1,
        }
    ]
    assert "REDUNDANT_CONSTANT_CONSTRAINT" in _codes(ModelSpec.model_validate(payload))


def test_empty_binary_domain_is_error(fixed_charge_model: ModelSpec) -> None:
    payload = fixed_charge_model.model_dump(mode="json")
    payload["variables"][1]["lower_bound"] = 0.2
    payload["variables"][1]["upper_bound"] = 0.8
    assert "EMPTY_BINARY_DOMAIN" in _codes(ModelSpec.model_validate(payload))


def test_negative_zero_tolerance_is_rejected() -> None:
    try:
        ModelValidator(zero_tolerance=-1)
    except ValueError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
