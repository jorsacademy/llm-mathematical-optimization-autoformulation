from __future__ import annotations

import pytest

from autoformulation.schema import ModelSpec
from autoformulation.solver import SolveStatus, check_solution, solve_model


def test_solves_continuous_production_model(production_model: ModelSpec) -> None:
    result = solve_model(production_model)
    assert result.status is SolveStatus.OPTIMAL
    assert result.success
    assert result.objective_value == pytest.approx(2200.0)
    assert result.variable_values == pytest.approx({"tables": 40.0, "chairs": 20.0})
    assert result.solution_check is not None and result.solution_check.feasible


def test_solves_mixed_integer_model(fixed_charge_model: ModelSpec) -> None:
    result = solve_model(fixed_charge_model)
    assert result.success
    assert result.objective_value == pytest.approx(28.0)
    assert result.variable_values["quantity"] == pytest.approx(5.0)
    assert result.variable_values["open_facility"] == pytest.approx(1.0)


def test_infeasible_model_is_reported(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["constraints"] = [
        {
            "name": "lower",
            "description": "At least two tables.",
            "lhs": {"terms": [{"variable": "tables", "coefficient": 1}]},
            "relation": ">=",
            "rhs": 2,
        },
        {
            "name": "upper",
            "description": "At most one table.",
            "lhs": {"terms": [{"variable": "tables", "coefficient": 1}]},
            "relation": "<=",
            "rhs": 1,
        },
    ]
    result = solve_model(ModelSpec.model_validate(payload))
    assert result.status is SolveStatus.INFEASIBLE
    assert not result.success


def test_invalid_model_never_calls_solver(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unresolved_questions"] = ["What is the wood capacity?"]
    result = solve_model(ModelSpec.model_validate(payload))
    assert result.status is SolveStatus.INVALID_MODEL
    assert result.solver_status_code is None
    assert not result.success


def test_postsolve_check_detects_multiple_violations(fixed_charge_model: ModelSpec) -> None:
    check = check_solution(
        fixed_charge_model,
        {"quantity": 6.0, "open_facility": 0.5},
    )
    assert not check.feasible
    codes = {violation.code for violation in check.violations}
    assert "BOUND_VIOLATION" in codes
    assert "INTEGRALITY_VIOLATION" in codes
    assert "CONSTRAINT_VIOLATION" in codes


def test_postsolve_check_detects_missing_value(production_model: ModelSpec) -> None:
    check = check_solution(production_model, {"tables": 0.0})
    assert not check.feasible
    assert check.violations[0].code == "MISSING_VALUE"


def test_postsolve_check_rejects_nonfinite_and_unexpected_values(
    production_model: ModelSpec,
) -> None:
    check = check_solution(
        production_model,
        {"tables": float("nan"), "chairs": 0.0, "ghost": 1.0},
    )
    codes = {violation.code for violation in check.violations}
    assert not check.feasible
    assert {"NONFINITE_VALUE", "UNEXPECTED_VALUE"} <= codes


def test_unbounded_model_is_reported(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["constraints"] = []
    result = solve_model(ModelSpec.model_validate(payload))
    assert result.status is SolveStatus.UNBOUNDED
    assert not result.success


def test_equality_and_greater_equal_constraints(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["objective"] = {
        "sense": "minimize",
        "description": "Minimize tables.",
        "expression": {"terms": [{"variable": "tables", "coefficient": 1}]},
    }
    payload["constraints"] = [
        {
            "name": "balance",
            "description": "Tables and chairs total ten.",
            "lhs": {
                "terms": [
                    {"variable": "tables", "coefficient": 1},
                    {"variable": "chairs", "coefficient": 1},
                ]
            },
            "relation": "==",
            "rhs": 10,
        },
        {
            "name": "minimum_tables",
            "description": "At least three tables.",
            "lhs": {"terms": [{"variable": "tables", "coefficient": 1}]},
            "relation": ">=",
            "rhs": 3,
        },
    ]
    result = solve_model(ModelSpec.model_validate(payload))
    assert result.success
    assert result.variable_values["tables"] == pytest.approx(3)


def test_solver_invocation_error_is_wrapped(
    production_model: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_milp(**kwargs: object) -> object:
        raise RuntimeError("broken solver")

    monkeypatch.setattr("autoformulation.solver.milp", broken_milp)
    result = solve_model(production_model)
    assert result.status is SolveStatus.ERROR
    assert "broken solver" in result.solver_message
