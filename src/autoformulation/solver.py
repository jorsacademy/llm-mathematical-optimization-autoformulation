"""Compilation to SciPy/HiGHS and independent post-solve verification."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from pydantic import Field
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from autoformulation.schema import (
    LinearExpression,
    ModelSpec,
    ObjectiveSense,
    Relation,
    StrictModel,
    VariableSpec,
    VariableType,
)
from autoformulation.validation import ModelValidator, ValidationReport


class SolveStatus(StrEnum):
    OPTIMAL = "optimal"
    LIMIT = "limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INVALID_MODEL = "invalid_model"
    ERROR = "error"


class SolveOptions(StrictModel):
    time_limit: float | None = Field(default=None, gt=0)
    mip_relative_gap: float | None = Field(default=None, ge=0)
    feasibility_tolerance: float = Field(default=1e-7, gt=0)
    integrality_tolerance: float = Field(default=1e-6, gt=0)


class SolutionViolation(StrictModel):
    code: str
    magnitude: float | None = Field(default=None, ge=0)
    location: str
    message: str


class SolutionCheck(StrictModel):
    feasible: bool
    max_violation: float = Field(ge=0)
    violations: list[SolutionViolation] = Field(default_factory=list)


class SolveResult(StrictModel):
    status: SolveStatus
    success: bool
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    solver_status_code: int | None = None
    solver_message: str
    validation: ValidationReport
    solution_check: SolutionCheck | None = None


def evaluate_expression(expression: LinearExpression, values: dict[str, float]) -> float:
    return expression.constant + sum(
        term.coefficient * values[term.variable] for term in expression.terms
    )


def _effective_bounds(variable: VariableSpec) -> tuple[float, float]:
    lower = -math.inf if variable.lower_bound is None else variable.lower_bound
    upper = math.inf if variable.upper_bound is None else variable.upper_bound
    if variable.variable_type is VariableType.BINARY:
        lower = max(lower, 0.0)
        upper = min(upper, 1.0)
    return lower, upper


def check_solution(
    model: ModelSpec,
    values: dict[str, float],
    *,
    feasibility_tolerance: float = 1e-7,
    integrality_tolerance: float = 1e-6,
) -> SolutionCheck:
    violations: list[SolutionViolation] = []
    expected_names = set(model.variable_names())

    for name in sorted(set(values) - expected_names):
        violations.append(
            SolutionViolation(
                code="UNEXPECTED_VALUE",
                magnitude=None,
                location=f"variables.{name}",
                message=f"A value was returned for unknown variable '{name}'.",
            )
        )

    for variable in model.variables:
        if variable.name not in values:
            violations.append(
                SolutionViolation(
                    code="MISSING_VALUE",
                    magnitude=None,
                    location=f"variables.{variable.name}",
                    message=f"No value was returned for '{variable.name}'.",
                )
            )
            continue

        value = values[variable.name]
        if not math.isfinite(value):
            violations.append(
                SolutionViolation(
                    code="NONFINITE_VALUE",
                    magnitude=None,
                    location=f"variables.{variable.name}",
                    message=f"Value for '{variable.name}' is not finite.",
                )
            )
            continue

        lower, upper = _effective_bounds(variable)
        lower_violation = max(lower - value, 0.0)
        upper_violation = max(value - upper, 0.0)
        bound_violation = max(lower_violation, upper_violation)
        if bound_violation > feasibility_tolerance:
            violations.append(
                SolutionViolation(
                    code="BOUND_VIOLATION",
                    magnitude=bound_violation,
                    location=f"variables.{variable.name}",
                    message=f"Value {value:.12g} violates bounds [{lower}, {upper}].",
                )
            )

        if variable.variable_type in {VariableType.INTEGER, VariableType.BINARY}:
            integrality_violation = abs(value - round(value))
            if integrality_violation > integrality_tolerance:
                violations.append(
                    SolutionViolation(
                        code="INTEGRALITY_VIOLATION",
                        magnitude=integrality_violation,
                        location=f"variables.{variable.name}",
                        message=f"Value {value:.12g} is not integral.",
                    )
                )

    blocking_value_codes = {"MISSING_VALUE", "NONFINITE_VALUE"}
    if not any(violation.code in blocking_value_codes for violation in violations):
        for constraint in model.constraints:
            lhs = evaluate_expression(constraint.lhs, values)
            if constraint.relation is Relation.LE:
                magnitude = max(lhs - constraint.rhs, 0.0)
            elif constraint.relation is Relation.GE:
                magnitude = max(constraint.rhs - lhs, 0.0)
            else:
                magnitude = abs(lhs - constraint.rhs)
            if magnitude > feasibility_tolerance:
                violations.append(
                    SolutionViolation(
                        code="CONSTRAINT_VIOLATION",
                        magnitude=magnitude,
                        location=f"constraints.{constraint.name}",
                        message=(
                            f"Constraint evaluates to {lhs:.12g} {constraint.relation.value} "
                            f"{constraint.rhs:.12g}."
                        ),
                    )
                )

    max_violation = max(
        (violation.magnitude for violation in violations if violation.magnitude is not None),
        default=0.0,
    )
    return SolutionCheck(
        feasible=not violations,
        max_violation=max_violation,
        violations=violations,
    )


def solve_model(
    model: ModelSpec,
    options: SolveOptions | None = None,
    *,
    validator: ModelValidator | None = None,
) -> SolveResult:
    """Validate, compile, solve, and independently re-check a finite LP/MILP model."""

    options = options or SolveOptions()
    validator = validator or ModelValidator()
    validation = validator.validate(model)
    if not validation.ok:
        return SolveResult(
            status=SolveStatus.INVALID_MODEL,
            success=False,
            solver_message="Static validation failed; the solver was not called.",
            validation=validation,
        )

    names = model.variable_names()
    index = {name: position for position, name in enumerate(names)}
    objective = np.zeros(len(names), dtype=float)
    for name, coefficient in model.objective.expression.coefficient_map().items():
        objective[index[name]] = coefficient
    if model.objective.sense is ObjectiveSense.MAXIMIZE:
        objective = -objective

    lower_bounds = np.empty(len(names), dtype=float)
    upper_bounds = np.empty(len(names), dtype=float)
    integrality = np.zeros(len(names), dtype=int)
    for position, variable in enumerate(model.variables):
        lower_bounds[position], upper_bounds[position] = _effective_bounds(variable)
        if variable.variable_type in {VariableType.INTEGER, VariableType.BINARY}:
            integrality[position] = 1

    linear_constraints: LinearConstraint | None = None
    if model.constraints:
        row_indices: list[int] = []
        column_indices: list[int] = []
        coefficients: list[float] = []
        constraint_lower = np.full(len(model.constraints), -np.inf, dtype=float)
        constraint_upper = np.full(len(model.constraints), np.inf, dtype=float)
        for row, constraint in enumerate(model.constraints):
            for name, coefficient in constraint.lhs.coefficient_map().items():
                if coefficient != 0.0:
                    row_indices.append(row)
                    column_indices.append(index[name])
                    coefficients.append(coefficient)
            adjusted_rhs = constraint.rhs - constraint.lhs.constant
            if constraint.relation is Relation.LE:
                constraint_upper[row] = adjusted_rhs
            elif constraint.relation is Relation.GE:
                constraint_lower[row] = adjusted_rhs
            else:
                constraint_lower[row] = adjusted_rhs
                constraint_upper[row] = adjusted_rhs
        matrix = coo_matrix(
            (coefficients, (row_indices, column_indices)),
            shape=(len(model.constraints), len(names)),
            dtype=float,
        ).tocsr()
        linear_constraints = LinearConstraint(matrix, constraint_lower, constraint_upper)

    scipy_options: dict[str, float] = {}
    if options.time_limit is not None:
        scipy_options["time_limit"] = options.time_limit
    if options.mip_relative_gap is not None:
        scipy_options["mip_rel_gap"] = options.mip_relative_gap

    try:
        raw = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=linear_constraints,
            options=scipy_options or None,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return SolveResult(
            status=SolveStatus.ERROR,
            success=False,
            solver_message=f"Solver invocation failed: {type(exc).__name__}: {exc}",
            validation=validation,
        )

    status = {
        0: SolveStatus.OPTIMAL,
        1: SolveStatus.LIMIT,
        2: SolveStatus.INFEASIBLE,
        3: SolveStatus.UNBOUNDED,
        4: SolveStatus.ERROR,
    }.get(int(raw.status), SolveStatus.ERROR)

    values: dict[str, float] = {}
    objective_value: float | None = None
    solution_check: SolutionCheck | None = None
    if raw.x is not None and np.all(np.isfinite(raw.x)):
        values = {name: float(raw.x[position]) for position, name in enumerate(names)}
        objective_value = evaluate_expression(model.objective.expression, values)
        solution_check = check_solution(
            model,
            values,
            feasibility_tolerance=options.feasibility_tolerance,
            integrality_tolerance=options.integrality_tolerance,
        )

    success = status is SolveStatus.OPTIMAL and bool(solution_check and solution_check.feasible)
    message = str(raw.message)
    if status is SolveStatus.OPTIMAL and solution_check is not None and not solution_check.feasible:
        message += " Post-solve verification failed."

    return SolveResult(
        status=status,
        success=success,
        objective_value=objective_value,
        variable_values=values,
        solver_status_code=int(raw.status),
        solver_message=message,
        validation=validation,
        solution_check=solution_check,
    )
