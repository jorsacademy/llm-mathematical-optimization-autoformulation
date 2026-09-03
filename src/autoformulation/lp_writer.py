"""Deterministic export to the conventional LP file format."""

from __future__ import annotations

import math

from autoformulation.schema import LinearExpression, ModelSpec, ObjectiveSense, VariableType


def _format_number(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-15):
        value = 0.0
    return f"{value:.12g}"


def _format_expression(expression: LinearExpression, *, include_constant: bool) -> str:
    entries: list[tuple[str, float]] = list(expression.coefficient_map().items())
    if include_constant and not math.isclose(expression.constant, 0.0, abs_tol=1e-15):
        entries.append(("", expression.constant))

    parts: list[str] = []
    for symbol, coefficient in entries:
        if math.isclose(coefficient, 0.0, abs_tol=1e-15):
            continue
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        if symbol:
            body = (
                symbol
                if math.isclose(magnitude, 1.0)
                else f"{_format_number(magnitude)} {symbol}"
            )
        else:
            body = _format_number(magnitude)

        if not parts:
            parts.append(f"- {body}" if sign == "-" else body)
        else:
            parts.append(f" {sign} {body}")

    return "".join(parts) or "0"


def write_lp(model: ModelSpec) -> str:
    """Render a validated model to an LP-format string.

    Constants in constraint left-hand sides are moved to the right-hand side.
    """

    lines = ["Maximize" if model.objective.sense is ObjectiveSense.MAXIMIZE else "Minimize"]
    lines.append(f" obj: {_format_expression(model.objective.expression, include_constant=True)}")
    lines.append("Subject To")
    for constraint in model.constraints:
        rhs = constraint.rhs - constraint.lhs.constant
        lhs = _format_expression(constraint.lhs, include_constant=False)
        lines.append(f" {constraint.name}: {lhs} {constraint.relation.value} {_format_number(rhs)}")

    lines.append("Bounds")
    for variable in model.variables:
        lower = variable.lower_bound
        upper = variable.upper_bound
        if variable.variable_type is VariableType.BINARY:
            effective_lower = 0.0 if lower is None else max(lower, 0.0)
            effective_upper = 1.0 if upper is None else min(upper, 1.0)
            if math.isclose(effective_lower, 0.0) and math.isclose(effective_upper, 1.0):
                continue
            lower, upper = effective_lower, effective_upper

        if lower is None and upper is None:
            lines.append(f" {variable.name} free")
        elif lower is None:
            lines.append(f" {variable.name} <= {_format_number(upper)}")
        elif upper is None:
            lines.append(f" {_format_number(lower)} <= {variable.name}")
        elif math.isclose(lower, upper):
            lines.append(f" {variable.name} = {_format_number(lower)}")
        else:
            lines.append(
                f" {_format_number(lower)} <= {variable.name} <= {_format_number(upper)}"
            )

    binaries = [
        variable.name
        for variable in model.variables
        if variable.variable_type is VariableType.BINARY
    ]
    integers = [
        variable.name
        for variable in model.variables
        if variable.variable_type is VariableType.INTEGER
    ]
    if binaries:
        lines.append("Binaries")
        lines.extend(f" {name}" for name in binaries)
    if integers:
        lines.append("Generals")
        lines.extend(f" {name}" for name in integers)
    lines.append("End")
    return "\n".join(lines) + "\n"
