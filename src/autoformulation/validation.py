"""Static checks for the autoformulation intermediate representation."""

from __future__ import annotations

import math
from collections import Counter
from enum import StrEnum

from pydantic import Field

from autoformulation.schema import (
    ConstraintSpec,
    LinearExpression,
    ModelSpec,
    Relation,
    StrictModel,
    VariableSpec,
    VariableType,
)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(StrictModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    severity: Severity
    message: str
    location: str


class ValidationReport(StrictModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is Severity.WARNING for issue in self.issues)

    def summary(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }


class ModelValidator:
    """Perform deterministic semantic checks before a model reaches a solver.

    Unresolved questions are errors by default: the pipeline fails closed instead
    of silently inventing missing business data.
    """

    def __init__(
        self,
        *,
        unresolved_as_error: bool = True,
        assumptions_as_error: bool = False,
        zero_tolerance: float = 1e-12,
    ) -> None:
        if zero_tolerance < 0:
            raise ValueError("zero_tolerance must be nonnegative")
        self.unresolved_as_error = unresolved_as_error
        self.assumptions_as_error = assumptions_as_error
        self.zero_tolerance = zero_tolerance

    def validate(self, model: ModelSpec) -> ValidationReport:
        issues: list[ValidationIssue] = []
        variable_names = model.variable_names()
        variable_set = set(variable_names)

        self._check_duplicate_names(variable_names, "variables", issues)
        self._check_duplicate_names(
            [constraint.name for constraint in model.constraints], "constraints", issues
        )

        for index, variable in enumerate(model.variables):
            self._check_variable(variable, f"variables[{index}]", issues)

        self._check_expression(
            model.objective.expression,
            variable_set,
            "objective.expression",
            issues,
        )

        for index, constraint in enumerate(model.constraints):
            location = f"constraints[{index}]"
            self._check_expression(constraint.lhs, variable_set, f"{location}.lhs", issues)
            self._check_constant_constraint(constraint, location, issues)

        used_variables = model.objective.expression.variables_used()
        for constraint in model.constraints:
            used_variables.update(constraint.lhs.variables_used())
        for name in variable_names:
            if name not in used_variables:
                issues.append(
                    ValidationIssue(
                        code="UNUSED_VARIABLE",
                        severity=Severity.WARNING,
                        message=f"Variable '{name}' is never used in the objective or constraints.",
                        location=f"variables.{name}",
                    )
                )

        if not model.objective.expression.coefficient_map() and math.isclose(
            model.objective.expression.constant,
            0.0,
            abs_tol=self.zero_tolerance,
        ):
            issues.append(
                ValidationIssue(
                    code="EMPTY_OBJECTIVE",
                    severity=Severity.WARNING,
                    message="The objective is identically zero.",
                    location="objective.expression",
                )
            )

        if not model.constraints:
            issues.append(
                ValidationIssue(
                    code="NO_CONSTRAINTS",
                    severity=Severity.WARNING,
                    message="The model has no constraints; check whether this was intended.",
                    location="constraints",
                )
            )

        unresolved_severity = (
            Severity.ERROR if self.unresolved_as_error else Severity.WARNING
        )
        for index, question in enumerate(model.unresolved_questions):
            issues.append(
                ValidationIssue(
                    code="UNRESOLVED_QUESTION",
                    severity=unresolved_severity,
                    message=question,
                    location=f"unresolved_questions[{index}]",
                )
            )

        assumption_severity = Severity.ERROR if self.assumptions_as_error else Severity.WARNING
        for index, assumption in enumerate(model.assumptions):
            issues.append(
                ValidationIssue(
                    code="MODELING_ASSUMPTION",
                    severity=assumption_severity,
                    message=assumption,
                    location=f"assumptions[{index}]",
                )
            )

        return ValidationReport(issues=issues)

    def _check_duplicate_names(
        self, names: list[str], category: str, issues: list[ValidationIssue]
    ) -> None:
        for name, count in Counter(names).items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        code="DUPLICATE_NAME",
                        severity=Severity.ERROR,
                        message=f"Name '{name}' appears {count} times in {category}.",
                        location=category,
                    )
                )

    def _check_variable(
        self, variable: VariableSpec, location: str, issues: list[ValidationIssue]
    ) -> None:
        lower = variable.lower_bound
        upper = variable.upper_bound

        if variable.variable_type is VariableType.BINARY:
            allowed = [
                value
                for value in (0.0, 1.0)
                if (lower is None or value >= lower) and (upper is None or value <= upper)
            ]
            if not allowed:
                issues.append(
                    ValidationIssue(
                        code="EMPTY_BINARY_DOMAIN",
                        severity=Severity.ERROR,
                        message=(
                            "Bounds exclude both 0 and 1 for binary variable "
                            f"'{variable.name}'."
                        ),
                        location=location,
                    )
                )
            elif len(allowed) == 1:
                issues.append(
                    ValidationIssue(
                        code="BINARY_FIXED_BY_BOUNDS",
                        severity=Severity.WARNING,
                        message=f"Binary variable '{variable.name}' is fixed to {int(allowed[0])}.",
                        location=location,
                    )
                )
            if (lower is not None and lower < 0) or (upper is not None and upper > 1):
                issues.append(
                    ValidationIssue(
                        code="BINARY_BOUNDS_NORMALIZED",
                        severity=Severity.INFO,
                        message=(
                            f"Binary variable '{variable.name}' will be intersected with [0, 1] "
                            "during compilation."
                        ),
                        location=location,
                    )
                )

        if (
            variable.variable_type is VariableType.INTEGER
            and lower is not None
            and upper is not None
            and math.ceil(lower) > math.floor(upper)
        ):
            issues.append(
                ValidationIssue(
                    code="EMPTY_INTEGER_DOMAIN",
                    severity=Severity.ERROR,
                    message=f"Bounds contain no integer value for '{variable.name}'.",
                    location=location,
                )
            )

    def _check_expression(
        self,
        expression: LinearExpression,
        variable_names: set[str],
        location: str,
        issues: list[ValidationIssue],
    ) -> None:
        term_names = [term.variable for term in expression.terms]
        for name, count in Counter(term_names).items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        code="DUPLICATE_TERM",
                        severity=Severity.WARNING,
                        message=(
                            f"Variable '{name}' appears {count} times; "
                            "coefficients will be combined."
                        ),
                        location=location,
                    )
                )

        for index, term in enumerate(expression.terms):
            if term.variable not in variable_names:
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_SYMBOL",
                        severity=Severity.ERROR,
                        message=f"Expression references undefined variable '{term.variable}'.",
                        location=f"{location}.terms[{index}]",
                    )
                )
            if math.isclose(term.coefficient, 0.0, abs_tol=self.zero_tolerance):
                issues.append(
                    ValidationIssue(
                        code="ZERO_COEFFICIENT",
                        severity=Severity.INFO,
                        message=f"Term for '{term.variable}' has a zero coefficient.",
                        location=f"{location}.terms[{index}]",
                    )
                )

    def _check_constant_constraint(
        self,
        constraint: ConstraintSpec,
        location: str,
        issues: list[ValidationIssue],
    ) -> None:
        coefficients = constraint.lhs.coefficient_map()
        if any(abs(value) > self.zero_tolerance for value in coefficients.values()):
            return

        lhs = constraint.lhs.constant
        rhs = constraint.rhs
        satisfied = {
            Relation.LE: lhs <= rhs + self.zero_tolerance,
            Relation.GE: lhs >= rhs - self.zero_tolerance,
            Relation.EQ: math.isclose(lhs, rhs, abs_tol=self.zero_tolerance),
        }[constraint.relation]
        issues.append(
            ValidationIssue(
                code="REDUNDANT_CONSTANT_CONSTRAINT" if satisfied else "CONSTANT_CONTRADICTION",
                severity=Severity.WARNING if satisfied else Severity.ERROR,
                message=(
                    "Constraint contains no nonzero variable terms and is always satisfied."
                    if satisfied
                    else "Constraint contains no nonzero variable terms and is impossible."
                ),
                location=location,
            )
        )
