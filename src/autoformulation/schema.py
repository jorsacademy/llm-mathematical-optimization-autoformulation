"""Typed intermediate representation for finite linear and mixed-integer models.

The schema deliberately represents an expanded, solver-ready model rather than
arbitrary mathematical notation. This makes every symbol auditable and avoids
executing model-generated source code.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN, min_length=1, max_length=128)]


class StrictModel(BaseModel):
    """Base model shared by all public schemas."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class Relation(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


class LinearTerm(StrictModel):
    variable: Identifier
    coefficient: float


class LinearExpression(StrictModel):
    terms: list[LinearTerm] = Field(default_factory=list)
    constant: float = 0.0

    def coefficient_map(self) -> dict[str, float]:
        """Return coefficients with duplicate terms combined, preserving first-seen order."""

        coefficients: dict[str, float] = {}
        for term in self.terms:
            coefficients[term.variable] = coefficients.get(term.variable, 0.0) + term.coefficient
        return coefficients

    def variables_used(self) -> set[str]:
        return {term.variable for term in self.terms}


class VariableSpec(StrictModel):
    name: Identifier
    description: str = Field(min_length=1, max_length=500)
    variable_type: VariableType
    lower_bound: float | None
    upper_bound: float | None
    unit: str | None = Field(default=None, max_length=80)
    source_excerpt: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> VariableSpec:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class ObjectiveSpec(StrictModel):
    sense: ObjectiveSense
    expression: LinearExpression
    description: str = Field(min_length=1, max_length=500)
    unit: str | None = Field(default=None, max_length=80)
    source_excerpt: str | None = Field(default=None, max_length=500)


class ConstraintSpec(StrictModel):
    name: Identifier
    description: str = Field(min_length=1, max_length=500)
    lhs: LinearExpression
    relation: Relation
    rhs: float
    unit: str | None = Field(default=None, max_length=80)
    source_excerpt: str | None = Field(default=None, max_length=500)


class ModelSpec(StrictModel):
    """Versioned, flat LP/MILP intermediate representation."""

    schema_version: Literal["1.0"] = "1.0"
    name: Identifier
    problem_summary: str = Field(min_length=1, max_length=2000)
    source_language: str = Field(default="en", min_length=2, max_length=20)
    variables: list[VariableSpec] = Field(min_length=1)
    objective: ObjectiveSpec
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    def variable_names(self) -> list[str]:
        return [variable.name for variable in self.variables]
