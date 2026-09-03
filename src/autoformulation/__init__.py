"""Verification-first autoformulation for finite LP and MILP models."""

from autoformulation.schema import ModelSpec
from autoformulation.solver import SolveResult, solve_model
from autoformulation.validation import ModelValidator, ValidationReport

__all__ = [
    "ModelSpec",
    "ModelValidator",
    "SolveResult",
    "ValidationReport",
    "solve_model",
]

__version__ = "0.1.0"
