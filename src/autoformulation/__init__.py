"""Verification-first autoformulation and methodology benchmarking for LP/MILP models."""

from autoformulation.model_comparison import ModelComparison, compare_models
from autoformulation.schema import ModelSpec
from autoformulation.solver import SolveResult, solve_model
from autoformulation.validation import ModelValidator, ValidationReport

__all__ = [
    "ModelComparison",
    "ModelSpec",
    "ModelValidator",
    "SolveResult",
    "ValidationReport",
    "compare_models",
    "solve_model",
]

__version__ = "0.2.0"
