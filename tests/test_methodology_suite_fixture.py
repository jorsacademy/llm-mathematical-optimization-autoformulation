from __future__ import annotations

from pathlib import Path

from autoformulation.research_benchmark import (
    ExpectedOutcome,
    VariantKind,
    load_suite,
    suite_fingerprint,
)
from autoformulation.solver import solve_model

EXPECTED_SUITE_SHA256 = "88c2672bef44a17c80d00b890f0f2c71c6f0da31580c6408c426ae2023440d30"


def test_methodology_smoke_suite_is_valid_and_immutable() -> None:
    suite = load_suite(Path("benchmarks/methodology_suite.json"))
    assert suite.suite_id == "methodology-smoke-v1"
    assert suite_fingerprint(suite) == EXPECTED_SUITE_SHA256
    assert len(suite.cases) == 8
    assert set(suite.reference_models) == {"production_planning", "transportation"}

    formulation_cases = [
        case for case in suite.cases if case.expected_outcome is ExpectedOutcome.FORMULATION
    ]
    abstention_cases = [
        case for case in suite.cases if case.expected_outcome is ExpectedOutcome.ABSTENTION
    ]
    assert len(formulation_cases) == 6
    assert len(abstention_cases) == 2
    assert {case.variant for case in suite.cases} == set(VariantKind)

    for reference in suite.reference_models.values():
        result = solve_model(reference)
        assert result.success
        assert result.solution_check is not None
        assert result.solution_check.feasible
