"""Small, inspectable benchmark runner for solver-grounded evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, ValidationError

from autoformulation.pipeline import AutoformulationPipeline, model_fingerprint
from autoformulation.schema import ModelSpec, StrictModel
from autoformulation.solver import SolveStatus, solve_model


class BenchmarkCase(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1, max_length=128)
    statement: str = Field(min_length=1)
    reference_model: ModelSpec


class BenchmarkCaseResult(StrictModel):
    id: str
    completed: bool
    valid_model: bool
    solved: bool
    solver_status: SolveStatus | None = None
    objective_value: float | None = None
    reference_objective_value: float | None = None
    objective_gap_percent: float | None = Field(default=None, ge=0)
    objective_match: bool = False
    exact_model_match: bool = False
    variable_count_delta: int | None = None
    constraint_count_delta: int | None = None
    error_message: str | None = None


class BenchmarkSummary(StrictModel):
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    valid_models: int = Field(ge=0)
    solved_cases: int = Field(ge=0)
    objective_matches: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    validity_rate: float = Field(ge=0, le=1)
    solve_rate: float = Field(ge=0, le=1)
    objective_match_rate: float = Field(ge=0, le=1)
    results: list[BenchmarkCaseResult]


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(BenchmarkCase.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"Invalid benchmark case at {source}:{line_number}: {exc}") from exc
    return cases


def run_benchmark(
    cases: list[BenchmarkCase],
    pipeline: AutoformulationPipeline,
    *,
    objective_tolerance_percent: float = 1e-5,
) -> BenchmarkSummary:
    if objective_tolerance_percent < 0:
        raise ValueError("objective_tolerance_percent must be nonnegative")

    results = [
        _run_case(case, pipeline, objective_tolerance_percent=objective_tolerance_percent)
        for case in cases
    ]
    total = len(results)
    completed = sum(result.completed for result in results)
    valid = sum(result.valid_model for result in results)
    solved = sum(result.solved for result in results)
    matched = sum(result.objective_match for result in results)
    denominator = total or 1
    return BenchmarkSummary(
        total_cases=total,
        completed_cases=completed,
        valid_models=valid,
        solved_cases=solved,
        objective_matches=matched,
        completion_rate=completed / denominator,
        validity_rate=valid / denominator,
        solve_rate=solved / denominator,
        objective_match_rate=matched / denominator,
        results=results,
    )


def _run_case(
    case: BenchmarkCase,
    pipeline: AutoformulationPipeline,
    *,
    objective_tolerance_percent: float,
) -> BenchmarkCaseResult:
    reference_solution = solve_model(case.reference_model)
    if not reference_solution.success or reference_solution.objective_value is None:
        raise ValueError(f"Reference model for benchmark case '{case.id}' is not solvable.")

    try:
        candidate = pipeline.run(case.statement, solve=True)
    except Exception as exc:  # benchmark should record provider failures and continue
        return BenchmarkCaseResult(
            id=case.id,
            completed=False,
            valid_model=False,
            solved=False,
            reference_objective_value=reference_solution.objective_value,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    final_report = candidate.validation_history[-1]
    solution = candidate.solution
    solved = bool(solution and solution.success and solution.objective_value is not None)
    gap: float | None = None
    objective_match = False
    objective_value: float | None = None
    solver_status: SolveStatus | None = None
    if solution is not None:
        solver_status = solution.status
        objective_value = solution.objective_value
    if solved and objective_value is not None:
        denominator = max(abs(reference_solution.objective_value), 1e-12)
        gap = 100.0 * abs(objective_value - reference_solution.objective_value) / denominator
        objective_match = gap <= objective_tolerance_percent

    return BenchmarkCaseResult(
        id=case.id,
        completed=True,
        valid_model=final_report.ok,
        solved=solved,
        solver_status=solver_status,
        objective_value=objective_value,
        reference_objective_value=reference_solution.objective_value,
        objective_gap_percent=gap,
        objective_match=objective_match,
        exact_model_match=(
            model_fingerprint(candidate.final_model) == model_fingerprint(case.reference_model)
        ),
        variable_count_delta=(
            len(candidate.final_model.variables) - len(case.reference_model.variables)
        ),
        constraint_count_delta=(
            len(candidate.final_model.constraints) - len(case.reference_model.constraints)
        ),
    )


def save_summary(summary: BenchmarkSummary, path: str | Path) -> None:
    Path(path).write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def cases_to_jsonl(cases: list[BenchmarkCase]) -> str:
    return "\n".join(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False) for case in cases
    ) + ("\n" if cases else "")
