from __future__ import annotations

from pathlib import Path

from autoformulation.benchmark import (
    BenchmarkCase,
    cases_to_jsonl,
    load_cases,
    run_benchmark,
)
from autoformulation.extractors.base import ModelExtractor
from autoformulation.pipeline import AutoformulationPipeline
from autoformulation.schema import ModelSpec


class StaticExtractor(ModelExtractor):
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    def extract(self, statement: str) -> ModelSpec:
        return self.model


def test_identical_model_matches_reference(production_model: ModelSpec) -> None:
    case = BenchmarkCase(
        id="production",
        statement="Production planning statement.",
        reference_model=production_model,
    )
    summary = run_benchmark(
        [case], AutoformulationPipeline(StaticExtractor(production_model), max_repair_rounds=0)
    )
    assert summary.total_cases == 1
    assert summary.solve_rate == 1.0
    assert summary.objective_match_rate == 1.0
    assert summary.results[0].exact_model_match


def test_cases_round_trip_jsonl(tmp_path: Path, production_model: ModelSpec) -> None:
    cases = [
        BenchmarkCase(
            id="production",
            statement="Production planning statement.",
            reference_model=production_model,
        )
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text(cases_to_jsonl(cases), encoding="utf-8")
    loaded = load_cases(path)
    assert loaded == cases
