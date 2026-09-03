from __future__ import annotations

from pathlib import Path

from autoformulation.cli import main
from autoformulation.research_benchmark import (
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkSuiteCase,
    PredictionRecord,
    SystemDescriptor,
    VariantKind,
    load_report,
    save_artifact,
    suite_fingerprint,
)
from autoformulation.schema import ModelSpec


def _production_model() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "production",
            "problem_summary": "Select a production quantity.",
            "variables": [
                {
                    "name": "x",
                    "description": "Production quantity.",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": 10,
                }
            ],
            "objective": {
                "sense": "maximize",
                "description": "Maximize contribution.",
                "expression": {
                    "terms": [{"variable": "x", "coefficient": 2}],
                    "constant": 0,
                },
            },
            "constraints": [],
            "assumptions": [],
            "unresolved_questions": [],
        }
    )


def test_score_and_compare_commands(tmp_path: Path) -> None:
    production_model = _production_model()
    suite = BenchmarkSuite(
        suite_id="cli-suite",
        description="CLI methodology test.",
        reference_models={"production": production_model},
        cases=[
            BenchmarkSuiteCase(
                id="production.canonical",
                family_id="production",
                variant=VariantKind.CANONICAL,
                statement="Production planning statement.",
                reference_model_id="production",
            )
        ],
    )
    run = BenchmarkRun(
        suite_id=suite.suite_id,
        suite_sha256=suite_fingerprint(suite),
        system=SystemDescriptor(
            system_id="static-system",
            provider="test",
            model="reviewed-model",
            repair_rounds=0,
        ),
        predictions=[
            PredictionRecord(
                case_id="production.canonical",
                completed=True,
                model=production_model,
                valid=True,
                latency_seconds=0.01,
            )
        ],
    )

    suite_path = tmp_path / "suite.json"
    run_path = tmp_path / "run.json"
    report_path = tmp_path / "report.json"
    leaderboard_path = tmp_path / "leaderboard.md"
    save_artifact(suite, suite_path)
    save_artifact(run, run_path)

    assert (
        main(
            [
                "benchmark-score",
                str(suite_path),
                str(run_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    report = load_report(report_path)
    assert report.summary.expectation_pass_rate == 1.0
    assert report.run_sha256

    assert (
        main(
            [
                "benchmark-compare",
                str(report_path),
                "--output",
                str(leaderboard_path),
            ]
        )
        == 0
    )
    assert "static-system" in leaderboard_path.read_text(encoding="utf-8")
