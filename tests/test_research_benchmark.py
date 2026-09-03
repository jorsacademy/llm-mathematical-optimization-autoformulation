from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformulation.extractors.base import ModelExtractor
from autoformulation.pipeline import AutoformulationPipeline
from autoformulation.research_benchmark import (
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkSuiteCase,
    ExpectedOutcome,
    FormulationPassPolicy,
    PredictionRecord,
    ScoringConfig,
    SystemDescriptor,
    VariantKind,
    generate_benchmark_run,
    load_report,
    load_run,
    load_suite,
    render_leaderboard,
    save_artifact,
    score_benchmark_run,
    suite_fingerprint,
)
from autoformulation.schema import ModelSpec


def model() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "simple",
            "problem_summary": "simple",
            "variables": [
                {
                    "name": "x",
                    "description": "decision",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": 1,
                }
            ],
            "objective": {
                "sense": "maximize",
                "description": "maximize x",
                "expression": {"terms": [{"variable": "x", "coefficient": 1}]},
            },
            "constraints": [],
            "assumptions": [],
            "unresolved_questions": [],
        }
    )


def suite() -> BenchmarkSuite:
    reference = model()
    return BenchmarkSuite(
        suite_id="suite",
        description="test suite",
        reference_models={"simple": reference},
        cases=[
            BenchmarkSuiteCase(
                id="canonical",
                family_id="family",
                variant=VariantKind.CANONICAL,
                statement="canonical",
                reference_model_id="simple",
                tags=["base"],
            ),
            BenchmarkSuiteCase(
                id="paraphrase",
                family_id="family",
                variant=VariantKind.PARAPHRASE,
                statement="paraphrase",
                reference_model_id="simple",
                tags=["base", "paraphrase"],
            ),
            BenchmarkSuiteCase(
                id="attack",
                family_id="family",
                variant=VariantKind.ADVERSARIAL,
                statement="attack",
                reference_model_id="simple",
                tags=["adversarial"],
            ),
            BenchmarkSuiteCase(
                id="ambiguous",
                family_id="underspecified",
                variant=VariantKind.AMBIGUOUS,
                statement="ambiguous",
                expected_outcome=ExpectedOutcome.ABSTENTION,
                tags=["ambiguity"],
            ),
        ],
    )


class MappingExtractor(ModelExtractor):
    def __init__(self, mapping: dict[str, ModelSpec | Exception]) -> None:
        self.mapping = mapping

    def extract(self, statement: str) -> ModelSpec:
        result = self.mapping[statement]
        if isinstance(result, Exception):
            raise result
        return result

    def metadata(self) -> dict[str, str]:
        return {"provider": "test", "model": "static", "prompt_version": "v1"}


def make_run(system_id: str = "system") -> BenchmarkRun:
    reference = model()
    abstention = reference.model_copy(update={"unresolved_questions": ["missing value"]})
    pipeline = AutoformulationPipeline(
        MappingExtractor(
            {
                "canonical": reference,
                "paraphrase": reference,
                "attack": reference,
                "ambiguous": abstention,
            }
        ),
        max_repair_rounds=0,
    )
    return generate_benchmark_run(suite(), pipeline, system_id=system_id)


def test_generate_score_round_trip_and_leaderboards(tmp_path: Path) -> None:
    benchmark_suite = suite()
    run = make_run()
    report = score_benchmark_run(benchmark_suite, run)
    assert report.summary.expectation_pass_rate == 1
    assert report.summary.family_robustness_rate == 1
    assert report.summary.paraphrase_retention_rate == 1
    assert report.summary.adversarial_retention_rate == 1
    assert report.summary.abstention_accuracy == 1
    assert report.summary.by_tag["base"].rate == 1
    assert report.summary.by_tag["ambiguity"].passed == 1

    suite_path = tmp_path / "nested" / "suite.json"
    run_path = tmp_path / "run.json"
    report_path = tmp_path / "report.json"
    save_artifact(benchmark_suite, suite_path)
    save_artifact(run, run_path)
    save_artifact(report, report_path)
    assert load_suite(suite_path) == benchmark_suite
    assert load_run(run_path) == run
    assert load_report(report_path) == report

    markdown = render_leaderboard([report])
    assert "| system |" in markdown
    assert "strict" in markdown
    assert "expectation_pass" in render_leaderboard([report], output_format="csv")
    assert '"system": "system"' in render_leaderboard([report], output_format="json")


def test_generation_records_provider_failure() -> None:
    reference = model()
    benchmark_suite = BenchmarkSuite(
        suite_id="failure-suite",
        description="failure",
        reference_models={"simple": reference},
        cases=[
            BenchmarkSuiteCase(
                id="case",
                family_id="family",
                variant=VariantKind.CANONICAL,
                statement="boom",
                reference_model_id="simple",
            )
        ],
    )
    run = generate_benchmark_run(
        benchmark_suite,
        AutoformulationPipeline(
            MappingExtractor({"boom": RuntimeError("provider unavailable")}),
            max_repair_rounds=0,
        ),
        system_id="failure",
    )
    assert not run.predictions[0].completed
    report = score_benchmark_run(benchmark_suite, run)
    assert report.summary.error_taxonomy["missing_candidate"] == 1


def test_failure_to_abstain_and_unexpected_abstention() -> None:
    benchmark_suite = suite()
    reference = model()
    unresolved = reference.model_copy(update={"unresolved_questions": ["missing"]})
    run = BenchmarkRun(
        suite_id=benchmark_suite.suite_id,
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="bad", provider="test", repair_rounds=0),
        predictions=[
            PredictionRecord(
                case_id="canonical",
                completed=True,
                model=unresolved,
                valid=False,
                latency_seconds=0,
            ),
            PredictionRecord(
                case_id="paraphrase",
                completed=True,
                model=reference,
                valid=True,
                latency_seconds=0,
            ),
            PredictionRecord(
                case_id="attack",
                completed=True,
                model=reference,
                valid=True,
                latency_seconds=0,
            ),
            PredictionRecord(
                case_id="ambiguous",
                completed=True,
                model=reference,
                valid=True,
                latency_seconds=0,
            ),
        ],
    )
    report = score_benchmark_run(benchmark_suite, run)
    assert "unexpected_abstention" in {issue.code for issue in report.cases[0].issues}
    assert "failure_to_abstain" in {issue.code for issue in report.cases[-1].issues}
    assert report.summary.expectation_pass_rate == 0.5


def test_abstention_provider_failure_uses_separate_taxonomy() -> None:
    benchmark_suite = BenchmarkSuite(
        suite_id="abstain",
        description="abstain",
        cases=[
            BenchmarkSuiteCase(
                id="a",
                family_id="a",
                variant=VariantKind.AMBIGUOUS,
                statement="a",
                expected_outcome=ExpectedOutcome.ABSTENTION,
            )
        ],
    )
    run = BenchmarkRun(
        suite_id="abstain",
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="x", provider="p", repair_rounds=0),
        predictions=[
            PredictionRecord(
                case_id="a",
                completed=False,
                latency_seconds=0,
                error_message="failed",
            )
        ],
    )
    report = score_benchmark_run(benchmark_suite, run)
    assert report.summary.error_taxonomy["provider_failure"] == 1


def test_schema_and_run_invariants() -> None:
    reference = model()
    with pytest.raises(ValidationError):
        BenchmarkSuiteCase(
            id="x",
            family_id="x",
            variant=VariantKind.CANONICAL,
            statement="x",
        )
    with pytest.raises(ValidationError):
        BenchmarkSuiteCase(
            id="x",
            family_id="x",
            variant=VariantKind.AMBIGUOUS,
            statement="x",
            expected_outcome=ExpectedOutcome.ABSTENTION,
            reference_model_id="simple",
        )
    case = BenchmarkSuiteCase(
        id="x",
        family_id="x",
        variant=VariantKind.CANONICAL,
        statement="x",
        reference_model_id="simple",
    )
    with pytest.raises(ValidationError):
        BenchmarkSuite(
            suite_id="x",
            description="x",
            reference_models={"simple": reference},
            cases=[case, case],
        )
    with pytest.raises(ValidationError):
        BenchmarkSuite(
            suite_id="x",
            description="x",
            reference_models={},
            cases=[case],
        )
    with pytest.raises(ValidationError):
        PredictionRecord(case_id="x", completed=True, latency_seconds=0)
    prediction = PredictionRecord(case_id="x", completed=False, latency_seconds=0)
    with pytest.raises(ValidationError):
        BenchmarkRun(
            suite_id="x",
            suite_sha256="0" * 64,
            system=SystemDescriptor(system_id="x", provider="p", repair_rounds=0),
            predictions=[prediction, prediction],
        )


def test_scoring_and_leaderboard_input_validation() -> None:
    benchmark_suite = suite()
    run = make_run()
    with pytest.raises(ValueError):
        score_benchmark_run(
            benchmark_suite,
            run.model_copy(update={"suite_id": "other"}),
        )
    with pytest.raises(ValueError):
        score_benchmark_run(benchmark_suite, run, coefficient_tolerance=-1)
    with pytest.raises(ValueError):
        score_benchmark_run(benchmark_suite, run, objective_tolerance_percent=-1)
    with pytest.raises(ValueError):
        score_benchmark_run(
            benchmark_suite,
            run.model_copy(update={"suite_sha256": "0" * 64}),
        )
    with pytest.raises(ValueError):
        score_benchmark_run(benchmark_suite, run, minimum_alignment_score=2)
    with pytest.raises(ValueError):
        score_benchmark_run(benchmark_suite, run, solve_time_limit_seconds=0)
    unknown = run.model_copy(
        update={
            "predictions": [
                *run.predictions,
                PredictionRecord(case_id="unknown", completed=False, latency_seconds=0),
            ]
        }
    )
    with pytest.raises(ValueError):
        score_benchmark_run(benchmark_suite, unknown)
    with pytest.raises(ValueError):
        render_leaderboard([])
    report = score_benchmark_run(benchmark_suite, run)
    other = report.model_copy(update={"suite_id": "other"})
    with pytest.raises(ValueError):
        render_leaderboard([report, other])
    with pytest.raises(ValueError):
        render_leaderboard(
            [
                report,
                report.model_copy(
                    update={"scoring": ScoringConfig(objective_tolerance_percent=1.0)}
                ),
            ]
        )
    with pytest.raises(ValueError):
        render_leaderboard([report], output_format="xml")


def test_report_sorting_and_missing_prediction() -> None:
    benchmark_suite = suite()
    complete = score_benchmark_run(benchmark_suite, make_run("z-system"))
    incomplete_run = BenchmarkRun(
        suite_id=benchmark_suite.suite_id,
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="a-system", provider="p", repair_rounds=0),
        predictions=[],
    )
    incomplete = score_benchmark_run(benchmark_suite, incomplete_run)
    table = render_leaderboard([incomplete, complete])
    assert table.index("z-system") < table.index("a-system")
    assert incomplete.summary.completion_rate == 0
    assert incomplete.summary.paraphrase_retention_rate is None


def test_abstention_with_other_static_error_is_not_accepted() -> None:
    benchmark_suite = BenchmarkSuite(
        suite_id="abstention-integrity",
        description="abstention integrity",
        cases=[
            BenchmarkSuiteCase(
                id="ambiguous",
                family_id="ambiguous",
                variant=VariantKind.AMBIGUOUS,
                statement="missing data",
                expected_outcome=ExpectedOutcome.ABSTENTION,
            )
        ],
    )
    payload = model().model_dump(mode="json")
    payload["unresolved_questions"] = ["missing value"]
    payload["constraints"] = [
        {
            "name": "bad",
            "description": "unknown symbol",
            "lhs": {
                "terms": [{"variable": "ghost", "coefficient": 1}],
                "constant": 0,
            },
            "relation": "<=",
            "rhs": 1,
        }
    ]
    bad = ModelSpec.model_validate(payload)
    run = BenchmarkRun(
        suite_id=benchmark_suite.suite_id,
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="bad", provider="test", repair_rounds=0),
        predictions=[
            PredictionRecord(
                case_id="ambiguous",
                completed=True,
                model=bad,
                latency_seconds=0,
            )
        ],
    )
    report = score_benchmark_run(benchmark_suite, run)
    assert report.summary.abstention_accuracy == 0
    assert report.summary.error_taxonomy["invalid_abstention"] == 1


def test_unsolved_reference_model_is_rejected() -> None:
    payload = model().model_dump(mode="json")
    payload["constraints"] = [
        {
            "name": "impossible",
            "description": "x must exceed its upper bound",
            "lhs": {
                "terms": [{"variable": "x", "coefficient": 1}],
                "constant": 0,
            },
            "relation": ">=",
            "rhs": 2,
        }
    ]
    reference = ModelSpec.model_validate(payload)
    benchmark_suite = BenchmarkSuite(
        suite_id="bad-gold",
        description="bad gold",
        reference_models={"bad": reference},
        cases=[
            BenchmarkSuiteCase(
                id="case",
                family_id="family",
                variant=VariantKind.CANONICAL,
                statement="case",
                reference_model_id="bad",
            )
        ],
    )
    run = BenchmarkRun(
        suite_id=benchmark_suite.suite_id,
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="x", provider="test", repair_rounds=0),
        predictions=[
            PredictionRecord(
                case_id="case",
                completed=True,
                model=reference,
                latency_seconds=0,
            )
        ],
    )
    with pytest.raises(ValueError, match="gold model"):
        score_benchmark_run(benchmark_suite, run)


def test_formulation_pass_policy_is_explicit() -> None:
    reference = model()
    payload = reference.model_dump(mode="json")
    for term in payload["objective"]["expression"]["terms"]:
        term["coefficient"] *= 2
    scaled = ModelSpec.model_validate(payload)
    benchmark_suite = BenchmarkSuite(
        suite_id="policy",
        description="policy",
        reference_models={"simple": reference},
        cases=[
            BenchmarkSuiteCase(
                id="case",
                family_id="family",
                variant=VariantKind.CANONICAL,
                statement="case",
                reference_model_id="simple",
            )
        ],
    )
    run = BenchmarkRun(
        suite_id=benchmark_suite.suite_id,
        suite_sha256=suite_fingerprint(benchmark_suite),
        system=SystemDescriptor(system_id="scaled", provider="test", repair_rounds=0),
        predictions=[
            PredictionRecord(
                case_id="case",
                completed=True,
                model=scaled,
                latency_seconds=0,
            )
        ],
    )
    strict = score_benchmark_run(benchmark_suite, run)
    behavioral = score_benchmark_run(
        benchmark_suite,
        run,
        formulation_pass_policy=FormulationPassPolicy.BEHAVIORAL,
    )
    assert strict.cases[0].behavioral_match
    assert not strict.cases[0].structural_match
    assert not strict.cases[0].expectation_pass
    assert behavioral.cases[0].expectation_pass
