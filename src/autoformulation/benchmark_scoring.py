"""Generation, gold scoring, abstention checks, and robustness aggregation."""

from __future__ import annotations

import time
from collections import Counter, defaultdict

from autoformulation.benchmark_models import (
    BenchmarkCaseScore,
    BenchmarkIssue,
    BenchmarkReport,
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkSuiteCase,
    BenchmarkSummaryV2,
    ExpectedOutcome,
    FormulationPassPolicy,
    MetricSlice,
    PredictionRecord,
    ScoringConfig,
    SystemDescriptor,
    VariantKind,
    run_fingerprint,
    suite_fingerprint,
)
from autoformulation.model_comparison import compare_models
from autoformulation.pipeline import AutoformulationPipeline
from autoformulation.schema import ModelSpec
from autoformulation.solver import SolveOptions, solve_model
from autoformulation.validation import ModelValidator, Severity


def _system_from_pipeline(
    pipeline: AutoformulationPipeline,
    *,
    system_id: str,
) -> SystemDescriptor:
    metadata = pipeline.extractor.metadata()
    provider = metadata.get("provider", metadata.get("extractor", "unknown"))
    return SystemDescriptor(
        system_id=system_id,
        provider=provider,
        model=metadata.get("model"),
        prompt_version=metadata.get("prompt_version"),
        repair_rounds=pipeline.max_repair_rounds,
        metadata=metadata,
    )


def generate_benchmark_run(
    suite: BenchmarkSuite,
    pipeline: AutoformulationPipeline,
    *,
    system_id: str,
) -> BenchmarkRun:
    predictions: list[PredictionRecord] = []
    for case in suite.cases:
        started = time.perf_counter()
        try:
            result = pipeline.run(case.statement, solve=False)
        except Exception as exc:  # provider and parse failures are data, not run aborts
            predictions.append(
                PredictionRecord(
                    case_id=case.id,
                    completed=False,
                    latency_seconds=time.perf_counter() - started,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        report = result.validation_history[-1]
        predictions.append(
            PredictionRecord(
                case_id=case.id,
                completed=True,
                model=result.final_model,
                valid=report.ok,
                validation_codes=[issue.code for issue in report.issues],
                repair_attempts=result.repair_attempts,
                latency_seconds=time.perf_counter() - started,
            )
        )
    return BenchmarkRun(
        suite_id=suite.suite_id,
        suite_sha256=suite_fingerprint(suite),
        system=_system_from_pipeline(pipeline, system_id=system_id),
        predictions=predictions,
    )


def _score_formulation_case(
    case: BenchmarkSuiteCase,
    prediction: PredictionRecord,
    reference: ModelSpec,
    *,
    coefficient_tolerance: float,
    objective_tolerance_percent: float,
    minimum_alignment_score: float,
    solve_time_limit_seconds: float,
    formulation_pass_policy: FormulationPassPolicy,
) -> BenchmarkCaseScore:
    model = prediction.model
    if model is None:
        return BenchmarkCaseScore(
            case_id=case.id,
            family_id=case.family_id,
            variant=case.variant,
            expected_outcome=case.expected_outcome,
            tags=case.tags,
            prediction_completed=prediction.completed,
            candidate_valid=False,
            candidate_solved=False,
            abstained=False,
            expectation_pass=False,
            latency_seconds=prediction.latency_seconds,
            issues=[
                BenchmarkIssue(
                    code="missing_candidate",
                    message=prediction.error_message or "No candidate model was recorded.",
                )
            ],
        )

    abstained = bool(model.unresolved_questions)
    comparison = compare_models(
        reference,
        model,
        coefficient_tolerance=coefficient_tolerance,
        objective_tolerance_percent=objective_tolerance_percent,
        minimum_alignment_score=minimum_alignment_score,
        solve_time_limit_seconds=solve_time_limit_seconds,
    )
    issues = [
        BenchmarkIssue(code=issue.code.value, message=issue.message)
        for issue in comparison.issues
    ]
    if abstained:
        issues.append(
            BenchmarkIssue(
                code="unexpected_abstention",
                message="A complete formulation was expected, but unresolved questions remain.",
            )
        )
    formulation_pass = (
        comparison.strict_match
        if formulation_pass_policy is FormulationPassPolicy.STRICT
        else comparison.behavioral_match
    )
    return BenchmarkCaseScore(
        case_id=case.id,
        family_id=case.family_id,
        variant=case.variant,
        expected_outcome=case.expected_outcome,
        tags=case.tags,
        prediction_completed=prediction.completed,
        candidate_valid=comparison.candidate_valid,
        candidate_solved=comparison.candidate_solved,
        abstained=abstained,
        expectation_pass=formulation_pass and not abstained,
        behavioral_match=comparison.behavioral_match,
        structural_match=comparison.structural_match,
        strict_match=comparison.strict_match,
        latency_seconds=prediction.latency_seconds,
        comparison=comparison,
        issues=issues,
    )


def _score_abstention_case(
    case: BenchmarkSuiteCase,
    prediction: PredictionRecord,
) -> BenchmarkCaseScore:
    model = prediction.model
    abstained = bool(model and model.unresolved_questions)
    validation = ModelValidator().validate(model) if model is not None else None
    candidate_valid = bool(validation and validation.ok)
    blocking_codes = (
        {issue.code for issue in validation.issues if issue.severity is Severity.ERROR}
        if validation is not None
        else set()
    )
    clean_abstention = blocking_codes == {"UNRESOLVED_QUESTION"}
    passed = prediction.completed and abstained and clean_abstention
    issues: list[BenchmarkIssue] = []
    if not prediction.completed:
        issues.append(
            BenchmarkIssue(
                code="provider_failure",
                message=prediction.error_message or "Generation did not complete.",
            )
        )
    elif not abstained:
        issues.append(
            BenchmarkIssue(
                code="failure_to_abstain",
                message=(
                    "The case is intentionally underspecified, but no unresolved "
                    "question was reported."
                ),
            )
        )
    elif not clean_abstention:
        issues.append(
            BenchmarkIssue(
                code="invalid_abstention",
                message=(
                    "The model reported an unresolved question but also contained other "
                    "blocking validation errors."
                ),
            )
        )
    return BenchmarkCaseScore(
        case_id=case.id,
        family_id=case.family_id,
        variant=case.variant,
        expected_outcome=case.expected_outcome,
        tags=case.tags,
        prediction_completed=prediction.completed,
        candidate_valid=candidate_valid,
        candidate_solved=False,
        abstained=abstained,
        expectation_pass=passed,
        latency_seconds=prediction.latency_seconds,
        issues=issues,
    )


def _validate_reference_models(
    suite: BenchmarkSuite,
    *,
    solve_time_limit_seconds: float,
) -> None:
    referenced_ids: set[str] = set()
    for case in suite.cases:
        if case.reference_model_id is not None:
            referenced_ids.add(case.reference_model_id)
    for reference_id in sorted(referenced_ids):
        reference = suite.reference_models[reference_id]
        result = solve_model(reference, SolveOptions(time_limit=solve_time_limit_seconds))
        if not result.success:
            raise ValueError(
                f"reference model '{reference_id}' is not a valid, solvable gold model: "
                f"{result.status.value}"
            )


def score_benchmark_run(
    suite: BenchmarkSuite,
    run: BenchmarkRun,
    *,
    coefficient_tolerance: float = 1e-7,
    objective_tolerance_percent: float = 1e-5,
    minimum_alignment_score: float = 0.55,
    solve_time_limit_seconds: float = 60.0,
    formulation_pass_policy: FormulationPassPolicy = FormulationPassPolicy.STRICT,
) -> BenchmarkReport:
    if run.suite_id != suite.suite_id:
        raise ValueError(
            f"run targets suite '{run.suite_id}', not requested suite '{suite.suite_id}'"
        )
    expected_suite_sha256 = suite_fingerprint(suite)
    if run.suite_sha256 != expected_suite_sha256:
        raise ValueError("run suite fingerprint does not match the supplied benchmark suite")
    if coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance must be nonnegative")
    if objective_tolerance_percent < 0:
        raise ValueError("objective_tolerance_percent must be nonnegative")
    if not 0 <= minimum_alignment_score <= 1:
        raise ValueError("minimum_alignment_score must be between 0 and 1")
    if solve_time_limit_seconds <= 0:
        raise ValueError("solve_time_limit_seconds must be positive")

    _validate_reference_models(
        suite,
        solve_time_limit_seconds=solve_time_limit_seconds,
    )

    predictions = {prediction.case_id: prediction for prediction in run.predictions}
    unknown = sorted(set(predictions) - {case.id for case in suite.cases})
    if unknown:
        raise ValueError("run contains unknown case IDs: " + ", ".join(unknown))

    scores: list[BenchmarkCaseScore] = []
    for case in suite.cases:
        prediction = predictions.get(
            case.id,
            PredictionRecord(
                case_id=case.id,
                completed=False,
                latency_seconds=0.0,
                error_message="Prediction missing from run.",
            ),
        )
        if case.expected_outcome is ExpectedOutcome.ABSTENTION:
            scores.append(_score_abstention_case(case, prediction))
            continue
        reference = suite.reference_for(case)
        if reference is None:  # guarded by suite validation
            raise ValueError(f"case '{case.id}' has no reference model")
        scores.append(
            _score_formulation_case(
                case,
                prediction,
                reference,
                coefficient_tolerance=coefficient_tolerance,
                objective_tolerance_percent=objective_tolerance_percent,
                minimum_alignment_score=minimum_alignment_score,
                solve_time_limit_seconds=solve_time_limit_seconds,
                formulation_pass_policy=formulation_pass_policy,
            )
        )

    return BenchmarkReport(
        suite_id=suite.suite_id,
        suite_sha256=expected_suite_sha256,
        run_sha256=run_fingerprint(run),
        scoring=ScoringConfig(
            coefficient_tolerance=coefficient_tolerance,
            objective_tolerance_percent=objective_tolerance_percent,
            minimum_alignment_score=minimum_alignment_score,
            solve_time_limit_seconds=solve_time_limit_seconds,
            formulation_pass_policy=formulation_pass_policy,
        ),
        system=run.system,
        summary=_summarize(scores),
        cases=scores,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _variant_slices(scores: list[BenchmarkCaseScore]) -> dict[str, MetricSlice]:
    result: dict[str, MetricSlice] = {}
    for variant in VariantKind:
        subset = [score for score in scores if score.variant is variant]
        passed = sum(score.expectation_pass for score in subset)
        result[variant.value] = MetricSlice(
            total=len(subset),
            passed=passed,
            rate=_rate(passed, len(subset)),
        )
    return result


def _tag_slices(scores: list[BenchmarkCaseScore]) -> dict[str, MetricSlice]:
    tags = sorted({tag for score in scores for tag in score.tags})
    result: dict[str, MetricSlice] = {}
    for tag in tags:
        subset = [score for score in scores if tag in score.tags]
        passed = sum(score.expectation_pass for score in subset)
        result[tag] = MetricSlice(
            total=len(subset),
            passed=passed,
            rate=_rate(passed, len(subset)),
        )
    return result


def _retention_rate(
    scores: list[BenchmarkCaseScore],
    target: VariantKind,
) -> float | None:
    by_family: dict[str, list[BenchmarkCaseScore]] = defaultdict(list)
    for score in scores:
        by_family[score.family_id].append(score)
    eligible: list[BenchmarkCaseScore] = []
    for family_scores in by_family.values():
        canonical = [score for score in family_scores if score.variant is VariantKind.CANONICAL]
        if canonical and all(score.expectation_pass for score in canonical):
            eligible.extend(score for score in family_scores if score.variant is target)
    return _rate(sum(score.expectation_pass for score in eligible), len(eligible))


def _summarize(scores: list[BenchmarkCaseScore]) -> BenchmarkSummaryV2:
    total = len(scores)
    completed = sum(score.prediction_completed for score in scores)
    formulation = [
        score for score in scores if score.expected_outcome is ExpectedOutcome.FORMULATION
    ]
    abstention = [
        score for score in scores if score.expected_outcome is ExpectedOutcome.ABSTENTION
    ]
    by_family: dict[str, list[BenchmarkCaseScore]] = defaultdict(list)
    for score in scores:
        by_family[score.family_id].append(score)
    robust_families = sum(
        bool(family_scores) and all(score.expectation_pass for score in family_scores)
        for family_scores in by_family.values()
    )
    taxonomy = Counter(issue.code for score in scores for issue in score.issues)
    gaps: list[float] = []
    for score in formulation:
        if score.comparison is None:
            continue
        gap = score.comparison.reference_decision_gap_percent
        if gap is not None:
            gaps.append(gap)
    latencies: list[float] = []
    for score in scores:
        if score.latency_seconds is not None:
            latencies.append(score.latency_seconds)
    passes = sum(score.expectation_pass for score in scores)
    return BenchmarkSummaryV2(
        total_cases=total,
        completed_cases=completed,
        formulation_cases=len(formulation),
        abstention_cases=len(abstention),
        expectation_passes=passes,
        completion_rate=completed / total,
        expectation_pass_rate=passes / total,
        valid_formulation_rate=_rate(
            sum(score.candidate_valid for score in formulation),
            len(formulation),
        ),
        solve_rate=_rate(sum(score.candidate_solved for score in formulation), len(formulation)),
        behavioral_match_rate=_rate(
            sum(score.behavioral_match for score in formulation),
            len(formulation),
        ),
        structural_match_rate=_rate(
            sum(score.structural_match for score in formulation),
            len(formulation),
        ),
        strict_match_rate=_rate(sum(score.strict_match for score in formulation), len(formulation)),
        abstention_accuracy=_rate(
            sum(score.expectation_pass for score in abstention),
            len(abstention),
        ),
        family_robustness_rate=robust_families / len(by_family),
        paraphrase_retention_rate=_retention_rate(scores, VariantKind.PARAPHRASE),
        adversarial_retention_rate=_retention_rate(scores, VariantKind.ADVERSARIAL),
        mean_reference_decision_gap_percent=(sum(gaps) / len(gaps) if gaps else None),
        mean_latency_seconds=(sum(latencies) / len(latencies) if latencies else None),
        by_variant=_variant_slices(scores),
        by_tag=_tag_slices(scores),
        error_taxonomy=dict(sorted(taxonomy.items())),
    )


__all__ = ["generate_benchmark_run", "score_benchmark_run"]
