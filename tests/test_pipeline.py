from __future__ import annotations

from autoformulation.extractors.base import ModelExtractor
from autoformulation.pipeline import AutoformulationPipeline, PipelineStage, model_fingerprint
from autoformulation.schema import ModelSpec
from autoformulation.validation import ValidationReport


class RepairingExtractor(ModelExtractor):
    def __init__(self, initial: ModelSpec, repaired: ModelSpec) -> None:
        self.initial = initial
        self.repaired = repaired
        self.repairs = 0

    def extract(self, statement: str) -> ModelSpec:
        return self.initial

    def repair(
        self, statement: str, model: ModelSpec, report: ValidationReport
    ) -> ModelSpec:
        self.repairs += 1
        return self.repaired


class NonRepairingExtractor(ModelExtractor):
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    def extract(self, statement: str) -> ModelSpec:
        return self.model


def test_pipeline_repairs_then_solves(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unresolved_questions"] = ["Unknown capacity."]
    initial = ModelSpec.model_validate(payload)
    extractor = RepairingExtractor(initial, production_model)

    result = AutoformulationPipeline(extractor, max_repair_rounds=2).run("statement")

    assert result.valid
    assert result.repair_attempts == 1
    assert result.solution is not None and result.solution.success
    assert len(result.validation_history) == 2
    assert extractor.repairs == 1
    assert result.model_sha256 == model_fingerprint(production_model)
    assert result.extractor_metadata == {"extractor": "RepairingExtractor"}


def test_pipeline_stops_when_repair_is_unsupported(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unresolved_questions"] = ["Unknown capacity."]
    initial = ModelSpec.model_validate(payload)

    result = AutoformulationPipeline(
        NonRepairingExtractor(initial), max_repair_rounds=2
    ).run("statement")

    assert not result.valid
    assert result.solution is None
    assert any(event.stage is PipelineStage.REPAIR_STOPPED for event in result.events)


def test_pipeline_stops_on_unchanged_repair(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unresolved_questions"] = ["Unknown capacity."]
    initial = ModelSpec.model_validate(payload)
    extractor = RepairingExtractor(initial, initial)

    result = AutoformulationPipeline(extractor, max_repair_rounds=3).run("statement")

    assert result.repair_attempts == 1
    assert any("unchanged" in event.message for event in result.events)


def test_pipeline_can_skip_solving(production_model: ModelSpec) -> None:
    result = AutoformulationPipeline(NonRepairingExtractor(production_model)).run(
        "statement", solve=False
    )
    assert result.valid
    assert result.solution is None
    assert result.events[-1].stage is PipelineStage.NOT_SOLVED


def test_pipeline_enforces_statement_size_limit(production_model: ModelSpec) -> None:
    pipeline = AutoformulationPipeline(
        NonRepairingExtractor(production_model), max_statement_chars=5
    )
    try:
        pipeline.run("123456")
    except ValueError as exc:
        assert "safety limit" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    try:
        AutoformulationPipeline(NonRepairingExtractor(production_model), max_statement_chars=0)
    except ValueError as exc:
        assert "max_statement_chars" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_pipeline_rejects_invalid_configuration_and_blank_statement(
    production_model: ModelSpec,
) -> None:
    extractor = NonRepairingExtractor(production_model)
    try:
        AutoformulationPipeline(extractor, max_repair_rounds=6)
    except ValueError as exc:
        assert "between 0 and 5" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    pipeline = AutoformulationPipeline(extractor)
    try:
        pipeline.run(" ")
    except ValueError as exc:
        assert "nonempty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
