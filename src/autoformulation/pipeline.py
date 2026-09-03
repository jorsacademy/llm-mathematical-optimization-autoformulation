"""Bounded extraction, validation, repair, and solve orchestration."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field

from autoformulation.extractors.base import ModelExtractor
from autoformulation.schema import ModelSpec, StrictModel
from autoformulation.solver import SolveOptions, SolveResult, solve_model
from autoformulation.validation import ModelValidator, ValidationReport

DEFAULT_MAX_STATEMENT_CHARS = 100_000
MAX_STATEMENT_CHARS_HARD_LIMIT = 1_000_000


class PipelineStage(StrEnum):
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    REPAIRED = "repaired"
    REPAIR_STOPPED = "repair_stopped"
    SOLVED = "solved"
    NOT_SOLVED = "not_solved"


class PipelineEvent(StrictModel):
    stage: PipelineStage
    message: str


class PipelineResult(StrictModel):
    statement_sha256: str
    model_sha256: str
    extractor_metadata: dict[str, str]
    final_model: ModelSpec
    validation_history: list[ValidationReport]
    repair_attempts: int = Field(ge=0)
    solution: SolveResult | None = None
    events: list[PipelineEvent] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.validation_history[-1].ok


class AutoformulationPipeline:
    def __init__(
        self,
        extractor: ModelExtractor,
        *,
        validator: ModelValidator | None = None,
        max_repair_rounds: int = 1,
        solve_options: SolveOptions | None = None,
        max_statement_chars: int = DEFAULT_MAX_STATEMENT_CHARS,
    ) -> None:
        if not 0 <= max_repair_rounds <= 5:
            raise ValueError("max_repair_rounds must be between 0 and 5")
        if not 1 <= max_statement_chars <= MAX_STATEMENT_CHARS_HARD_LIMIT:
            raise ValueError(
                f"max_statement_chars must be between 1 and {MAX_STATEMENT_CHARS_HARD_LIMIT}"
            )
        self.extractor = extractor
        self.validator = validator or ModelValidator()
        self.max_repair_rounds = max_repair_rounds
        self.solve_options = solve_options or SolveOptions(time_limit=60.0)
        self.max_statement_chars = max_statement_chars

    def run(self, statement: str, *, solve: bool = True) -> PipelineResult:
        if not statement.strip():
            raise ValueError("statement must be nonempty")
        if len(statement) > self.max_statement_chars:
            raise ValueError(
                f"statement exceeds the {self.max_statement_chars}-character safety limit"
            )

        events = [PipelineEvent(stage=PipelineStage.EXTRACTED, message="Candidate extracted.")]
        model = self.extractor.extract(statement)
        validation_history = [self.validator.validate(model)]
        events.append(
            PipelineEvent(
                stage=PipelineStage.VALIDATED,
                message=(
                    f"Validation found {validation_history[-1].error_count} error(s) and "
                    f"{validation_history[-1].warning_count} warning(s)."
                ),
            )
        )

        repair_attempts = 0
        for _ in range(self.max_repair_rounds):
            report = validation_history[-1]
            if report.ok:
                break
            previous_hash = model_fingerprint(model)
            try:
                candidate = self.extractor.repair(statement, model, report)
            except NotImplementedError:
                events.append(
                    PipelineEvent(
                        stage=PipelineStage.REPAIR_STOPPED,
                        message="Extractor does not support repair.",
                    )
                )
                break

            repair_attempts += 1
            candidate_hash = model_fingerprint(candidate)
            model = candidate
            validation_history.append(self.validator.validate(model))
            events.append(
                PipelineEvent(
                    stage=PipelineStage.REPAIRED,
                    message=(
                        f"Repair round {repair_attempts} produced "
                        f"{validation_history[-1].error_count} validation error(s)."
                    ),
                )
            )
            if candidate_hash == previous_hash:
                events.append(
                    PipelineEvent(
                        stage=PipelineStage.REPAIR_STOPPED,
                        message="Repair returned an unchanged model; stopping to avoid a loop.",
                    )
                )
                break

        solution: SolveResult | None = None
        if solve and validation_history[-1].ok:
            solution = solve_model(
                model,
                self.solve_options,
                validator=self.validator,
            )
            events.append(
                PipelineEvent(
                    stage=PipelineStage.SOLVED,
                    message=f"Solver status: {solution.status.value}.",
                )
            )
        else:
            reason = "solve disabled" if not solve else "validation errors remain"
            events.append(
                PipelineEvent(
                    stage=PipelineStage.NOT_SOLVED,
                    message=f"Solver was not called: {reason}.",
                )
            )

        return PipelineResult(
            statement_sha256=hashlib.sha256(statement.encode("utf-8")).hexdigest(),
            model_sha256=model_fingerprint(model),
            extractor_metadata=self.extractor.metadata(),
            final_model=model,
            validation_history=validation_history,
            repair_attempts=repair_attempts,
            solution=solution,
            events=events,
        )


def model_fingerprint(model: ModelSpec) -> str:
    canonical = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
