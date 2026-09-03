"""Provider-neutral extractor contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autoformulation.schema import ModelSpec
from autoformulation.validation import ValidationReport


class ExtractionError(RuntimeError):
    """Raised when an LLM provider cannot return a valid ModelSpec."""


class ModelExtractor(ABC):
    @abstractmethod
    def extract(self, statement: str) -> ModelSpec:
        """Create a candidate model from a natural-language statement."""

    def metadata(self) -> dict[str, str]:
        """Return non-secret provenance recorded in pipeline artifacts."""

        return {"extractor": type(self).__name__}

    def repair(
        self,
        statement: str,
        model: ModelSpec,
        report: ValidationReport,
    ) -> ModelSpec:
        """Return a complete replacement after deterministic validation feedback."""

        raise NotImplementedError("This extractor does not support repair.")
