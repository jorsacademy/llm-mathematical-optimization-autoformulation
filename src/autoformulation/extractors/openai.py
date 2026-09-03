"""Optional OpenAI Responses API adapter using schema-constrained output."""

from __future__ import annotations

from typing import Any

from autoformulation.extractors.base import ExtractionError, ModelExtractor
from autoformulation.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_formulation_prompt,
    build_repair_prompt,
)
from autoformulation.schema import ModelSpec
from autoformulation.validation import ValidationReport


class OpenAIExtractor(ModelExtractor):
    """Extract a ModelSpec without executing model-generated code.

    The OpenAI dependency is optional. Passing a compatible client is useful for tests and for
    advanced deployments that configure custom transports or workload identity.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model

        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise ExtractionError(
                "OpenAI support is not installed. Run: pip install -e '.[openai]'"
            ) from exc

        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def metadata(self) -> dict[str, str]:
        return {
            "extractor": type(self).__name__,
            "provider": "openai",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "schema_version": "1.0",
        }

    def extract(self, statement: str) -> ModelSpec:
        if not statement.strip():
            raise ValueError("statement must be nonempty")
        return self._request(build_formulation_prompt(statement))

    def repair(
        self,
        statement: str,
        model: ModelSpec,
        report: ValidationReport,
    ) -> ModelSpec:
        return self._request(build_repair_prompt(statement, model, report))

    def _request(self, prompt: str) -> ModelSpec:
        try:
            response = self._client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                text_format=ModelSpec,
            )
        except Exception as exc:  # provider exceptions vary across SDK versions
            raise ExtractionError(
                f"OpenAI request failed: {type(exc).__name__}: {exc}"
            ) from exc
        return self._extract_parsed_model(response)

    @staticmethod
    def _extract_parsed_model(response: Any) -> ModelSpec:
        direct = getattr(response, "output_parsed", None)
        if isinstance(direct, ModelSpec):
            return direct
        if isinstance(direct, dict):
            return ModelSpec.model_validate(direct)

        refusals: list[str] = []
        for output in getattr(response, "output", []):
            for content in getattr(output, "content", []):
                parsed = getattr(content, "parsed", None)
                if isinstance(parsed, ModelSpec):
                    return parsed
                if isinstance(parsed, dict):
                    return ModelSpec.model_validate(parsed)
                if getattr(content, "type", None) == "refusal":
                    refusals.append(str(getattr(content, "refusal", "provider refusal")))

        if refusals:
            raise ExtractionError("Provider refused the request: " + " | ".join(refusals))
        raise ExtractionError("Provider response did not contain a parsed ModelSpec.")
