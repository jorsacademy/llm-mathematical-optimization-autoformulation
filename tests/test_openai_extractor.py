from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoformulation.extractors.base import ExtractionError
from autoformulation.extractors.openai import OpenAIExtractor
from autoformulation.schema import ModelSpec


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


def test_openai_extractor_accepts_direct_parsed_model(production_model: ModelSpec) -> None:
    client = FakeClient(SimpleNamespace(output_parsed=production_model))
    extractor = OpenAIExtractor(model="test-model", client=client)
    model = extractor.extract("A production planning problem.")
    assert model == production_model
    call = client.responses.calls[0]
    assert call["text_format"] is ModelSpec
    assert call["model"] == "test-model"
    assert extractor.metadata()["prompt_version"] == "1.1"


def test_openai_extractor_accepts_nested_parsed_model(production_model: ModelSpec) -> None:
    content = SimpleNamespace(type="output_text", parsed=production_model)
    response = SimpleNamespace(output=[SimpleNamespace(content=[content])])
    extractor = OpenAIExtractor(model="test-model", client=FakeClient(response))
    assert extractor.extract("statement") == production_model


def test_openai_extractor_surfaces_refusal() -> None:
    content = SimpleNamespace(type="refusal", refusal="not allowed", parsed=None)
    response = SimpleNamespace(output=[SimpleNamespace(content=[content])])
    extractor = OpenAIExtractor(model="test-model", client=FakeClient(response))
    with pytest.raises(ExtractionError, match="not allowed"):
        extractor.extract("statement")


def test_openai_extractor_wraps_provider_error() -> None:
    class BrokenResponses:
        def parse(self, **kwargs: object) -> object:
            raise RuntimeError("network down")

    client = SimpleNamespace(responses=BrokenResponses())
    extractor = OpenAIExtractor(model="test-model", client=client)
    with pytest.raises(ExtractionError, match="network down"):
        extractor.extract("statement")


def test_openai_extractor_accepts_direct_dictionary(production_model: ModelSpec) -> None:
    response = SimpleNamespace(output_parsed=production_model.model_dump(mode="json"))
    extractor = OpenAIExtractor(model="test-model", client=FakeClient(response))
    assert extractor.extract("statement") == production_model


def test_openai_extractor_rejects_empty_statement(production_model: ModelSpec) -> None:
    extractor = OpenAIExtractor(
        model="test-model", client=FakeClient(SimpleNamespace(output_parsed=production_model))
    )
    with pytest.raises(ValueError, match="statement"):
        extractor.extract("  ")


def test_openai_extractor_rejects_invalid_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIExtractor(model=" ", client=FakeClient(SimpleNamespace()))
    with pytest.raises(ValueError, match="timeout"):
        OpenAIExtractor(model="test", timeout_seconds=0, client=FakeClient(SimpleNamespace()))


def test_openai_extractor_rejects_response_without_model() -> None:
    extractor = OpenAIExtractor(model="test", client=FakeClient(SimpleNamespace(output=[])))
    with pytest.raises(ExtractionError, match="did not contain"):
        extractor.extract("statement")
