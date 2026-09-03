from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoformulation.schema import LinearExpression, ModelSpec


def test_expression_combines_duplicate_terms() -> None:
    expression = LinearExpression.model_validate(
        {
            "terms": [
                {"variable": "x", "coefficient": 2},
                {"variable": "x", "coefficient": -0.5},
                {"variable": "y", "coefficient": 3},
            ]
        }
    )
    assert expression.coefficient_map() == {"x": 1.5, "y": 3.0}
    assert expression.variables_used() == {"x", "y"}


def test_schema_forbids_unknown_fields(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelSpec.model_validate(payload)


def test_schema_rejects_invalid_identifier(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["variables"][0]["name"] = "tables[0]"
    with pytest.raises(ValidationError, match="String should match pattern"):
        ModelSpec.model_validate(payload)


def test_schema_rejects_reversed_bounds(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["variables"][0]["lower_bound"] = 5
    payload["variables"][0]["upper_bound"] = 2
    with pytest.raises(ValidationError, match="lower_bound must not exceed upper_bound"):
        ModelSpec.model_validate(payload)
