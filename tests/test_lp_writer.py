from __future__ import annotations

from autoformulation.lp_writer import write_lp
from autoformulation.schema import ModelSpec


def test_lp_writer_emits_objective_constraints_and_bounds(production_model: ModelSpec) -> None:
    rendered = write_lp(production_model)
    assert rendered.startswith("Maximize\n obj: 40 tables + 30 chairs")
    assert "labor_capacity: 2 tables + chairs <= 100" in rendered
    assert "0 <= tables" in rendered
    assert rendered.endswith("End\n")


def test_lp_writer_emits_binary_section(fixed_charge_model: ModelSpec) -> None:
    rendered = write_lp(fixed_charge_model)
    assert "obj: 8 quantity - 12 open_facility" in rendered
    assert "Binaries\n open_facility" in rendered
    assert "0 <= quantity <= 5" in rendered


def test_lp_writer_moves_lhs_constant_to_rhs(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["constraints"][0]["lhs"]["constant"] = 5
    rendered = write_lp(ModelSpec.model_validate(payload))
    assert "labor_capacity: 2 tables + chairs <= 95" in rendered


def test_lp_writer_covers_free_upper_fixed_and_integer_bounds(production_model: ModelSpec) -> None:
    payload = production_model.model_dump(mode="json")
    payload["variables"] = [
        {
            "name": "free_x",
            "description": "Free variable.",
            "variable_type": "continuous",
            "lower_bound": None,
            "upper_bound": None,
        },
        {
            "name": "upper_x",
            "description": "Upper-bounded variable.",
            "variable_type": "continuous",
            "lower_bound": None,
            "upper_bound": 7,
        },
        {
            "name": "fixed_x",
            "description": "Fixed variable.",
            "variable_type": "integer",
            "lower_bound": 2,
            "upper_bound": 2,
        },
    ]
    payload["objective"] = {
        "sense": "minimize",
        "description": "Simple objective.",
        "expression": {
            "terms": [
                {"variable": "free_x", "coefficient": -1},
                {"variable": "upper_x", "coefficient": 1},
                {"variable": "fixed_x", "coefficient": 0},
            ],
            "constant": -2,
        },
    }
    payload["constraints"] = []
    rendered = write_lp(ModelSpec.model_validate(payload))
    assert rendered.startswith("Minimize")
    assert "obj: - free_x + upper_x - 2" in rendered
    assert "free_x free" in rendered
    assert "upper_x <= 7" in rendered
    assert "fixed_x = 2" in rendered
    assert "Generals\n fixed_x" in rendered
