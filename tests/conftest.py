from __future__ import annotations

import pytest

from autoformulation.schema import ModelSpec


@pytest.fixture
def production_model() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "production_planning",
            "problem_summary": "Choose table and chair production quantities to maximize profit.",
            "source_language": "en",
            "variables": [
                {
                    "name": "tables",
                    "description": "Number of tables produced.",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": None,
                    "unit": "units",
                },
                {
                    "name": "chairs",
                    "description": "Number of chairs produced.",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": None,
                    "unit": "units",
                },
            ],
            "objective": {
                "sense": "maximize",
                "description": "Maximize total contribution margin.",
                "unit": "currency",
                "expression": {
                    "terms": [
                        {"variable": "tables", "coefficient": 40},
                        {"variable": "chairs", "coefficient": 30},
                    ],
                    "constant": 0,
                },
            },
            "constraints": [
                {
                    "name": "labor_capacity",
                    "description": "Available labor cannot be exceeded.",
                    "lhs": {
                        "terms": [
                            {"variable": "tables", "coefficient": 2},
                            {"variable": "chairs", "coefficient": 1},
                        ],
                        "constant": 0,
                    },
                    "relation": "<=",
                    "rhs": 100,
                    "unit": "labor_hours",
                },
                {
                    "name": "wood_capacity",
                    "description": "Available wood cannot be exceeded.",
                    "lhs": {
                        "terms": [
                            {"variable": "tables", "coefficient": 1},
                            {"variable": "chairs", "coefficient": 2},
                        ],
                        "constant": 0,
                    },
                    "relation": "<=",
                    "rhs": 80,
                    "unit": "wood_units",
                },
            ],
            "assumptions": [],
            "unresolved_questions": [],
        }
    )


@pytest.fixture
def fixed_charge_model() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "fixed_charge_production",
            "problem_summary": "Choose whether to open and how much to produce.",
            "variables": [
                {
                    "name": "quantity",
                    "description": "Production quantity.",
                    "variable_type": "continuous",
                    "lower_bound": 0,
                    "upper_bound": 5,
                },
                {
                    "name": "open_facility",
                    "description": "One when the facility is opened.",
                    "variable_type": "binary",
                    "lower_bound": 0,
                    "upper_bound": 1,
                },
            ],
            "objective": {
                "sense": "maximize",
                "description": "Maximize unit contribution less fixed cost.",
                "expression": {
                    "terms": [
                        {"variable": "quantity", "coefficient": 8},
                        {"variable": "open_facility", "coefficient": -12},
                    ]
                },
            },
            "constraints": [
                {
                    "name": "activation",
                    "description": "Production requires the facility to be open.",
                    "lhs": {
                        "terms": [
                            {"variable": "quantity", "coefficient": 1},
                            {"variable": "open_facility", "coefficient": -5},
                        ]
                    },
                    "relation": "<=",
                    "rhs": 0,
                }
            ],
        }
    )
