from __future__ import annotations

import json
from pathlib import Path

from autoformulation.cli import main
from autoformulation.schema import ModelSpec


def test_validate_and_solve_commands(
    tmp_path: Path, production_model: ModelSpec, capsys: object
) -> None:
    path = tmp_path / "model.json"
    path.write_text(production_model.model_dump_json(indent=2), encoding="utf-8")

    assert main(["validate", str(path)]) == 0
    assert main(["solve", str(path)]) == 0


def test_render_lp_to_file(tmp_path: Path, production_model: ModelSpec) -> None:
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "model.lp"
    model_path.write_text(production_model.model_dump_json(indent=2), encoding="utf-8")
    assert main(["render-lp", str(model_path), "-o", str(output_path)]) == 0
    assert output_path.read_text(encoding="utf-8").endswith("End\n")


def test_schema_command_outputs_json(capsys: object) -> None:
    assert main(["schema"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    schema = json.loads(captured.out)
    assert schema["title"] == "ModelSpec"


def test_bad_model_path_returns_error(tmp_path: Path) -> None:
    assert main(["validate", str(tmp_path / "missing.json")]) == 1
