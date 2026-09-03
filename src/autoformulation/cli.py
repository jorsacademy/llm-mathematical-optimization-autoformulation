"""Command-line interface for validation, solving, formulation, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from autoformulation import __version__
from autoformulation.benchmark import load_cases, run_benchmark, save_summary
from autoformulation.extractors.base import ExtractionError
from autoformulation.extractors.openai import OpenAIExtractor
from autoformulation.lp_writer import write_lp
from autoformulation.pipeline import AutoformulationPipeline
from autoformulation.schema import ModelSpec
from autoformulation.solver import SolveOptions, solve_model
from autoformulation.validation import ModelValidator


def _load_model(path: str | Path) -> ModelSpec:
    source = Path(path)
    try:
        return ModelSpec.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"Could not load ModelSpec from '{source}': {exc}") from exc


def _write_json(path: Path, value: object) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_json(value: object) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="or-autoformulate",
        description="Verification-first LP/MILP autoformulation laboratory.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Print the ModelSpec JSON Schema.")
    schema_parser.set_defaults(handler=_handle_schema)

    validate_parser = subparsers.add_parser("validate", help="Statically validate a model JSON.")
    validate_parser.add_argument("model_json")
    validate_parser.add_argument(
        "--strict-assumptions",
        action="store_true",
        help="Treat every declared modeling assumption as an error.",
    )
    validate_parser.set_defaults(handler=_handle_validate)

    solve_parser = subparsers.add_parser("solve", help="Validate and solve a model JSON.")
    solve_parser.add_argument("model_json")
    solve_parser.add_argument("--time-limit", type=float)
    solve_parser.add_argument("--mip-gap", type=float)
    solve_parser.set_defaults(handler=_handle_solve)

    lp_parser = subparsers.add_parser("render-lp", help="Export a model JSON to LP format.")
    lp_parser.add_argument("model_json")
    lp_parser.add_argument("--output", "-o")
    lp_parser.set_defaults(handler=_handle_render_lp)

    formulate_parser = subparsers.add_parser(
        "formulate",
        help="Use an OpenAI model to extract, validate, optionally repair, and solve.",
    )
    formulate_parser.add_argument("statement_file")
    formulate_parser.add_argument(
        "--model",
        default=os.environ.get("AUTOFORMULATION_MODEL"),
        help="OpenAI model ID; may also be set with AUTOFORMULATION_MODEL.",
    )
    formulate_parser.add_argument("--output-dir", "-o", default="autoformulation-run")
    formulate_parser.add_argument("--repair-rounds", type=int, default=1)
    formulate_parser.add_argument("--timeout", type=float, default=120.0)
    formulate_parser.add_argument("--time-limit", type=float)
    formulate_parser.add_argument("--mip-gap", type=float)
    formulate_parser.add_argument("--no-solve", action="store_true")
    formulate_parser.add_argument("--strict-assumptions", action="store_true")
    formulate_parser.set_defaults(handler=_handle_formulate)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run JSONL benchmark cases through the OpenAI extraction pipeline.",
    )
    benchmark_parser.add_argument("cases_jsonl")
    benchmark_parser.add_argument(
        "--model",
        default=os.environ.get("AUTOFORMULATION_MODEL"),
        help="OpenAI model ID; may also be set with AUTOFORMULATION_MODEL.",
    )
    benchmark_parser.add_argument("--output", "-o", default="benchmark-results.json")
    benchmark_parser.add_argument("--repair-rounds", type=int, default=1)
    benchmark_parser.add_argument("--timeout", type=float, default=120.0)
    benchmark_parser.add_argument("--objective-tolerance-percent", type=float, default=1e-5)
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    return parser


def _handle_schema(_: argparse.Namespace) -> int:
    _print_json(ModelSpec.model_json_schema())
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    model = _load_model(args.model_json)
    report = ModelValidator(assumptions_as_error=args.strict_assumptions).validate(model)
    _print_json(report.summary())
    return 0 if report.ok else 2


def _handle_solve(args: argparse.Namespace) -> int:
    model = _load_model(args.model_json)
    result = solve_model(
        model,
        SolveOptions(time_limit=args.time_limit, mip_relative_gap=args.mip_gap),
    )
    _print_json(result)
    return 0 if result.success else 3


def _handle_render_lp(args: argparse.Namespace) -> int:
    model = _load_model(args.model_json)
    report = ModelValidator().validate(model)
    if not report.ok:
        _print_json(report.summary())
        return 2
    rendered = write_lp(model)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _require_model_id(value: str | None) -> str:
    if not value:
        raise ValueError("An OpenAI model ID is required via --model or AUTOFORMULATION_MODEL.")
    return value


def _make_pipeline(args: argparse.Namespace) -> AutoformulationPipeline:
    extractor = OpenAIExtractor(
        model=_require_model_id(args.model),
        timeout_seconds=args.timeout,
    )
    validator = ModelValidator(assumptions_as_error=getattr(args, "strict_assumptions", False))
    solve_options = SolveOptions(
        time_limit=getattr(args, "time_limit", None),
        mip_relative_gap=getattr(args, "mip_gap", None),
    )
    return AutoformulationPipeline(
        extractor,
        validator=validator,
        max_repair_rounds=args.repair_rounds,
        solve_options=solve_options,
    )


def _handle_formulate(args: argparse.Namespace) -> int:
    statement_path = Path(args.statement_file)
    statement = statement_path.read_text(encoding="utf-8")
    pipeline = _make_pipeline(args)
    result = pipeline.run(statement, solve=not args.no_solve)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statement.txt").write_text(statement, encoding="utf-8")
    _write_json(output_dir / "model.json", result.final_model)
    _write_json(output_dir / "validation.json", result.validation_history[-1].summary())
    _write_json(output_dir / "run.json", result)
    if result.valid:
        (output_dir / "model.lp").write_text(write_lp(result.final_model), encoding="utf-8")
    if result.solution is not None:
        _write_json(output_dir / "solution.json", result.solution)

    _print_json(
        {
            "output_dir": str(output_dir),
            "valid": result.valid,
            "repair_attempts": result.repair_attempts,
            "solver_status": result.solution.status.value if result.solution else None,
            "model_sha256": result.model_sha256,
        }
    )
    if not result.valid:
        return 2
    if result.solution is not None and not result.solution.success:
        return 3
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    pipeline = _make_pipeline(args)
    cases = load_cases(args.cases_jsonl)
    summary = run_benchmark(
        cases,
        pipeline,
        objective_tolerance_percent=args.objective_tolerance_percent,
    )
    save_summary(summary, args.output)
    _print_json(summary)
    return 0 if summary.completed_cases == summary.total_cases else 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, OSError, ExtractionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
