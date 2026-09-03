# Contributing

Contributions should improve correctness, auditability, reproducibility, or evaluation quality. A
larger agent workflow is not automatically an improvement if it weakens trust boundaries or hides
failure modes.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete local quality gate:

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=autoformulation --cov-report=term-missing
```

## Pull-request requirements

- Add tests for new validation, compilation, provider, or evaluation behavior.
- Keep external LLM calls out of CI; use deterministic fake clients.
- Do not add `eval`, `exec`, shell execution, or arbitrary execution of model-generated code to the
  trusted pipeline.
- Do not silently downgrade unresolved questions or invented-data findings.
- Document any new solver assumptions, tolerances, or status mappings.
- Use human-reviewed reference models for benchmark additions.
- Do not include benchmark test answers in extraction or repair prompts.
- Do not commit API keys, provider responses containing confidential data, or commercial datasets.
- Update `CHANGELOG.md` for user-visible changes.

## Licensing

Unless explicitly agreed otherwise in writing, contributions are provided under the repository's
PolyForm Noncommercial 1.0.0 terms and must preserve the Required Notice.
