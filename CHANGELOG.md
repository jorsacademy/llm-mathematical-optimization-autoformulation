# Changelog

All notable changes will be documented in this file.

## [0.1.0] - 2026-09-03

### Added

- Strict, versioned, fully expanded LP/MILP intermediate representation.
- JSON-encoded untrusted prompts, bounded statement size, and prompt-version provenance.
- Deterministic validation for symbols, names, domains, contradictions, ambiguity, and assumptions.
- Optional OpenAI Responses API structured-output adapter.
- Bounded validation-feedback repair loop with unchanged-model loop protection.
- SciPy/HiGHS compilation and solve path.
- Independent bound, integrality, finiteness, symbol, and constraint verification of returned
  solutions.
- Deterministic LP-format exporter.
- CLI commands for schema inspection, validation, solving, LP rendering, autoformulation, and
  benchmarking.
- Bilingual smoke benchmark and reviewed reference formulations.
- Python 3.11/3.12 CI with pinned GitHub Action commits, lint, format, type, test, and branch coverage
  gates.
- Research context, architecture, evaluation, security, contribution, citation, and noncommercial
  licensing documentation.
