# Changelog

All notable changes will be documented in this file.

## [0.2.0] - 2026-09-03

### Added

- Deterministic candidate-to-gold variable alignment with explicit confidence and ambiguity flags.
- Scale-normalized objective and constraint comparison for finite LP/MILP formulations.
- Bidirectional cross-feasibility checks between candidate and gold optima.
- Candidate-decision evaluation under the gold objective and decision-quality gap reporting.
- Semantic error taxonomy for variables, objectives, constraints, assumptions, ambiguity, solver
  failures, and cross-feasibility failures.
- Immutable benchmark-suite and raw-run SHA-256 fingerprints.
- Separate raw generation, offline deterministic scoring, and leaderboard rendering stages.
- Canonical, paraphrase, adversarial, and ambiguous benchmark variants.
- Explicit abstention scoring that rejects provider failures and malformed abstentions.
- Family robustness and conditional paraphrase/adversarial retention metrics.
- Markdown, CSV, and JSON provider/model comparison tables with compatibility checks.
- Variant and tag-level metric slices plus a finite solver time limit in the scoring configuration.
- Bilingual eight-case methodology smoke suite and detailed publication protocol.
- New `benchmark-generate`, `benchmark-score`, and `benchmark-compare` CLI commands.

### Changed

- The legacy inline JSONL benchmark remains available but is no longer the recommended research
  evaluation path.
- Package version advanced to 0.2.0.

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
