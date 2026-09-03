# LLM Mathematical Optimization Autoformulation

[![CI](https://github.com/jorsacademy/llm-mathematical-optimization-autoformulation/actions/workflows/ci.yml/badge.svg)](https://github.com/jorsacademy/llm-mathematical-optimization-autoformulation/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)

A verification-first research laboratory for translating natural-language Operations Research
problems into solver-ready linear and mixed-integer linear models.

This repository does **not** treat syntactically valid LLM output as a correct optimization model.
The pipeline forces model output through a typed intermediate representation, deterministic semantic
checks, a bounded repair loop, a real optimizer, and independent post-solve feasibility checks.

> **Status:** research prototype; finite LP/MILP only. The project is source-available for
> noncommercial use and is not OSI Open Source.

## Why this project exists

Recent autoformulation systems show that LLMs can help extract mathematical models, generate solver
artifacts, diagnose errors, and iterate on formulations. The hard part is reliability: a formulation
may parse, run, and even return an “optimal” solution while encoding the wrong objective, omitting a
constraint, inventing a parameter, or using the wrong variable domain.

The project therefore separates five concerns:

1. **Semantic extraction** from natural language.
2. **A constrained model representation** that cannot contain arbitrary executable code.
3. **Static verification** of symbols, domains, duplicate names, contradictions, and unresolved
   business questions.
4. **Solver grounding** through SciPy/HiGHS.
5. **Independent solution verification** after the solver returns.

## Architecture

```text
Natural-language statement
          │
          ▼
Schema-constrained LLM extraction
          │
          ▼
ModelSpec: flat, finite LP/MILP IR
          │
          ├── deterministic validation ── errors? ──► bounded LLM repair
          │                                      ▲              │
          │                                      └──────────────┘
          ▼
Safe compiler written by this project
          │
          ├── LP-format export
          └── SciPy/HiGHS solve
                       │
                       ▼
             independent feasibility,
             bound, and integrality check
```

The LLM never supplies Python that is executed. Solver matrices are compiled from validated JSON by
trusted project code.

## Reliability properties

- **Fail-closed ambiguity:** entries in `unresolved_questions` are validation errors by default.
- **Prompt-injection boundary:** problem statements are JSON-encoded as untrusted data, and the
  system contract rejects embedded role/tool/code instructions.
- **No hidden numeric completion:** prompts explicitly prohibit inventing missing numbers, sets,
  bounds, units, or rules.
- **Strict schema:** unknown JSON fields, invalid identifiers, nonfinite numbers, and reversed bounds
  are rejected by Pydantic.
- **Symbol checks:** undefined variables and duplicate variable/constraint names are errors.
- **Domain checks:** empty integer and binary domains are detected before solving.
- **Constant-constraint checks:** impossible constant constraints are rejected; redundant ones are
  reported.
- **Bounded input and repair:** statements default to a 100,000-character cap; repair defaults to
  one round with a hard limit of five.
- **Loop protection:** an unchanged repair terminates the repair cycle.
- **Auditability:** input and final-model SHA-256 fingerprints are written to the run record.
- **Post-solve verification:** solver output is rechecked against every bound, integrality condition,
  and constraint.
- **Deterministic CI:** tests never call an external LLM and never run expensive training.

These controls reduce predictable failure modes; they do not prove semantic equivalence between the
text and the generated optimization model. Human review remains necessary for consequential use.

## Scope

Version 0.1 supports fully expanded, finite:

- linear programs;
- mixed-integer linear programs;
- continuous, integer, and binary variables;
- minimization and maximization;
- `<=`, `>=`, and equality constraints;
- LP-format export and SciPy/HiGHS solving.

The following are intentionally out of scope in the first release:

- nonlinear, quadratic, conic, or logical expressions;
- symbolic indexed sets and summations;
- stochastic, robust, chance-constrained, or multi-stage models;
- arbitrary model-generated code execution;
- automatic acceptance of assumptions or missing business data;
- claims that objective-value agreement alone proves formulation correctness.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For runtime use without development tools:

```bash
python -m pip install -e "."
```

OpenAI integration is optional:

```bash
python -m pip install -e ".[openai]"
```

## Deterministic local workflow

Validate a reviewed model:

```bash
or-autoformulate validate examples/production_planning.model.json
```

Solve it:

```bash
or-autoformulate solve examples/production_planning.model.json
```

Export conventional LP format:

```bash
or-autoformulate render-lp \
  examples/production_planning.model.json \
  --output production_planning.lp
```

Print the machine-readable schema supplied to structured-output providers:

```bash
or-autoformulate schema > model-spec.schema.json
```

## LLM autoformulation workflow

The implementation uses the OpenAI Responses API structured-output interface when the optional
provider dependency is installed. A model ID is deliberately not hardcoded because availability,
access, and model aliases change over time.

```bash
export OPENAI_API_KEY="..."
export AUTOFORMULATION_MODEL="<model-id-available-to-your-account>"

or-autoformulate formulate \
  examples/production_planning_tr.txt \
  --repair-rounds 1 \
  --output-dir runs/production_planning
```

Equivalent explicit model selection:

```bash
or-autoformulate formulate statement.txt --model "<model-id>" -o runs/case-001
```

A run directory contains:

```text
statement.txt      exact submitted problem statement
model.json         final typed intermediate representation
validation.json    final deterministic validation report
model.lp           LP-format export, written only for a valid model
solution.json      solver result and post-solve checks, when solving is enabled
run.json           fingerprints, provider/model metadata, validation history, and stage events
```

Use `--no-solve` to stop after formulation and validation. Use `--strict-assumptions` to convert every
reported modeling assumption from a warning to an error.

## Intermediate representation

The model is deliberately flat and explicit. An abbreviated example is:

```json
{
  "schema_version": "1.0",
  "name": "production_planning",
  "problem_summary": "Choose production quantities to maximize contribution.",
  "source_language": "en",
  "variables": [
    {
      "name": "tables",
      "description": "Number of tables produced.",
      "variable_type": "continuous",
      "lower_bound": 0,
      "upper_bound": null,
      "unit": "units",
      "source_excerpt": "Production quantities are nonnegative."
    }
  ],
  "objective": {
    "sense": "maximize",
    "description": "Maximize contribution.",
    "expression": {
      "terms": [{"variable": "tables", "coefficient": 40}],
      "constant": 0
    }
  },
  "constraints": [],
  "assumptions": [],
  "unresolved_questions": []
}
```

The full schema is defined in [`src/autoformulation/schema.py`](src/autoformulation/schema.py).

## Benchmarking

A small bilingual smoke benchmark is included:

```bash
or-autoformulate benchmark \
  benchmarks/sample_cases.jsonl \
  --model "<model-id>" \
  --output benchmark-results.json
```

Reported metrics include:

- provider completion rate;
- static model-validity rate;
- optimizer solve rate;
- objective agreement with a reviewed reference model;
- exact normalized-model match;
- variable and constraint count deltas.

Objective agreement is only a coarse solver-grounded signal. Two semantically different models can
share an objective value, and two equivalent models can use different variable names or redundant
constraints. See [`docs/evaluation.md`](docs/evaluation.md) before reporting benchmark results.

## Repository structure

```text
src/autoformulation/
├── schema.py          typed LP/MILP intermediate representation
├── validation.py      deterministic static checks
├── solver.py          SciPy/HiGHS compiler and post-solve verification
├── lp_writer.py       deterministic LP-format exporter
├── prompts.py         extraction and repair contracts
├── pipeline.py        bounded extract/validate/repair/solve orchestration
├── benchmark.py       solver-grounded benchmark utilities
├── cli.py             command-line interface
└── extractors/
    ├── base.py        provider-neutral contract
    └── openai.py      optional structured-output adapter
```

## Research positioning, 2019–August 2026

This repository is informed by the shift from natural-language entity extraction toward modular,
solver-grounded, and search-based optimization modeling:

- **NL4Opt** formalized natural-language optimization formulation as a benchmark and competition
  task: Ramamonjison et al., *NL4Opt Competition*, PMLR 220, 2023.
- **OptiMUS** developed a modular agent that formulates, writes, debugs, solves, and evaluates
  (MI)LP models: AhmadiTeshnizi, Gao, and Udell, ICML 2024.
- **Mamo** introduced solver-integrated evaluation for mathematical modeling, including LP/MILP
  tasks: Huang et al., 2024.
- **ORLM** trained open models for optimization modeling and introduced OR-Instruct and IndustryOR:
  Huang et al., *Operations Research* 73(6), 2025.
- **Autoformulation with LLMs and MCTS** treated formulation as structured search with symbolic
  pruning: Astorga et al., ICML 2025.
- **ORQA** evaluated multistep Operations Research reasoning and documented a remaining gap between
  general-purpose LLMs and OR expertise: Mostajabdaveh et al., AAAI 2025.
- **OPT-ENGINE** and **FrontierOR** are 2026 preprints that emphasize controllable complexity and
  scalable algorithm design; they are useful research signals but should not be presented as settled
  peer-reviewed evidence.

See [`docs/research-context.md`](docs/research-context.md) for links, distinctions, and design
implications.

## Reproducibility and tests

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=autoformulation --cov-report=term-missing
```

CI runs Python 3.11 and 3.12. It performs formatting, linting, strict type checking, unit tests, and
branch coverage enforcement. External LLM calls are replaced with deterministic fake clients.

## Roadmap

The next technically meaningful extensions are:

1. an indexed algebraic IR with controlled expansion;
2. table/CSV parameter ingestion with provenance at cell level;
3. dimensional and unit-consistency checks;
4. candidate search rather than single-draft prompting;
5. IIS/conflict feedback for infeasible generated models;
6. adapters for NL4Opt, Mamo, IndustryOR, ORQA-derived reasoning checks, OPT-ENGINE, and FrontierOR;
7. semantic constraint matching beyond objective-value comparison;
8. provider-neutral local-model adapters;
9. sandboxed solver-code comparison as a separate, explicitly untrusted research track.

## License

Licensed under **PolyForm Noncommercial 1.0.0**. Commercial use is not granted. This makes the
project source-available rather than OSI Open Source. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
