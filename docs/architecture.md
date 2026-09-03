# Architecture and trust boundaries

## Objective

Convert a natural-language description into a reviewable finite LP/MILP while preventing common
failure modes from being silently promoted to solver input.

## Components

### `schema.py`

Defines the versioned `ModelSpec` intermediate representation. It accepts only linear expressions
made of named variables, scalar coefficients, and constants. Indexed notation must be expanded before
it enters the IR.

### `extractors/`

Defines a provider-neutral contract and an optional OpenAI structured-output adapter. Provider output
must parse directly into `ModelSpec`. The adapter does not request or execute solver source code.

### `validation.py`

Checks semantic properties that JSON Schema alone cannot express:

- duplicate names;
- undefined symbols;
- empty discrete domains;
- constant contradictions;
- duplicate and zero terms;
- unused variables;
- unresolved questions and assumptions.

### `pipeline.py`

Orchestrates extraction, validation, bounded repair, solving, and audit events. An unchanged repair
terminates the loop. Repair is feedback-driven but cannot override the fail-closed treatment of
unresolved questions unless the provider returns a genuinely changed model supported by the original
statement.

### `solver.py`

Compiles the trusted IR to NumPy vectors, a sparse SciPy constraint matrix, and SciPy `milp`, which uses HiGHS. It does not use `eval`,
`exec`, generated imports, or model-generated file paths. The returned point is checked independently
against bounds, integrality, and every constraint.

### `lp_writer.py`

Exports conventional LP text from the same validated IR. It is an inspection and interoperability
artifact, not the source used by the built-in solver.

## Threat model

The natural-language statement and LLM response are untrusted inputs. Relevant threats include:

- prompt injection embedded in a problem statement;
- denial-of-service through oversized statements or unbounded solve/repair loops;
- arbitrary code or shell commands in model output;
- invented coefficients and missing constraints;
- identifier collisions;
- nonfinite values;
- invalid or empty variable domains;
- infinite repair loops;
- API keys accidentally committed to the repository;
- misleading benchmark success based only on objective agreement.

Mitigations currently implemented:

- structured output into a strict schema;
- JSON-encoded untrusted statement payloads and an explicit prompt-injection boundary;
- a default 100,000-character statement cap and a 60-second generated-model solve limit;
- no arbitrary generated-code execution;
- ASCII identifier restrictions;
- finite-number validation;
- deterministic semantic checks;
- fail-closed unresolved questions;
- maximum five repair rounds;
- SHA-256 input/model fingerprints;
- independent solution verification;
- no external provider calls in CI;
- `.env` and common secret-file patterns ignored by Git.

## What is not guaranteed

The pipeline cannot prove that the model captures the user's intended real-world system. In
particular, it cannot generally detect an omitted constraint, a subtly reversed business rule, or a
plausible but unsupported coefficient unless that defect creates a detectable structural conflict.
Consequential models require domain-expert review, scenario tests, and comparison against a manually
reviewed formulation.
