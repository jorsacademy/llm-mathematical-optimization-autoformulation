# Evaluation protocol

## Why one metric is insufficient

Autoformulation has several distinct failure layers:

1. provider/API completion;
2. schema parsing;
3. deterministic semantic validation;
4. solver compilation;
5. solver termination;
6. feasibility and integrality of the returned solution;
7. mathematical equivalence to a reference formulation;
8. semantic fidelity to the original business problem.

A benchmark should report these layers separately.

## Included metrics

The built-in JSONL runner reports:

- **completion rate:** a provider response was obtained and processed;
- **validity rate:** the final model has no deterministic validation errors;
- **solve rate:** the candidate model solved to a verified optimum;
- **objective match rate:** the candidate and reviewed reference objective values agree within a
  specified relative tolerance;
- **exact model match:** canonical JSON fingerprints match;
- **variable/constraint count deltas:** coarse indicators of structural divergence.

## Interpretation limits

Objective agreement is not semantic equivalence. Examples:

- a missing constraint may be inactive at the reference optimum;
- a wrong coefficient may coincidentally yield the same objective;
- multiple optimal solutions can have different decision vectors;
- equivalent formulations can introduce auxiliary variables or redundant constraints;
- variable renaming breaks exact JSON matching without changing the mathematics.

Therefore, objective match should be presented as a solver-grounded smoke metric, not “accuracy.”

## Recommended research protocol

For credible experiments:

1. Freeze the provider model identifier or dated snapshot when available.
2. Record temperature/reasoning settings if the provider exposes them.
3. Record prompt and schema versions.
4. Run multiple seeds or repeated calls for stochastic providers.
5. Keep a human-reviewed reference model and its provenance.
6. Report parse, validation, solve, feasibility, objective, and semantic metrics separately.
7. Categorize errors: missing variable, wrong domain, wrong objective direction, missing constraint,
   coefficient error, relation error, unsupported nonlinearity, invented data, or unresolved input.
8. Retain failed outputs; excluding them biases results.
9. Report token usage, latency, and cost where available.
10. Avoid using benchmark test answers in prompts or repair feedback.

## Sample format

Each JSONL line contains an identifier, the exact statement, and a nested reviewed `reference_model`:

```json
{"id":"case-001","statement":"...","reference_model":{"schema_version":"1.0"}}
```

The bundled cases are smoke tests authored for this repository. They are not a substitute for
NL4Opt, Mamo, IndustryOR, ORQA, OPT-ENGINE, or FrontierOR evaluation.
