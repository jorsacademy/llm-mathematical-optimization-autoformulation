# Security policy

## Supported version

Security fixes target the current `main` branch until tagged releases are published.

## Reporting a vulnerability

Use GitHub private vulnerability reporting when it is available for this repository. Do not include
API keys, credentials, confidential problem statements, or exploit payloads in a public issue. If
private reporting is unavailable, open a minimal public issue requesting a private reporting channel
without disclosing sensitive details.

## Relevant threat classes

Reports are particularly useful when they concern:

- arbitrary code or command execution;
- prompt-injection paths that cross the structured-output trust boundary;
- path traversal or unintended file writes;
- credential exposure in logs or generated artifacts;
- validation bypasses that allow undefined symbols or nonfinite coefficients;
- solver-result handling that marks an unverified point as successful;
- denial-of-service through unbounded repair or oversized inputs.

The project does not execute LLM-generated source code by design. Proposals that add such execution
must isolate it as an explicitly untrusted, sandboxed research component and are outside the trusted
core.
