# Research context: LLMs for optimization modeling, 2019–August 2026

This note positions the repository within automated Operations Research modeling. It is not a
systematic-review claim and does not treat every preprint as equally established evidence.

## From semantic parsing to end-to-end modeling

The NL4Opt competition separated natural-language optimization modeling into machine-interpretable
subtasks and provided a public benchmark for extracting optimization formulations. Its scope helped
make evaluation more concrete, but benchmark correctness and real-world modeling correctness remain
different problems.

- Ramamonjison et al. (2023), “NL4Opt Competition: Formulating Optimization Problems Based on Their
  Natural Language Descriptions,” PMLR 220:
  https://proceedings.mlr.press/v220/ramamonjison23a.html
- Competition and dataset:
  https://nl4opt.github.io/
  https://github.com/nl4opt/nl4opt-competition

OptiMUS moved toward a modular workflow that develops mathematical models, writes and debugs solver
code, evaluates solutions, and iterates using evaluation feedback.

- AhmadiTeshnizi, Gao, and Udell (2024), “OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers
  and Large Language Models,” ICML 2024:
  https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html

## Solver-grounded evaluation and domain-specific models

Mamo evaluates mathematical modeling with solver integration rather than relying only on text
similarity. Its optimization subset covers easy and complex LP/MILP tasks, while the broader dataset
also contains differential-equation modeling.

- Huang et al. (2024), “Mamo: a Mathematical Modeling Benchmark with Solvers”:
  https://arxiv.org/abs/2405.13144

ORLM focuses on training open models for optimization modeling, proposing OR-Instruct and the
IndustryOR benchmark. The journal version appeared in Operations Research in 2025.

- Huang et al. (2025), “ORLM: A Customizable Framework in Training Large Models for Automated
  Optimization Modeling,” Operations Research 73(6), DOI 10.1287/opre.2024.1233:
  https://pubsonline.informs.org/doi/10.1287/opre.2024.1233
- Public implementation:
  https://github.com/Cardinal-Operations/ORLM

ORQA evaluates whether LLMs can reason through the components and relationships of Operations
Research models. It is a reasoning benchmark rather than a complete executable-autoformulation
benchmark.

- Mostajabdaveh et al. (2025), “Evaluating LLM Reasoning in the Operations Research Domain with
  ORQA,” AAAI 2025, DOI 10.1609/aaai.v39i23.34673:
  https://ojs.aaai.org/index.php/AAAI/article/view/34673

## Autoformulation as search

Astorga et al. frame autoformulation as a hierarchical search problem and combine LLM proposals with
Monte Carlo Tree Search, LLM evaluation, and symbolic pruning. This motivates a roadmap beyond a
single candidate formulation.

- Astorga et al. (2025), “Autoformulation of Mathematical Optimization Models Using Large Language
  Models,” ICML 2025:
  https://proceedings.mlr.press/v267/astorga25a.html

## 2026 research signals

Two 2026 preprints sharpen evaluation questions but should be labeled as preprints unless and until a
peer-reviewed version is identified:

- “Benchmarking the Limits of LLMs in Optimization Modeling” introduces OPT-ENGINE with
  controllable complexity:
  https://arxiv.org/abs/2601.19924
- “FrontierOR: Benchmarking LLMs' Capacity for Efficient Operations Research” emphasizes scalable
  algorithm design rather than only direct formulation-and-solve behavior:
  https://arxiv.org/abs/2605.25246

## Design implications for this repository

The literature motivates the following choices:

1. **Structured output is necessary but insufficient.** Schema validity prevents malformed output,
   not omitted constraints or wrong business semantics.
2. **Solver execution is necessary but insufficient.** A wrong model may still be feasible and
   optimally solved.
3. **Code execution is a separate trust boundary.** This project compiles a constrained IR rather
   than executing arbitrary code emitted by a model.
4. **Ambiguity must remain visible.** Missing data is recorded as an unresolved question and blocks
   solving by default.
5. **Evaluation must be multidimensional.** Parse rate, validation rate, solve rate, feasibility,
   objective agreement, and semantic model comparison measure different failure modes.
6. **Search and repair must be bounded.** Unlimited self-repair can hide cost, repeat errors, and
   create non-reproducible behavior.
7. **Benchmarks need provenance and reviewed references.** Automatically generated reference models
   should not be treated as ground truth merely because a solver accepts them.
