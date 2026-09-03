# Benchmarks

`sample_cases.jsonl` contains two small, reviewed smoke cases: a Turkish production-planning LP and
an English transportation LP. They verify end-to-end plumbing and bilingual prompting; they are not
large enough to support research conclusions.

Run with:

```bash
export OPENAI_API_KEY="..."
or-autoformulate benchmark \
  benchmarks/sample_cases.jsonl \
  --model "<model-id>" \
  --output benchmark-results.json
```

Read `docs/evaluation.md` before interpreting results. In particular, objective-value agreement does
not prove semantic equivalence.
