"""Compatible provider/model leaderboard rendering."""

from __future__ import annotations

import csv
import json
from io import StringIO

from autoformulation.benchmark_models import BenchmarkReport


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _leaderboard_rows(reports: list[BenchmarkReport]) -> list[dict[str, str]]:
    ordered = sorted(
        reports,
        key=lambda report: (
            -report.summary.expectation_pass_rate,
            -(report.summary.strict_match_rate or 0.0),
            -report.summary.family_robustness_rate,
            report.system.system_id,
        ),
    )
    rows: list[dict[str, str]] = []
    for report in ordered:
        summary = report.summary
        rows.append(
            {
                "system": report.system.system_id,
                "provider": report.system.provider,
                "model": report.system.model or "—",
                "cases": str(summary.total_cases),
                "expectation_pass": _format_rate(summary.expectation_pass_rate),
                "strict": _format_rate(summary.strict_match_rate),
                "behavioral": _format_rate(summary.behavioral_match_rate),
                "structural": _format_rate(summary.structural_match_rate),
                "abstention": _format_rate(summary.abstention_accuracy),
                "paraphrase_retention": _format_rate(summary.paraphrase_retention_rate),
                "adversarial_retention": _format_rate(summary.adversarial_retention_rate),
                "family_robustness": _format_rate(summary.family_robustness_rate),
                "mean_gap_percent": (
                    "—"
                    if summary.mean_reference_decision_gap_percent is None
                    else f"{summary.mean_reference_decision_gap_percent:.6g}"
                ),
            }
        )
    return rows


def render_leaderboard(
    reports: list[BenchmarkReport],
    *,
    output_format: str = "markdown",
) -> str:
    if not reports:
        raise ValueError("at least one benchmark report is required")
    suite_ids = {report.suite_id for report in reports}
    if len(suite_ids) != 1:
        raise ValueError("all reports must target the same benchmark suite")
    methodology_versions = {report.methodology_version for report in reports}
    if len(methodology_versions) != 1:
        raise ValueError("all reports must use the same methodology version")
    suite_fingerprints = {report.suite_sha256 for report in reports}
    if len(suite_fingerprints) != 1:
        raise ValueError("all reports must use the same immutable suite fingerprint")
    scoring_configs = {
        json.dumps(
            report.scoring.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for report in reports
    }
    if len(scoring_configs) != 1:
        raise ValueError("all reports must use the same scoring configuration")
    rows = _leaderboard_rows(reports)
    headers = list(rows[0])
    if output_format == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    if output_format == "csv":
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()
    if output_format != "markdown":
        raise ValueError(f"unsupported leaderboard format: {output_format}")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row[header] for header in headers) + " |" for row in rows)
    return "\n".join(lines) + "\n"


__all__ = ["render_leaderboard"]
