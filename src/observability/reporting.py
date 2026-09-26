from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    """Render a markdown summary of the baseline phase, including metrics and quality checks."""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1 Baseline Report",
        "",
        "## Source summary",
        f"- Source: {source_summary.get('source', 'Unknown')}",
        f"- Query: {source_summary.get('query', 'Unknown')}",
        f"- Records loaded: {source_summary.get('records_loaded', 0)}",
        "",
        "## Retrieval and evaluation metrics",
        f"- retrieval_hit_rate: {metrics.get('retrieval_hit_rate', 0.0):.4f}",
        f"- mean_token_f1: {metrics.get('mean_token_f1', 0.0):.4f}",
        f"- judge_accuracy: {metrics.get('judge_accuracy', 0.0):.4f}",
        f"- mean_judge_score: {metrics.get('mean_judge_score', 0.0):.4f}",
        "",
        "## Data quality",
        f"- Success: {quality.get('success', False)}",
        f"- Expectations evaluated: {quality.get('expectation_count', 0)}",
        "",
        "## Freshness report",
        f"- Latest published: {freshness.get('latest_published', 'N/A')}",
        f"- Oldest published: {freshness.get('oldest_published', 'N/A')}",
        f"- Stale rows: {freshness.get('stale_rows', 0)} / {freshness.get('total_rows', 0)}",
        f"- Ratio stale: {freshness.get('stale_ratio', 0.0):.4f}",
        f"- Is fresh: {freshness.get('is_fresh', False)}",
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """Render a markdown comparison among baseline, corrupted, and repaired states."""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    def metric_value(metrics: dict[str, Any], field: str) -> str:
        return f"{metrics.get(field, 0.0):.4f}" if isinstance(metrics.get(field, 0.0), (int, float)) else str(metrics.get(field, "N/A"))

    lines = [
        "# Corruption Comparison Report",
        "",
        "| State | retrieval_hit_rate | mean_token_f1 | judge_accuracy | mean_judge_score | quality_success | stale_ratio | is_fresh |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Baseline | {metric_value(baseline_metrics, 'retrieval_hit_rate')} | {metric_value(baseline_metrics, 'mean_token_f1')} | {metric_value(baseline_metrics, 'judge_accuracy')} | {metric_value(baseline_metrics, 'mean_judge_score')} | {str(bool(baseline_metrics.get('quality_success', True)))} | 0.0000 | True |",
        f"| Corrupted | {metric_value(corrupted_metrics, 'retrieval_hit_rate')} | {metric_value(corrupted_metrics, 'mean_token_f1')} | {metric_value(corrupted_metrics, 'judge_accuracy')} | {metric_value(corrupted_metrics, 'mean_judge_score')} | {str(corrupted_quality.get('success', False))} | {corrupted_freshness.get('stale_ratio', 0.0):.4f} | {corrupted_freshness.get('is_fresh', False)} |",
        f"| Repaired | {metric_value(repaired_metrics, 'retrieval_hit_rate')} | {metric_value(repaired_metrics, 'mean_token_f1')} | {metric_value(repaired_metrics, 'judge_accuracy')} | {metric_value(repaired_metrics, 'mean_judge_score')} | {str(repaired_quality.get('success', False))} | {repaired_freshness.get('stale_ratio', 0.0):.4f} | {repaired_freshness.get('is_fresh', False)} |",
        "",
        "## Interpretation",
        "- Baseline provides the healthy reference performance for the clean corpus.",
        "- Corrupted data simulates silent degradation through missing summaries, stale dates, duplicate rows, and noisy text.",
        "- Repaired data restores the corpus from the trusted raw snapshot and should recover around the baseline quality bar.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
