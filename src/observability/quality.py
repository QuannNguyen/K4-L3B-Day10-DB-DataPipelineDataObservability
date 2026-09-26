from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd

from core.config import Settings


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Compute freshness metrics and persist them to a JSON report."""
    if df.empty:
        payload = {
            "latest_published": None,
            "oldest_published": None,
            "stale_rows": 0,
            "total_rows": 0,
            "stale_ratio": 0.0,
            "is_fresh": True,
            "threshold_days": settings.freshness_threshold_days,
        }
    else:
        published = pd.to_datetime(df["published"], errors="coerce").dropna()
        stale_rows = int((df["age_days"].fillna(0) > settings.freshness_threshold_days).sum()) if "age_days" in df.columns else 0
        latest = published.max().strftime("%Y-%m-%d") if not published.empty else None
        oldest = published.min().strftime("%Y-%m-%d") if not published.empty else None
        payload = {
            "latest_published": latest,
            "oldest_published": oldest,
            "stale_rows": stale_rows,
            "total_rows": int(len(df)),
            "stale_ratio": round(stale_rows / len(df), 4) if len(df) else 0.0,
            "is_fresh": (stale_rows / len(df) <= 0.25) if len(df) else True,
            "threshold_days": settings.freshness_threshold_days,
        }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run a minimal Great Expectations 1.x suite and freshness check for data quality control."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name=f"{report_name}_source")
    data_asset = data_source.add_dataframe_asset(name=f"{report_name}_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe(f"{report_name}_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name=f"{report_name}_quality_suite")
    context.suites.add(suite)

    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=5000))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="paper_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="paper_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(column="title", min_value=5, max_value=500))
    suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(column="summary", min_value=20, max_value=20000))

    result = batch.validate(expect=suite)
    freshness = build_freshness_report(df, settings, settings.paths.freshness_report if report_name == "baseline" else settings.paths.freshness_report)
    summary = {
        "success": bool(result.success),
        "expectation_count": len(result.results),
        "result": result,
        "freshness": freshness,
    }

    quality_report = settings.paths.baseline_quality_report if report_name == "baseline" else settings.paths.corrupted_quality_report
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(json.dumps({"success": summary["success"], "expectation_count": summary["expectation_count"], "freshness": freshness}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary
