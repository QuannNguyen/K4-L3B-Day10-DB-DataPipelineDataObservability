from __future__ import annotations

from datetime import UTC, datetime

from core.config import load_settings
from core.utils import read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records, load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex
from retrieval.qa import answer_question


def main() -> None:
    """Run the baseline ingest → clean → index → evaluate → quality pipeline."""
    settings = load_settings()
    raw_records_path = settings.paths.raw_records_json
    if settings.refresh_source or not raw_records_path.exists():
        fetch_source_records(settings)
    records = load_raw_records(raw_records_path)
    if not records:
        raise RuntimeError(f"No raw records were loaded from {raw_records_path}.")

    clean_df = build_clean_dataframe(records, datetime.now(UTC))
    write_csv(clean_df, settings.paths.clean_csv)
    write_json(settings.paths.clean_json, clean_df.to_dict(orient="records"))

    index = LocalEmbeddingIndex.build(clean_df, settings, settings.paths.embeddings_json)

    test_set = build_test_set(clean_df, settings.paths.eval_testset)
    source_summary = {
        "source": settings.source_api,
        "query": settings.source_query,
        "records_loaded": len(records),
        "clean_rows": len(clean_df),
    }
    bundle = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )

    quality = run_data_quality_checks(clean_df, settings, "baseline")
    freshness = build_freshness_report(clean_df, settings, settings.paths.freshness_report)
    generate_phase1_report(settings.paths.baseline_report, source_summary, bundle.summary, quality, freshness)

    sample_question = test_set[0]["question"] if test_set else "What is this paper about?"
    answer = answer_question(sample_question, settings=settings, index=index)
    print(f"Baseline sample answer: {answer.answer}")
    print(f"retrieval_hit_rate={bundle.summary.get('retrieval_hit_rate', 0.0):.4f}")
    print(f"mean_token_f1={bundle.summary.get('mean_token_f1', 0.0):.4f}")
    print(f"quality_success={quality.get('success', False)}")
