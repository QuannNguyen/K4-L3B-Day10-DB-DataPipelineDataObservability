from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from ingestion.crossref import PaperRecord


def _normalise_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"<.*?>", " ", str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    value = str(value).strip()[:10]
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Normalize raw Crossref records into a clean DataFrame prepared for embedding and evaluation."""
    rows: list[dict[str, object]] = []
    for record in records:
        title = _normalise_text(record.title)
        summary = _normalise_text(record.summary)
        authors = [_normalise_text(author) for author in record.authors if _normalise_text(author)]
        categories = [_normalise_text(category) for category in record.categories if _normalise_text(category)]
        published = _parse_date(record.published)
        updated = _parse_date(record.updated)
        age_days = (run_date.date() - published.date()).days if pd.notna(published) else None

        if not title or not summary or not record.paper_id:
            continue

        rows.append(
            {
                "paper_id": record.paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": _normalise_text(record.primary_category),
                "published": published.strftime("%Y-%m-%d") if pd.notna(published) else "",
                "updated": updated.strftime("%Y-%m-%d") if pd.notna(updated) else "",
                "abs_url": _normalise_text(record.abs_url),
                "pdf_url": _normalise_text(record.pdf_url),
                "comment": _normalise_text(record.comment),
                "authors_joined": ", ".join(authors),
                "categories_joined": ", ".join(categories),
                "summary_chars": len(summary),
                "age_days": age_days,
                "text_for_embedding": " | ".join(
                    [
                        f"Title: {title}",
                        f"Authors: {', '.join(authors) if authors else 'Unknown'}",
                        f"Summary: {summary}",
                        f"Categories: {', '.join(categories) if categories else 'Unknown'}",
                        f"Published: {published.strftime('%Y-%m-%d') if pd.notna(published) else 'Unknown'}",
                    ]
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["paper_id"]).copy()
    df = df[df["title"].notna() & df["summary"].notna()]
    df = df[df["summary"].str.len() >= 20]

    if "published" in df.columns:
        published = pd.to_datetime(df["published"], errors="coerce")
        df["published"] = published.dt.strftime("%Y-%m-%d")
        run_ts = pd.Timestamp(run_date)
        if run_ts.tzinfo is not None:
            run_ts = run_ts.tz_localize(None)
        df["age_days"] = (run_ts - pd.to_datetime(df["published"], errors="coerce")).dt.days

    if "updated" in df.columns:
        df["updated"] = pd.to_datetime(df["updated"], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df.sort_values(["published", "paper_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return df
