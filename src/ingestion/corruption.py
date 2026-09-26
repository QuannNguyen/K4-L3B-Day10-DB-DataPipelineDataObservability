from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Inject a controlled set of corruption patterns and log them for later comparison."""
    corrupted = df.copy().reset_index(drop=True)
    if corrupted.empty:
        return corrupted

    log: dict[str, list[str]] = {
        "dropped_latest_records": [],
        "blanked_summaries": [],
        "noise_injected": [],
        "truncated_titles": [],
        "stale_published_dates": [],
        "duplicate_rows": [],
    }

    drop_count = max(1, len(corrupted) // 5)
    latest_indexes = corrupted.index[:drop_count]
    corrupted = corrupted.drop(index=latest_indexes).reset_index(drop=True)
    log["dropped_latest_records"] = [str(corrupted.loc[index, "paper_id"]) if index in corrupted.index else "" for index in latest_indexes]

    blank_targets = list(range(0, min(len(corrupted), max(1, len(corrupted) // 4))))
    for idx in blank_targets:
        corrupted.loc[idx, "summary"] = ""
        log["blanked_summaries"].append(corrupted.loc[idx, "paper_id"])

    noise_targets = list(range(0, min(len(corrupted), max(1, len(corrupted) // 3))))
    for idx in noise_targets:
        original = corrupted.loc[idx, "text_for_embedding"]
        corrupted.loc[idx, "text_for_embedding"] = f"NOISE!!! {original} ###CORRUPTED###"
        corrupted.loc[idx, "summary"] = f"NOISE!!! {corrupted.loc[idx, 'summary']}"
        log["noise_injected"].append(corrupted.loc[idx, "paper_id"])

    title_targets = list(range(0, min(len(corrupted), max(1, len(corrupted) // 6))))
    for idx in title_targets:
        title = str(corrupted.loc[idx, "title"])
        corrupted.loc[idx, "title"] = title[:7]
        log["truncated_titles"].append(corrupted.loc[idx, "paper_id"])

    stale_targets = list(range(0, min(len(corrupted), max(1, len(corrupted) // 5))))
    for idx in stale_targets:
        old_date = pd.to_datetime(corrupted.loc[idx, "published"]) - pd.Timedelta(days=600)
        corrupted.loc[idx, "published"] = old_date.strftime("%Y-%m-%d")
        log["stale_published_dates"].append(corrupted.loc[idx, "paper_id"])

    duplicate_row = corrupted.iloc[0].copy()
    duplicate_row["paper_id"] = f"{duplicate_row['paper_id']}-duplicate"
    corrupted = pd.concat([corrupted, pd.DataFrame([duplicate_row])], ignore_index=True)
    log["duplicate_rows"].append(str(duplicate_row["paper_id"]))

    if "text_for_embedding" in corrupted.columns:
        corrupted["text_for_embedding"] = corrupted.apply(
            lambda row: " | ".join(
                [
                    f"Title: {row.get('title', '')}",
                    f"Authors: {row.get('authors_joined', '')}",
                    f"Summary: {row.get('summary', '')}",
                    f"Categories: {row.get('categories_joined', '')}",
                    f"Published: {row.get('published', '')}",
                ]
            ),
            axis=1,
        )

    output_path = Path(output_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(log, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return corrupted
