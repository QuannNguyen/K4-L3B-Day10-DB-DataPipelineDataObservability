from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Create a benchmark test set covering summary, author, date, and category questions."""
    if df.empty:
        return []

    sample = df.head(10).copy().reset_index(drop=True)
    if len(sample) < 4:
        sample = df.copy().reset_index(drop=True)

    test_set: list[dict[str, Any]] = []
    for idx, row in sample.iterrows():
        paper_id = str(row["paper_id"])
        title = str(row["title"])
        authors = str(row.get("authors_joined", ""))
        categories = str(row.get("categories_joined", ""))
        published = str(row.get("published", ""))
        summary = str(row.get("summary", ""))

        question_types = [
            ("summary", f"What is the main idea of the paper '{title}'?", summary),
            ("authors", f"Who authored the paper '{title}'?", authors),
            ("date", f"When was the paper '{title}' published?", published),
            ("categories", f"What categories are associated with the paper '{title}'?", categories),
        ]

        for q_type, question, ground_truth in question_types:
            test_set.append(
                {
                    "id": f"{paper_id}-{q_type}",
                    "question_type": q_type,
                    "question": question,
                    "ground_truth": ground_truth,
                    "ground_truth_doc_ids": [paper_id],
                }
            )

    test_set = test_set[:10]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(test_set, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return test_set
