from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from core.config import Settings


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_date(parts: Any) -> str:
    if not parts:
        return ""
    if isinstance(parts, list):
        for item in parts:
            if isinstance(item, list) and item:
                date_values = [int(v) for v in item[:3] if v is not None]
                if len(date_values) >= 3:
                    return datetime(date_values[0], date_values[1], date_values[2]).date().isoformat()
                if len(date_values) >= 2:
                    return datetime(date_values[0], date_values[1], 1).date().isoformat()
                if len(date_values) == 1:
                    return datetime(date_values[0], 1, 1).date().isoformat()
    if isinstance(parts, str):
        return parts[:10]
    return ""


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse Crossref payload into a normalized list of PaperRecord objects."""
    records: list[PaperRecord] = []
    items = payload.get("message", {}).get("items", [])
    for item in items:
        doi = _clean_text(item.get("DOI") or item.get("doi") or "")
        title_value = item.get("title") or []
        title = _clean_text(title_value[0] if isinstance(title_value, list) else title_value)
        abstract_value = item.get("abstract") or item.get("description") or ""
        summary = _clean_text(abstract_value)
        authors: list[str] = []
        for author in item.get("author") or []:
            given = _clean_text(author.get("given"))
            family = _clean_text(author.get("family"))
            name = " ".join(part for part in [given, family] if part)
            if name:
                authors.append(name)
        categories = [
            _clean_text(category)
            for category in (item.get("subject") or item.get("topic") or [])
            if _clean_text(category)
        ]
        primary_category = categories[0] if categories else ""

        published = ""
        published_payload = item.get("published") or {}
        published = _extract_date(published_payload.get("date-parts", []))
        if not published:
            published = _extract_date(item.get("published-print", {}).get("date-parts", []))

        updated = ""
        updated_payload = item.get("updated") or item.get("created") or {}
        if isinstance(updated_payload, dict):
            updated = _clean_text(updated_payload.get("date-time") or updated_payload.get("date") or "")[:10]
        if not updated:
            updated = published

        abs_url = _clean_text(item.get("URL") or "")
        pdf_url = ""
        for link in item.get("link") or []:
            if str(link.get("content-type", "")).lower() == "application/pdf":
                pdf_url = _clean_text(link.get("URL"))
                break
        if not pdf_url:
            pdf_url = abs_url
        comment = _clean_text(item.get("comment") or "")

        if not doi or not title or not summary:
            continue

        records.append(
            PaperRecord(
                paper_id=doi,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                abs_url=abs_url,
                pdf_url=pdf_url,
                comment=comment,
            )
        )
    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Fetch Crossref records, persist the raw API payload, and normalize them to PaperRecord objects."""
    raw_path = settings.paths.raw_api_response
    records_path = settings.paths.raw_records_json
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "query.title": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
        "select": "DOI,title,abstract,author,subject,published,created,updated,URL,link,comment",
    }

    payload: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            response = requests.get("https://api.crossref.org/works", params=params, timeout=30)
            if response.status_code in {429, 503}:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except Exception:
            if attempt == 2:
                break
            time.sleep(1.5)

    if payload is None and raw_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))

    if payload is not None:
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    records = parse_crossref_payload(payload or {"message": {"items": []}})
    records_payload = [
        {
            "paper_id": record.paper_id,
            "title": record.title,
            "summary": record.summary,
            "authors": record.authors,
            "categories": record.categories,
            "primary_category": record.primary_category,
            "published": record.published,
            "updated": record.updated,
            "abs_url": record.abs_url,
            "pdf_url": record.pdf_url,
            "comment": record.comment,
        }
        for record in records
    ]
    records_path.write_text(json.dumps(records_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Load a JSON snapshot and map it to the PaperRecord schema."""
    if not path.exists():
        return []
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, dict):
        items = content.get("message", {}).get("items", [])
        if items:
            return parse_crossref_payload(content)
        return []
    if isinstance(content, list):
        built: list[PaperRecord] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            published = _clean_text(item.get("published") or item.get("published_date") or "")
            updated = _clean_text(item.get("updated") or item.get("updated_date") or published)
            categories = item.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]
            authors = item.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            title = _clean_text(item.get("title") or "")
            summary = _clean_text(item.get("summary") or item.get("abstract") or "")
            paper_id = _clean_text(item.get("paper_id") or item.get("DOI") or item.get("doi") or "")
            if not paper_id or not title or not summary:
                continue
            built.append(
                PaperRecord(
                    paper_id=paper_id,
                    title=title,
                    summary=summary,
                    authors=[_clean_text(author) for author in authors],
                    categories=[_clean_text(category) for category in categories],
                    primary_category=_clean_text(item.get("primary_category") or (categories[0] if categories else "")),
                    published=published[:10] if len(published) >= 10 else published,
                    updated=updated[:10] if len(updated) >= 10 else updated,
                    abs_url=_clean_text(item.get("abs_url") or item.get("URL") or ""),
                    pdf_url=_clean_text(item.get("pdf_url") or item.get("abs_url") or item.get("URL") or ""),
                    comment=_clean_text(item.get("comment") or ""),
                )
            )
        return built
    return []
