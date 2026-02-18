from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

SYNONYM_GROUPS: dict[str, set[str]] = {
    "time": {"time", "timestamp", "datetime", "date", "측정일시", "수집일시", "발생일시", "일시", "시간", "날짜"},
    "lat": {"lat", "latitude", "위도", "gps_lat"},
    "lon": {"lon", "lng", "longitude", "경도", "gps_lon"},
    "temp": {"temp", "temperature", "온도"},
    "hum": {"hum", "humidity", "습도"},
    "id": {"id", "uuid", "device_id", "sensor_id", "robot_id", "식별자"},
    "power": {"power", "전력", "watt", "kw"},
}


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    normalized = unicodedata.normalize("NFKC", lowered)
    normalized = re.sub(r"[\s\-_/()\[\],.:;]+", "", normalized)
    return normalized


def expand_term_with_synonyms(term: str) -> set[str]:
    norm = normalize_text(term)
    expanded = {norm}
    for words in SYNONYM_GROUPS.values():
        normalized_words = {normalize_text(w) for w in words}
        if norm in normalized_words:
            expanded.update(normalized_words)
    return expanded


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def rank_column_candidates(query: str, columns: list[str]) -> list[tuple[str, float]]:
    if not columns:
        return []

    query_terms = [t for t in re.split(r"\s+|,|/", query) if t.strip()]
    expanded_terms: set[str] = set()
    for term in query_terms:
        expanded_terms.update(expand_term_with_synonyms(term))

    scored: list[tuple[str, float]] = []
    for col in columns:
        best = max((similarity(term, col) for term in expanded_terms), default=0.0)
        scored.append((col, best))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def find_column_candidates(query: str, columns: list[str], top_n: int = 10) -> list[str]:
    ranked = rank_column_candidates(query, columns)
    return [col for col, score in ranked if score >= 0.45][:top_n]
