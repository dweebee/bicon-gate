"""Fact-verification and retrieval evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import unquote, urlparse
from typing import Any, Mapping, Sequence


# use our label order, not raw id2label
LABELS = ("CONTRADICTION", "NEUTRAL", "ENTAILMENT")
FV_CONTEXT_TURNS = 2


@dataclass(frozen=True)
class ClassificationScores:
    accuracy: float
    macro_f1: float
    macro_recall: float
    class_f1: dict[str, float]
    class_recall: dict[str, float]


def normalise_gold_label(label: Any) -> str:
    # DialFact and model wrappers name the same three labels differently
    value = str(label or "").strip().upper()
    if value in {"SUPPORTS", "SUPPORTED", "ENTAILMENT"}:
        return "ENTAILMENT"
    if value in {"REFUTES", "REFUTED", "CONTRADICTION"}:
        return "CONTRADICTION"
    if value in {"NOT ENOUGH INFO", "NEI", "NEUTRAL", "UNKNOWN"}:
        return "NEUTRAL"
    raise ValueError(f"Unsupported DialFact label: {label!r}")


def serialise_gold_evidence(evidence_list: Sequence[Any]) -> str:
    # keep mapping and tuple forms, DialFact dumps differ here
    parts: list[str] = []
    for index, evidence in enumerate(evidence_list or []):
        if isinstance(evidence, Mapping):
            title = str(evidence.get("title") or evidence.get("page") or "")
            text = str(evidence.get("text") or evidence.get("sentence") or evidence.get("evidence") or "")
        else:
            values = (
                list(evidence)
                if isinstance(evidence, Sequence) and not isinstance(evidence, str)
                else [evidence]
            )
            title = str(values[0]) if values else ""
            text = str(values[2]) if len(values) > 2 else (str(values[1]) if len(values) > 1 else "")
        # keep source order and numbering, verifier input is not a set
        parts.append(f"[EVIDENCE - {index}] ({title}) {text}")
    return ", ".join(parts)


def build_verifier_hypothesis(
    context: Sequence[str],
    claim: str,
    *,
    context_turns: int = FV_CONTEXT_TURNS,
) -> str:
    """Format the verifier hypothesis with the final context turns."""
    turns = [str(turn).strip() for turn in context if str(turn).strip()]
    # FV uses the final two turns, retrieval uses full context
    selected = turns[-context_turns:] if context_turns > 0 else []
    if selected:
        # [EOT] is part of verifier formatting
        return f"[CONTEXT]: {' [EOT] '.join(selected)} [RESPONSE]: {str(claim).strip()}"
    return f"[RESPONSE]: {str(claim).strip()}"


def build_fv_pair(
    sample: Mapping[str, Any],
    claim: str,
    *,
    context_turns: int = FV_CONTEXT_TURNS,
) -> tuple[str, str]:
    premise = serialise_gold_evidence(sample.get("evidence_list") or [])
    hypothesis = build_verifier_hypothesis(sample.get("context") or [], claim, context_turns=context_turns)
    return premise, hypothesis


def build_e2e_pair(
    sample: Mapping[str, Any],
    claim: str,
    retrieved_passage: Mapping[str, Any] | Any,
    *,
    context_turns: int = FV_CONTEXT_TURNS,
) -> tuple[str, str]:
    if isinstance(retrieved_passage, Mapping):
        title = str(retrieved_passage.get("title") or "")
        text = str(retrieved_passage.get("text") or "")
    else:
        title = str(getattr(retrieved_passage, "title", ""))
        text = str(getattr(retrieved_passage, "text", ""))
    # E2E changes the premise only, keep the FV hypothesis format
    premise = f"[EVIDENCE - 0] ({title}) {text}"
    hypothesis = build_verifier_hypothesis(sample.get("context") or [], claim, context_turns=context_turns)
    return premise, hypothesis


def classification_scores(gold: Sequence[str], predicted: Sequence[str]) -> ClassificationScores:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be non-empty and have equal length.")

    gold = [normalise_gold_label(label) for label in gold]
    predicted = [normalise_gold_label(label) for label in predicted]
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / len(gold)

    # macro average always includes all three classes
    class_f1: dict[str, float] = {}
    class_recall: dict[str, float] = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_recall[label] = recall
        class_f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ClassificationScores(
        accuracy=accuracy,
        macro_f1=sum(class_f1.values()) / len(LABELS),
        macro_recall=sum(class_recall.values()) / len(LABELS),
        class_f1=class_f1,
        class_recall=class_recall,
    )


def evaluate_pairs(
    verifier: Any,
    pairs: Sequence[tuple[str, str]],
    gold: Sequence[str],
    *,
    batch_size: int = 16,
) -> ClassificationScores:
    predicted = verifier.predict_labels(pairs, batch_size=batch_size)
    return classification_scores(gold, predicted)


def _normalise_page(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().split())


def _normalise_url(value: Any) -> str:
    raw = unquote(str(value or "").strip())
    if not raw:
        return ""
    parsed = urlparse(raw)
    # mobile and www aliases should resolve to the same page
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    path = parsed.path.rstrip("/").lower()
    return f"{host}{path}" if host else raw.lower().rstrip("/")


def _gold_documents(sample: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return unique (normalised title, normalised URL) gold documents."""
    documents: list[tuple[str, str]] = []
    seen: set[str] = set()
    for evidence in sample.get("evidence_list") or []:
        if isinstance(evidence, Mapping):
            title = evidence.get("title") or evidence.get("page") or evidence.get("page_title")
            url = evidence.get("url")
        else:
            values = (
                list(evidence)
                if isinstance(evidence, Sequence) and not isinstance(evidence, str)
                else [evidence]
            )
            title = values[0] if values else None
            url = values[1] if len(values) > 1 else None
        title_key = _normalise_page(title)
        url_key = _normalise_url(url)
        # URL first, normalised title as fallback
        key = f"url::{url_key}" if url_key else f"title::{title_key}"
        if (title_key or url_key) and key not in seen:
            seen.add(key)
            documents.append((title_key, url_key))
    return documents


def unique_ranked_documents(predictions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Collapse passage hits to first-occurrence document ranks."""
    unique: list[Mapping[str, Any]] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    # one page can have several passages, first rank only
    for prediction in predictions:
        title = _normalise_page(prediction.get("title"))
        url = _normalise_url(prediction.get("url"))
        if not title and not url:
            continue
        if (title and title in seen_titles) or (url and url in seen_urls):
            continue
        if title:
            seen_titles.add(title)
        if url:
            seen_urls.add(url)
        unique.append(prediction)
    return unique


def retrieval_relevance(sample: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> list[int]:
    gold = _gold_documents(sample)
    documents = unique_ranked_documents(predictions)
    relevance: list[int] = []
    for prediction in documents:
        title = _normalise_page(prediction.get("title"))
        url = _normalise_url(prediction.get("url"))
        # relevance is binary at document level
        relevant = any(
            (url and gold_url and url == gold_url)
            or (title and gold_title and title == gold_title)
            for gold_title, gold_url in gold
        )
        relevance.append(int(relevant))
    return relevance


def recall_at_k(relevance: Sequence[int], number_of_gold_pages: int, k: int) -> float:
    if number_of_gold_pages <= 0:
        return 0.0
    # cap aliases at the number of distinct gold pages
    return min(sum(relevance[:k]), number_of_gold_pages) / number_of_gold_pages


def ndcg_at_k(relevance: Sequence[int], number_of_gold_pages: int, k: int) -> float:
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance[:k]))
    ideal = [1] * min(number_of_gold_pages, k)
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def aggregate_ir_metrics(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    k: int,
) -> dict[str, float]:
    """Aggregate document-level retrieval metrics."""
    if len(samples) != len(predictions) or not samples:
        raise ValueError("samples and predictions must be non-empty and aligned.")

    macro_recall: list[float] = []
    macro_ndcg: list[float] = []
    total_hits = 0
    total_gold = 0
    queries_with_hit = 0
    evaluated_queries = 0

    for sample, ranked in zip(samples, predictions):
        gold_documents = _gold_documents(sample)
        if not gold_documents:
            # skip non-factual/no-page rows for document IR
            continue
        number_of_gold_documents = len(gold_documents)
        evaluated_queries += 1
        relevance = retrieval_relevance(sample, ranked)
        hits = min(sum(relevance[:k]), number_of_gold_documents)
        total_hits += hits
        total_gold += number_of_gold_documents
        queries_with_hit += int(hits > 0)
        macro_recall.append(recall_at_k(relevance, number_of_gold_documents, k))
        macro_ndcg.append(ndcg_at_k(relevance, number_of_gold_documents, k))

    if evaluated_queries == 0:
        raise ValueError("No samples contain document-level gold evidence.")

    return {
        f"macro_recall@{k}": sum(macro_recall) / evaluated_queries,
        f"macro_ndcg@{k}": sum(macro_ndcg) / evaluated_queries,
        f"micro_recall@{k}": total_hits / total_gold,
        # table name kept as 1-ZHR, this is query hit rate
        f"1-zhr@{k}": queries_with_hit / evaluated_queries,
    }
