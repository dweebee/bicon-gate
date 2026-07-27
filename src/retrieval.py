"""Wikipedia passage indexing and BM25-E5-BGE retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


# BM25 gets 300, E5 only sees 180, keep these separate
WINDOW_SIZE = 100
STRIDE = 50
BM25_K1 = 1.5
BM25_B = 0.4
BM25_RETRIEVE = 300
BM25_WORKING_DEPTH = 180
E5_SCORE_DEPTH = 20
E5_KEEP = 10


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    url: str
    text: str
    score: float | None = None


def build_ir_query(context: Sequence[str], claim: str) -> str:
    # IR uses full context, last-two-turn is FV only
    parts = [str(turn).strip() for turn in context if str(turn).strip()]
    if str(claim or "").strip():
        parts.append(str(claim).strip())
    return " ".join(parts)


def iter_passages(
    *,
    page_id: str,
    title: str,
    url: str,
    tokens: Sequence[str],
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Iterable[Passage]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    token_list = [str(token) for token in tokens]
    # keep the short tail window but do not emit it twice
    for start in range(0, len(token_list), stride):
        chunk = token_list[start:start + window_size]
        if not chunk:
            break
        yield Passage(
            doc_id=f"{page_id}:{start}",
            title=str(title),
            url=str(url),
            text=" ".join(chunk),
        )
        if start + window_size >= len(token_list):
            break


def pyserini_record(passage: Passage) -> dict[str, str]:
    # title, URL and body stay in one field, _parse_hit expects it
    return {
        "id": passage.doc_id,
        "contents": f"{passage.title}\n{passage.url}\n{passage.text}",
    }


def passage_record(passage: Passage) -> dict[str, Any]:
    return {
        "doc_id": passage.doc_id,
        "title": passage.title,
        "url": passage.url,
        "text": passage.text,
        "score": passage.score,
    }


def _parse_hit(searcher: Any, hit: Any) -> Passage:
    raw = searcher.doc(hit.docid).raw()
    payload = json.loads(raw)
    contents = str(payload.get("contents", ""))
    # split twice only, later newlines belong to passage text
    parts = contents.split("\n", 2)
    title = parts[0] if parts else ""
    url = parts[1] if len(parts) > 1 else ""
    text = parts[2] if len(parts) > 2 else contents
    return Passage(str(hit.docid), title, url, text, float(hit.score))


def bm25_candidates(
    searcher: Any,
    query: str,
    *,
    retrieve: int = BM25_RETRIEVE,
    working_depth: int = BM25_WORKING_DEPTH,
) -> list[Passage]:
    # set paper BM25 values every time, Pyserini defaults differ
    searcher.set_bm25(BM25_K1, BM25_B)
    hits = searcher.search(str(query), int(retrieve))
    return [_parse_hit(searcher, hit) for hit in hits[:working_depth]]


def e5_rerank(
    query: str,
    candidates: Sequence[Passage],
    encoder: Any,
    *,
    batch_size: int = 64,
) -> list[Passage]:
    if not candidates:
        return []

    # query and passage use different E5 prefixes
    query_vector = encoder.encode_queries([query], batch_size=1)[0]
    passage_texts = [
        f"{candidate.title}\n{candidate.text}"
        if candidate.title
        else candidate.text
        for candidate in candidates[:BM25_WORKING_DEPTH]
    ]
    passage_vectors = encoder.encode_passages(
        passage_texts,
        batch_size=batch_size,
    )
    scores = passage_vectors @ query_vector
    # dense scores replace BM25 order, metadata follows source index
    order = scores.argsort(descending=True).tolist()[:E5_SCORE_DEPTH]
    # only top ten continue to BGE
    reranked = [
        Passage(
            doc_id=candidates[index].doc_id,
            title=candidates[index].title,
            url=candidates[index].url,
            text=candidates[index].text,
            score=float(scores[index].item()),
        )
        for index in order
    ]
    return reranked[:E5_KEEP]


def bge_top1(
    query: str,
    candidates: Sequence[Passage],
    reranker: Any,
    *,
    batch_size: int = 64,
) -> Passage | None:
    if not candidates:
        return None
    selected = list(candidates[:E5_KEEP])
    # URL is metadata, BGE sees title + passage text
    pairs = [
        (
            str(query),
            f"{candidate.title}\n{candidate.text}"
            if candidate.title
            else candidate.text,
        )
        for candidate in selected
    ]
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    # FV takes one passage, relative BGE score is enough
    best = max(range(len(scores)), key=lambda index: float(scores[index]))
    candidate = selected[best]
    return Passage(
        doc_id=candidate.doc_id,
        title=candidate.title,
        url=candidate.url,
        text=candidate.text,
        score=float(scores[best]),
    )


def retrieve_top1(
    context: Sequence[str],
    claim: str,
    *,
    searcher: Any,
    e5_encoder: Any,
    bge_reranker: Any,
    dense_batch_size: int = 64,
    rerank_batch_size: int = 64,
) -> Passage | None:
    # same query string through BM25, E5 and BGE
    query = build_ir_query(context, claim)
    sparse = bm25_candidates(searcher, query)
    dense = e5_rerank(
        query,
        sparse,
        e5_encoder,
        batch_size=dense_batch_size,
    )
    return bge_top1(
        query,
        dense,
        bge_reranker,
        batch_size=rerank_batch_size,
    )
