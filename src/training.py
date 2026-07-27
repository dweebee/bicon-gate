"""NLI calibration and gate-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation import (
    FV_CONTEXT_TURNS,
    aggregate_ir_metrics,
    build_fv_pair,
    classification_scores,
    normalise_gold_label,
)
# calibration and validation selection only
# backbone weights stay frozen
from retrieval import (
    BM25_WORKING_DEPTH,
    bm25_candidates,
    build_ir_query,
    passage_record,
)


# inclusive 0.20...1.00 grid, keep rounding or float keys get messy
THRESHOLDS = tuple(round(0.20 + 0.05 * index, 2) for index in range(17))


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    count: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


def expected_calibration_error(
    probabilities: Any,
    labels: Any,
    *,
    bins: int = 15,
) -> float:
    import torch

    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    # same open-left/closed-right bins for every calibration report
    boundaries = torch.linspace(0, 1, steps=bins + 1)
    value = torch.tensor(0.0)

    for index in range(bins):
        mask = (
            (confidence > boundaries[index])
            & (confidence <= boundaries[index + 1])
        )
        if mask.any():
            value += (
                (correctness[mask].mean() - confidence[mask].mean()).abs()
                * mask.float().mean()
            )
    return float(value.item())


def fit_temperature(
    logits: Any,
    labels: Any,
    *,
    max_iter: int = 100,
) -> CalibrationResult:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("torch is required.") from exc

    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Expected logits with shape [N, 3].")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must have shape [N].")
    if logits.shape[0] == 0:
        raise ValueError("Calibration data is empty.")

    with torch.no_grad():
        probability_before = torch.softmax(logits, dim=-1)
        nll_before = float(functional.cross_entropy(logits, labels).item())
        ece_before = expected_calibration_error(probability_before, labels)

    # optimise log(T), temperature must stay positive
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimiser = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        # LBFGS calls this more than once, no logging or buffer changes here
        optimiser.zero_grad()
        temperature = log_temperature.exp()
        loss = functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(log_temperature.detach().exp().item())

    # before/after use the same unshuffled validation logits
    with torch.no_grad():
        calibrated = logits / temperature
        probability_after = torch.softmax(calibrated, dim=-1)
        nll_after = float(functional.cross_entropy(calibrated, labels).item())
        ece_after = expected_calibration_error(probability_after, labels)

    return CalibrationResult(
        temperature=temperature,
        count=int(labels.numel()),
        nll_before=nll_before,
        nll_after=nll_after,
        ece_before=ece_before,
        ece_after=ece_after,
    )


def calibrate_nli(
    nli: Any,
    pairs: Sequence[tuple[str, str]],
    labels: Sequence[str],
    *,
    batch_size: int = 16,
    max_iter: int = 100,
) -> CalibrationResult:
    if len(pairs) != len(labels):
        raise ValueError("Calibration pairs and labels must have equal length.")
    logits = nli.logits(pairs, batch_size=batch_size)
    # resolve ids through the wrapper, never assume 0/1/2 order
    label_ids = [nli.label_index(normalise_gold_label(label)) for label in labels]
    return fit_temperature(logits, label_ids, max_iter=max_iter)


def select_gated_claims(
    gate_scores: Sequence[float],
    original_claims: Sequence[str],
    rewritten_claims: Sequence[str],
    threshold: float,
) -> list[str]:
    if not (
        len(gate_scores)
        == len(original_claims)
        == len(rewritten_claims)
    ):
        raise ValueError("Gate arrays must have equal length.")
    # >= follows the gate rule, boundary selects Rk
    return [
        str(rewritten if score >= threshold else original)
        for score, original, rewritten in zip(
            gate_scores,
            original_claims,
            rewritten_claims,
        )
    ]


def tune_fv_threshold(
    samples: Sequence[Mapping[str, Any]],
    gate_scores: Sequence[float],
    original_claims: Sequence[str],
    rewritten_claims: Sequence[str],
    verifier: Any,
    *,
    thresholds: Sequence[float] = THRESHOLDS,
    context_turns: int = FV_CONTEXT_TURNS,
    batch_size: int = 16,
) -> list[dict[str, float]]:
    size = len(samples)
    if not (
        size
        == len(gate_scores)
        == len(original_claims)
        == len(rewritten_claims)
    ):
        raise ValueError("FV threshold inputs must have equal length.")
    if size == 0:
        raise ValueError("FV threshold data is empty.")

    gold = [normalise_gold_label(sample.get("response_label")) for sample in samples]
    curve: list[dict[str, float]] = []

    for threshold in thresholds:
        # only selected claim changes across the curve
        claims = select_gated_claims(
            gate_scores,
            original_claims,
            rewritten_claims,
            threshold,
        )
        pairs = [
            build_fv_pair(sample, claim, context_turns=context_turns)
            for sample, claim in zip(samples, claims)
        ]
        predicted = verifier.predict_labels(pairs, batch_size=batch_size)
        scores = classification_scores(gold, predicted)
        # coverage is share routed to Rk, not verifier confidence
        coverage = sum(score >= threshold for score in gate_scores) / size
        curve.append(
            {
                "threshold": float(threshold),
                "accuracy": scores.accuracy,
                "macro_f1": scores.macro_f1,
                "macro_recall": scores.macro_recall,
                "coverage": coverage,
            }
        )
    return curve


def tune_ir_threshold(
    samples: Sequence[Mapping[str, Any]],
    gate_scores: Sequence[float],
    original_claims: Sequence[str],
    rewritten_claims: Sequence[str],
    searcher: Any,
    *,
    thresholds: Sequence[float] = THRESHOLDS,
    k: int = BM25_WORKING_DEPTH,
) -> list[dict[str, float]]:
    size = len(samples)
    if not (
        size
        == len(gate_scores)
        == len(original_claims)
        == len(rewritten_claims)
    ):
        raise ValueError("IR threshold inputs must have equal length.")
    if size == 0:
        raise ValueError("IR threshold data is empty.")

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        # routing changes the query, rerun BM25 for every tau
        claims = select_gated_claims(
            gate_scores,
            original_claims,
            rewritten_claims,
            threshold,
        )
        predictions = []
        for sample, claim in zip(samples, claims):
            query = build_ir_query(sample.get("context") or [], claim)
            ranked = bm25_candidates(
                searcher,
                query,
                working_depth=k,
            )
            predictions.append([passage_record(passage) for passage in ranked])

        # evaluator collapses passage hits to document ranks
        metrics = aggregate_ir_metrics(samples, predictions, k=k)
        metrics["threshold"] = float(threshold)
        metrics["coverage"] = sum(
            score >= threshold for score in gate_scores
        ) / size
        curve.append(metrics)
    return curve


def best_threshold(
    curve: Sequence[Mapping[str, float]],
    metric: str,
) -> float:
    if not curve:
        raise ValueError("Threshold curve is empty.")
    if any(metric not in point for point in curve):
        raise KeyError(metric)
    # curve is ascending, max() keeps lower tau on ties
    return float(max(curve, key=lambda point: float(point[metric]))["threshold"])
