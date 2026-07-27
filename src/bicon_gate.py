"""BiCon-Gate scoring and claim routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# paper values only
TEMPERATURE = 4.96
ALPHA = 0.4
BETA = 0.2
GAMMA = 0.4
TAU_IR = 0.50
TAU_FV = 0.70


@dataclass(frozen=True)
class DirectionalNLI:
    entailment: float
    contradiction: float


@dataclass(frozen=True)
class GateResult:
    entailment: float
    contradiction: float
    similarity: float
    score: float
    accepted: bool
    selected_claim: str


def join_context_claim(context: Sequence[str], claim: str) -> str:
    # keep the blank line, it is part of the NLI input
    context_block = "\n".join(
        str(turn).strip()
        for turn in context
        if str(turn).strip()
    )
    claim = str(claim or "").strip()
    if context_block and claim:
        return f"{context_block}\n\n{claim}"
    return context_block or claim


def build_bidirectional_pairs(
    context: Sequence[str],
    original_claim: str,
    rewritten_claim: str,
) -> tuple[tuple[str, str], tuple[str, str]]:
    context_original = join_context_claim(context, original_claim)

    # do not swap these: (C + R0) -> Rk and the exact reverse
    return (
        (context_original, str(rewritten_claim)),
        (str(rewritten_claim), context_original),
    )


def combine_gate_components(
    forward: DirectionalNLI | Any,
    backward: DirectionalNLI | Any,
    similarity: float,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> tuple[float, float, float]:
    if min(alpha, beta, gamma) < 0:
        raise ValueError("Gate weights must be non-negative.")
    if abs((alpha + beta + gamma) - 1.0) > 1e-8:
        raise ValueError("Gate weights must sum to one.")

    # intentional asymmetry here
    # both sides need entailment, one contradiction is enough
    entailment = min(float(forward.entailment), float(backward.entailment))
    contradiction = max(float(forward.contradiction), float(backward.contradiction))
    score = (
        alpha * entailment
        + beta * float(similarity)
        + gamma * (1.0 - contradiction)
    )
    return entailment, contradiction, score


def route_claim(
    original_claim: str,
    rewritten_claim: str,
    score: float,
    threshold: float,
) -> tuple[bool, str]:
    # >= is intentional, the boundary goes to Rk
    accepted = float(score) >= float(threshold)
    selected = rewritten_claim if accepted else original_claim
    return accepted, str(selected)


class BiConGate:
    def __init__(
        self,
        nli: Any,
        encoder: Any,
        *,
        temperature: float = TEMPERATURE,
    ):
        self.nli = nli
        self.encoder = encoder
        self.temperature = float(temperature)

    def score(
        self,
        context: Sequence[str],
        original_claim: str,
        rewritten_claim: str,
        *,
        threshold: float,
    ) -> GateResult:
        forward_pair, backward_pair = build_bidirectional_pairs(
            context,
            original_claim,
            rewritten_claim,
        )
        forward = self.nli.directional(
            *forward_pair,
            temperature=self.temperature,
        )
        backward = self.nli.directional(
            *backward_pair,
            temperature=self.temperature,
        )

        # context only goes into the NLI check
        # E5 compares claim surfaces or long context takes over the cosine
        similarity = self.encoder.cosine(original_claim, rewritten_claim)
        entailment, contradiction, score = combine_gate_components(
            forward,
            backward,
            similarity,
        )
        accepted, selected_claim = route_claim(
            original_claim,
            rewritten_claim,
            score,
            threshold,
        )
        return GateResult(
            entailment=entailment,
            contradiction=contradiction,
            similarity=similarity,
            score=score,
            accepted=accepted,
            selected_claim=selected_claim,
        )

    def score_many(
        self,
        contexts: Sequence[Sequence[str]],
        original_claims: Sequence[str],
        rewritten_claims: Sequence[str],
        *,
        threshold: float,
    ) -> list[GateResult]:
        # small helper for checks
        # full runs precompute once and reuse for the tau sweep
        if not (
            len(contexts)
            == len(original_claims)
            == len(rewritten_claims)
        ):
            raise ValueError("Gate inputs must have equal length.")
        return [
            self.score(context, original, rewritten, threshold=threshold)
            for context, original, rewritten in zip(
                contexts,
                original_claims,
                rewritten_claims,
            )
        ]
