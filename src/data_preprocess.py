"""DialFact loading and R1-R5 claim preprocessing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence


# R1-R4 are cumulative, R5 starts again from R0
# -----------------------------------------------
# DialFact data helpers
# -------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a DialFact JSONL file."""
    # stop on bad JSON, skipping a row shifts all counts
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc


def normalise_turns(value: Sequence[str] | str | None) -> list[str]:
    """Return dialogue turns as a clean list while preserving their order."""
    if value is None:
        return []
    if isinstance(value, str):
        # some exports flatten context with [EOT]
        turns = value.split("[EOT]") if "[EOT]" in value else [value]
    else:
        turns = value
    return [str(turn).strip() for turn in turns if str(turn).strip()]


@dataclass(frozen=True)
class ClaimVariants:
    """Cumulative claim surfaces."""

    context_r0: tuple[str, ...]
    context_r1: tuple[str, ...]
    context_r2: tuple[str, ...]
    context_r3: tuple[str, ...]
    r0: str
    r1: str
    r2: str
    r3: str
    r4: str
    # R5 is separate, never apply it after R4
    r5: str | None = None


# --------------------------------------------------------------------------
# R1: conservative apostrophe restoration and contraction expansion
# ---------------------------------------------------------------------------


_URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]{1,4}\.){2,}")
_HONORIFIC_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.", re.IGNORECASE)

# protect these before the apostrophe/period regex
# URLs and ``Dr.`` are painful to reconstruct later

# high-confidence joined forms only
# leave ambiguous ordinary words such as ``its`` alone
_APOSTROPHE_FORMS = {
    "im": "I'm",
    "ive": "I've",
    "youre": "you're",
    "youve": "you've",
    "youll": "you'll",
    "youd": "you'd",
    "weve": "we've",
    "wed": "we'd",
    "theyre": "they're",
    "theyve": "they've",
    "theyll": "they'll",
    "theyd": "they'd",
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "isnt": "isn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "cant": "can't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "arent": "aren't",
    "wont": "won't",
}

_CONTRACTION_EXPANSIONS = {
    "i'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "let's": "let us",
    "i've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "I will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "i'd": "I would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "aren't": "are not",
    "won't": "will not",
}

# only treat ``ill`` as ``I'll`` before a clear verb
_I_WILL_FOLLOW = {
    "be", "go", "do", "have", "get", "see", "try", "call", "send",
    "take", "make", "give", "come", "show", "tell", "check", "meet",
}


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _protect_fragments(text: str) -> tuple[str, dict[str, str]]:
    # placeholders must stay clear of punctuation rules below
    protected = str(text or "")
    placeholders: dict[str, str] = {}

    for pattern in (_URL_RE, _DOTTED_ACRONYM_RE, _HONORIFIC_RE):
        def replace(match: re.Match[str], *, _placeholders: dict[str, str] = placeholders) -> str:
            key = f"__BICON_PROTECTED_{len(_placeholders)}__"
            _placeholders[key] = match.group(0)
            return key

        protected = pattern.sub(replace, protected)
    return protected, placeholders


def _restore_fragments(text: str, placeholders: Mapping[str, str]) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def restore_missing_apostrophes(text: str) -> str:
    """Apply conservative regex-gated apostrophe restoration."""
    # order matters here
    # joined forms -> narrow ``ill + verb`` case -> split-token artefacts
    protected, placeholders = _protect_fragments(text)

    for plain, contracted in _APOSTROPHE_FORMS.items():
        pattern = re.compile(rf"(?<!\w){re.escape(plain)}(?!\w)", re.IGNORECASE)
        protected = pattern.sub(
            lambda match, value=contracted: _match_case(match.group(0), value),
            protected,
        )

    verbs = "|".join(sorted(_I_WILL_FOLLOW))
    protected = re.sub(
        rf"(?<!\w)ill(?=\s+(?:{verbs})\b)",
        lambda match: _match_case(match.group(0), "I'll"),
        protected,
        flags=re.IGNORECASE,
    )

    # tokenisation leftovers such as ``do n t``
    split_forms = {
        r"\bi\s+m\b": "I'm",
        r"\bi\s+ve\b": "I've",
        r"\byou\s+re\b": "you're",
        r"\bwe\s+re\b": "we're",
        r"\bthey\s+re\b": "they're",
        r"\bdo\s+n['’]?t\b": "don't",
        r"\bcan\s+n['’]?t\b": "can't",
    }
    for raw_pattern, contracted in split_forms.items():
        protected = re.sub(
            raw_pattern,
            lambda match, value=contracted: _match_case(match.group(0), value),
            protected,
            flags=re.IGNORECASE,
        )

    return _restore_fragments(protected, placeholders)


def expand_contractions(text: str) -> str:
    """Expand contractions while preserving protected fragments."""
    protected, placeholders = _protect_fragments(str(text or "").replace("’", "'"))

    # handle the obvious auxiliary case before the general map
    protected = re.sub(
        r"(?i)(?<!\w)([A-Za-z]+)'s\s+been\b",
        lambda match: f"{match.group(1)} has been",
        protected,
    )
    protected = re.sub(
        r"(?i)(?<!\w)([A-Za-z]+)'d\s+been\b",
        lambda match: f"{match.group(1)} had been",
        protected,
    )

    keys = sorted(_CONTRACTION_EXPANSIONS, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(key) for key in keys) + r")(?!\w)",
        re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        return _match_case(source, _CONTRACTION_EXPANSIONS[source.lower()])

    return _restore_fragments(pattern.sub(replace, protected), placeholders)


def apply_r1(text: str) -> str:
    # restore first or ``dont`` never reaches the expansion table
    return expand_contractions(restore_missing_apostrophes(text))


# --------------------------------------------
# R2: turn-preserving punctuation restoration
# ---------------------------------------------


class PunctuationPredictor(Protocol):
    # model can rewrite words, R2 only borrows punctuation
    def __call__(self, text: str) -> str: ...


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_TERMINAL_RE = re.compile(r"[.!?][\"'”’)]*\s*$")


def _word_matches(text: str) -> list[re.Match[str]]:
    return list(_WORD_RE.finditer(text))


def restore_turn_punctuation(text: str, predictor: PunctuationPredictor) -> str:
    """Overlay only predicted commas and sentence-final marks on one turn.

    The lexical token sequence must agree between the original and model output;
    otherwise the original turn is retained.  A non-question turn without final
    punctuation receives a period.
    """
    original = str(text or "")
    predicted = str(predictor(original))
    original_words = _word_matches(original)
    predicted_words = _word_matches(predicted)

    original_tokens = [match.group(0).lower().replace("’", "'") for match in original_words]
    predicted_tokens = [match.group(0).lower().replace("’", "'") for match in predicted_words]

    insertions: list[tuple[int, str]] = []

    # token mismatch means lexical content changed
    # fall back to the original turn rather than guessing alignment
    if original_words and original_tokens == predicted_tokens:
        for index, predicted_word in enumerate(predicted_words):
            predicted_right = (
                predicted_words[index + 1].start()
                if index + 1 < len(predicted_words)
                else len(predicted)
            )
            predicted_gap = predicted[predicted_word.end():predicted_right]
            is_final = index == len(predicted_words) - 1
            allowed = [
                mark for mark in predicted_gap
                if mark == "," or (is_final and mark in ".!?")
            ]
            if not allowed:
                continue

            original_word = original_words[index]
            original_right = (
                original_words[index + 1].start()
                if index + 1 < len(original_words)
                else len(original)
            )
            existing_gap = original[original_word.end():original_right]
            for mark in allowed:
                if mark not in existing_gap:
                    insertions.append((original_word.end(), mark))

    restored = original
    # replace from right to left, offsets come from the original string
    for position, mark in sorted(insertions, reverse=True):
        restored = restored[:position] + mark + restored[position:]

    # no final period when either side marks a question
    predicted_question = bool(re.search(r"\?[\"'”’)]*\s*$", predicted))
    original_question = bool(re.search(r"\?[\"'”’)]*\s*$", original))
    if not _TERMINAL_RE.search(restored) and not (predicted_question or original_question):
        restored = restored.rstrip() + "."
    return restored


def apply_r2(
    context: Sequence[str],
    response: str,
    predictor: PunctuationPredictor,
) -> tuple[list[str], str]:
    """Apply the punctuation model independently to every dialogue turn."""
    # keep turns separate until R2 is done
    # joining early creates punctuation at speaker boundaries
    return (
        [restore_turn_punctuation(turn, predictor) for turn in normalise_turns(context)],
        restore_turn_punctuation(response, predictor),
    )


# ---------------------------------------------------------------------------
# R3: BERT masked-LM true-casing
# ---------------------------------------------------------------------------


class Truecaser(Protocol):
    def truecase(self, text: str) -> str: ...


def apply_r3(
    context: Sequence[str],
    response: str,
    truecaser: Truecaser,
) -> tuple[list[str], str]:
    # case only, no token/spacing/turn changes
    return (
        [truecaser.truecase(turn) for turn in normalise_turns(context)],
        truecaser.truecase(response),
    )


# ---------------------------------------------------------------------------
# R4: scoped pronominal coreference rewriting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PronounMention:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class AntecedentCandidate:
    text: str
    score: float = 0.0


class CandidateProvider(Protocol):
    def __call__(
        self,
        context: Sequence[str],
        response: str,
        mention: PronounMention,
    ) -> Sequence[Any]: ...


class AntecedentSelector(Protocol):
    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str: ...


_PERSONAL_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "you", "your", "yours", "yourself", "yourselves",
}
# keep deictics separate from personal pronouns
_DEICTIC_FORMS = {"this", "that", "these", "those"}


def detect_pronominal_anchors(
    response: str,
    nlp: Callable[[str], Iterable[Any]],
) -> list[PronounMention]:
    """Detect in-claim pronouns, excluding deictics and dependency-marked expletive *it*."""
    mentions: list[PronounMention] = []
    for token in nlp(str(response or "")):
        surface = str(getattr(token, "text", token))
        lower = surface.lower()
        if lower in _DEICTIC_FORMS or lower not in _PERSONAL_PRONOUNS:
            continue
        dependency = str(getattr(token, "dep_", "")).lower()
        # surface rules cannot reliably split referential/expletive ``it``
        if lower == "it" and dependency in {"expl", "expletive"}:
            continue
        start = int(getattr(token, "idx", -1))
        if start >= 0:
            mentions.append(PronounMention(surface, start, start + len(surface)))
    return mentions


def build_selector_messages(
    context: Sequence[str],
    masked_response: str,
    candidates: Sequence[AntecedentCandidate],
) -> list[dict[str, str]]:
    """Build the small candidate-selection task used for the R4 selector."""
    context_block = "\n".join(normalise_turns(context))
    # candidate ids are 1-based in both places
    # prompt and parser have to agree here
    candidate_block = "\n".join(
        f"{index}. {candidate.text}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Select the antecedent noun phrase for the masked pronoun using only "
                "the dialogue context. Return one candidate number and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Response:\n{masked_response}\n\n"
                f"Candidates:\n{candidate_block}"
            ),
        },
    ]


def parse_selector_index(text: str, number_of_candidates: int) -> int | None:
    """Return a zero-based candidate index only when the generated choice is valid."""
    # reject free-form text and numbers outside the candidate list
    match = re.search(r"\b(\d+)\b", str(text or ""))
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < number_of_candidates else None


def apply_r4(
    context_r3: Sequence[str],
    response_r3: str,
    mentions: Sequence[PronounMention],
    candidate_provider: CandidateProvider,
    selector: AntecedentSelector,
    *,
    max_candidates: int = 10,
) -> str:
    """Replace each in-scope pronoun with the selected context NP only."""
    if max_candidates != 10:
        raise ValueError("R4 uses at most 10 candidates.")

    rewritten = str(response_r3 or "")
    # right to left so earlier offsets survive
    for mention in sorted(mentions, key=lambda item: item.start, reverse=True):
        proposed = list(candidate_provider(context_r3, response_r3, mention))

        # keep provider order, max ten as in the paper
        unique: list[AntecedentCandidate] = []
        seen: set[str] = set()
        for candidate in proposed:
            if isinstance(candidate, Mapping):
                text = str(candidate.get("text") or "").strip()
                score = float(candidate.get("score") or 0.0)
            elif isinstance(candidate, AntecedentCandidate):
                text = candidate.text.strip()
                score = float(candidate.score)
            else:
                text = str(candidate).strip()
                score = 0.0
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            unique.append(AntecedentCandidate(text=text, score=score))
            if len(unique) == max_candidates:
                break
        if not unique:
            continue

        # always build selector input from R3
        # after one replacement, ``rewritten`` offsets are no longer source offsets
        masked_response = (
            response_r3[:mention.start] + "[MASK]" + response_r3[mention.end:]
        )
        generated = selector(build_selector_messages(context_r3, masked_response, unique))
        selected_index = parse_selector_index(generated, len(unique))
        if selected_index is None:
            # bad selector output means leave this pronoun alone
            continue

        antecedent = unique[selected_index].text
        rewritten = rewritten[:mention.start] + antecedent + rewritten[mention.end:]

    return rewritten


# ---------------------------------------------------------------------------
# R5: decoder-based one-shot reformulation
# ---------------------------------------------------------------------------


# keep this prompt fixed
# one wording change can affect several normalisation steps at once
R5_SYSTEM_PROMPT = "Follow the instructions exactly. Do not add or change facts."
R5_USER_TEMPLATE = """You are an expert editor who rewrites informal, chatty utterances into well-formed declarative English without changing their meaning.

You will receive: (i) Context, a list of previous dialogue turns; and (ii) Response, the claim text to be normalised.
Task: Rewrite Response into New_Response by applying only the following operations.
(1) Add missing sentence-ending punctuation and fix spacing around punctuation.
(2) Fix capitalisation at sentence starts and for proper nouns.
(3) Insert missing apostrophes (e.g., dont→don’t, cant→can’t, im→I’m).
(4) Expand all contractions to full forms (e.g., isn’t→is not, aren’t→are not, won’t→will not, wouldn’t→would not, I’m→I am, it’s→it is, they’re→they are, don’t→do not, can’t→cannot).
(5) If a pronoun in Response (this/that/it/he/she/they/these/those) has a unique, clear antecedent in Context, replace it with that antecedent phrase; if ambiguous, leave it unchanged.
(6) Do not add, remove, or correct any facts, numbers, names, or dates; preserve the claim’s semantics exactly.
(7) Output only the rewritten text as one or more sentences, with no explanations, lists, or markdown.

The input is formatted as:
Context (earliest→latest):
{context_lines}
Response: {response_text}
Output:
(New_Response only; no explanations)."""


def build_r5_messages(context: Sequence[str], response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": R5_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": R5_USER_TEMPLATE.format(
                context_lines="\n".join(normalise_turns(context)),
                response_text=str(response or ""),
            ),
        },
    ]


def clean_generated_rewrite(text: str) -> str:
    # strip wrappers only, never edit generated facts or wording
    value = str(text or "").strip()
    value = re.sub(r"^```(?:text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(
        r"^(?:New_Response|Output|Assistant)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


# ---------------------------------------------------------------------------
# Cumulative pipeline
# ---------------------------------------------------------------------------


def build_claim_variants(
    context: Sequence[str],
    response: str,
    *,
    punctuation_predictor: PunctuationPredictor,
    truecaser: Truecaser,
    pronoun_detector: Callable[[str], Sequence[PronounMention]],
    candidate_provider: CandidateProvider,
    antecedent_selector: AntecedentSelector,
    decoder_rewriter: Callable[[Sequence[Mapping[str, str]]], str] | None = None,
) -> ClaimVariants:
    """Build cumulative R1-R4 and the alternative R5 surface."""
    context_r0 = normalise_turns(context)
    r0 = str(response or "")

    # leave the stages explicit for now
    # easier to spot an accidental R1-R4 reorder
    context_r1 = [apply_r1(turn) for turn in context_r0]
    r1 = apply_r1(r0)

    context_r2, r2 = apply_r2(context_r1, r1, punctuation_predictor)
    context_r3, r3 = apply_r3(context_r2, r2, truecaser)

    r4 = apply_r4(
        context_r3,
        r3,
        list(pronoun_detector(r3)),
        candidate_provider,
        antecedent_selector,
        max_candidates=10,
    )

    r5 = None
    if decoder_rewriter is not None:
        # R5 starts from R0, never from r3/r4
        generated = decoder_rewriter(build_r5_messages(context_r0, r0))
        r5 = clean_generated_rewrite(generated) or r0

    return ClaimVariants(
        context_r0=tuple(context_r0),
        context_r1=tuple(context_r1),
        context_r2=tuple(context_r2),
        context_r3=tuple(context_r3),
        r0=r0,
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        r5=r5,
    )
