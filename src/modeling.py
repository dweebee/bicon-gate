"""Model identifiers and inference wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from data_preprocess import AntecedentCandidate, PronounMention, normalise_turns


# keep checkpoint names together, easy to mix tokenizer/length settings
# same interface does not mean the same setup
PUNCTUATION_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
TRUECASING_MODEL = "bert-base-cased"
COREFERENCE_MODEL = "sapienzanlp/maverick-mes-ontonotes"
ANTECEDENT_SELECTOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DECODER_REWRITE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
E5_MODEL = "intfloat/multilingual-e5-large"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-large"


@dataclass(frozen=True)
class NLIOutput:
    entailment: float
    neutral: float
    contradiction: float
    label: str


@dataclass(frozen=True)
class TruecasingSettings:
    # no default margin, it changes the R3 output
    margin_threshold: float
    max_length: int = 512
    device: str | None = None

    def __post_init__(self) -> None:
        if self.margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative.")


def _resolve_device(torch: Any, requested: str | None) -> Any:
    # cluster jobs pass device explicitly, local runs use what is available
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalise_nli_label(label: Any) -> str:
    # HF label names are inconsistent, sometimes only LABEL_0 etc
    value = str(label or "").upper().replace("LABEL_", "")
    if "ENTAIL" in value or value in {"TRUE", "SUPPORTS", "SUPPORTED"}:
        return "ENTAILMENT"
    if "CONTR" in value or value in {"FALSE", "REFUTES", "REFUTED"}:
        return "CONTRADICTION"
    if "NEUT" in value or value in {"NEI", "UNKNOWN", "NOT ENOUGH INFO"}:
        return "NEUTRAL"
    return value


class PunctuationRestorer:
    # Maverick handles its own tokenizer/model loading
    # keep it outside the generic wrappers below
    def __init__(self, model_name: str = PUNCTUATION_MODEL):
        try:
            from deepmultilingualpunctuation import PunctuationModel
        except ImportError as exc:
            raise ImportError("deepmultilingualpunctuation is required.") from exc
        self.model = PunctuationModel(model_name)

    def __call__(self, text: str) -> str:
        return str(self.model.restore_punctuation(str(text)))


class BertMarginTruecaser:
    TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

    def __init__(
        self,
        settings: TruecasingSettings,
        model_name: str = TRUECASING_MODEL,
    ):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("torch and transformers are required.") from exc

        self.torch = torch
        self.settings = settings
        self.device = _resolve_device(torch, settings.device)
        # need fast-tokenizer offsets to score the current word only
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        if self.tokenizer.mask_token_id is None:
            raise ValueError("The tokenizer must provide a mask token.")

    def _variant_log_probability(self, text: str, start: int, end: int) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.settings.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if left != right and left >= start and right <= end
        ]
        if not positions:
            return float("-inf")

        # one word can split into several pieces
        # mask all pieces and use mean log-probability
        targets = input_ids[0, positions].clone()
        masked = input_ids.clone()
        masked[0, positions] = self.tokenizer.mask_token_id
        with self.torch.inference_mode():
            logits = self.model(masked, attention_mask=attention_mask).logits[0, positions]
        log_probs = self.torch.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        return float(chosen.mean().item())

    def _score_variant(self, text: str, start: int, end: int, token: str) -> float:
        candidate = text[:start] + token + text[end:]
        return self._variant_log_probability(candidate, start, start + len(token))

    def truecase(self, text: str) -> str:
        original = str(text or "")
        # score every token against the untouched sentence
        # editing as we go shifts later offsets and context
        characters = list(original)
        margin_threshold = self.settings.margin_threshold

        for match in self.TOKEN_RE.finditer(original):
            token = match.group(0)
            upper = token[0].upper() + token[1:]
            lower = token[0].lower() + token[1:]
            if upper == lower:
                continue

            upper_score = self._score_variant(original, match.start(), match.end(), upper)
            lower_score = self._score_variant(original, match.start(), match.end(), lower)
            margin = upper_score - lower_score
            # initial character only
            # do not let the checkpoint rewrite the rest of the token
            if margin > margin_threshold:
                characters[match.start()] = upper[0]
            elif margin < -margin_threshold:
                characters[match.start()] = lower[0]

        return "".join(characters)


class MaverickCandidateProvider:
    # full coreference output is broader than R4
    # filter mentions that should not return as noun-phrase antecedents
    _PRONOUNS = {
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "you", "your", "yours", "yourself", "yourselves",
        "this", "that", "these", "those",
    }

    def __init__(
        self,
        model_name: str = COREFERENCE_MODEL,
        *,
        device: str = "cuda:0",
    ):
        try:
            from maverick import Maverick
        except ImportError as exc:
            raise ImportError("maverick-coref is required.") from exc
        self.model = Maverick(hf_name_or_path=model_name, device=device)

    @staticmethod
    def _find_cluster(
        clusters: Sequence[Sequence[Sequence[int]]],
        mention_start: int,
        mention_end: int,
    ) -> tuple[int | None, int | None]:
        for cluster_index, cluster in enumerate(clusters):
            for mention_index, span in enumerate(cluster):
                if len(span) < 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if not (end <= mention_start or mention_end <= start):
                    return cluster_index, mention_index
        return None, None

    def __call__(
        self,
        context: Sequence[str],
        response: str,
        mention: PronounMention,
    ) -> Sequence[AntecedentCandidate]:
        context_text = " ".join(normalise_turns(context)).strip()
        response_text = str(response or "")
        # remember the one joining space when mapping response offsets
        # back into Maverick full-document spans
        response_offset = len(context_text) + 1 if context_text else 0
        document = f"{context_text} {response_text}".strip() if context_text else response_text
        output = self.model.predict(document)

        clusters = output.get("clusters_char_offsets") or []
        texts = output.get("clusters_text_mentions") or output.get("clusters_token_text") or []
        mention_start = response_offset + mention.start
        mention_end = response_offset + mention.end
        cluster_index, mention_index = self._find_cluster(clusters, mention_start, mention_end)
        if cluster_index is None:
            return []

        candidates: list[AntecedentCandidate] = []
        seen: set[str] = set()
        cluster_spans = clusters[cluster_index]
        cluster_texts = texts[cluster_index] if cluster_index < len(texts) else []

        for index, span in enumerate(cluster_spans):
            if index == mention_index or len(span) < 2:
                continue
            start, end = int(span[0]), int(span[1])
            # antecedents must come from context
            # response mentions cannot replace the response pronoun
            if end > len(context_text):
                continue
            text = (
                str(cluster_texts[index]).strip()
                if index < len(cluster_texts)
                else document[start:end].strip()
            )
            key = text.lower()
            if not text or key in self._PRONOUNS or key in seen:
                continue
            seen.add(key)
            candidates.append(AntecedentCandidate(text=text))

        return candidates


class GreedyChatModel:
    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        max_input_tokens: int = 4096,
        device_map: str | Mapping[str, Any] | None = "auto",
        torch_dtype: str | Any = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("torch and transformers are required.") from exc

        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # keep left padding for decoder-only checkpoints
        # generated tails stay aligned in batches
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Llama/Qwen often have no separate pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        ).eval()

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_device = next(self.model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                # no sampling here, R4/R5 must stay deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # decode the continuation only
        # full decode leaks the chat template into the parser
        new_tokens = generated[0, input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LlamaAntecedentSelector(GreedyChatModel):
    # expected output is one candidate id, keep generation short
    def __init__(
        self,
        model_name: str = ANTECEDENT_SELECTOR_MODEL,
        *,
        max_new_tokens: int = 64,
        max_input_tokens: int = 4096,
        device_map: str | Mapping[str, Any] | None = "auto",
        torch_dtype: str | Any = "bfloat16",
    ):
        super().__init__(
            model_name,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )


class QwenRewriteGenerator(GreedyChatModel):
    # R5 can need several sentences, give it more room
    def __init__(
        self,
        model_name: str = DECODER_REWRITE_MODEL,
        *,
        max_new_tokens: int = 160,
        max_input_tokens: int = 4096,
        device_map: str | Mapping[str, Any] | None = "auto",
        torch_dtype: str | Any = "auto",
    ):
        super().__init__(
            model_name,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )


class ThreeWayNLI:
    def __init__(
        self,
        model_name: str = NLI_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        fp16: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("torch and transformers are required.") from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # half precision on GPU only for this path
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.eval()

        raw = getattr(self.model.config, "id2label", {}) or {}
        # trust config labels only when all three are present
        id2label = {int(key): _normalise_nli_label(value) for key, value in raw.items()}
        if set(id2label.values()) != {"CONTRADICTION", "NEUTRAL", "ENTAILMENT"}:
            id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        self.id2label = id2label
        self.label2id = {label: index for index, label in id2label.items()}

    def logits(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 16,
    ) -> Any:
        rows = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            premises = [str(premise) for premise, _ in chunk]
            hypotheses = [str(hypothesis) for _, hypothesis in chunk]
            # do not swap premise and hypothesis for the gate
            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # DeBERTa ignores token-type ids, some tokenizers still emit them
            encoded.pop("token_type_ids", None)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                rows.append(self.model(**encoded).logits.float().cpu())
        if not rows:
            return self.torch.empty((0, 3), dtype=self.torch.float32)
        return self.torch.cat(rows, dim=0)

    def probabilities(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        temperature: float = 1.0,
        batch_size: int = 16,
    ) -> list[NLIOutput]:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        logits = self.logits(pairs, batch_size=batch_size)
        # temperature goes on logits, not probabilities
        probabilities = self.torch.softmax(logits / float(temperature), dim=-1)
        outputs: list[NLIOutput] = []
        for row in probabilities:
            predicted = int(row.argmax().item())
            outputs.append(
                NLIOutput(
                    entailment=float(row[self.label2id["ENTAILMENT"]]),
                    neutral=float(row[self.label2id["NEUTRAL"]]),
                    contradiction=float(row[self.label2id["CONTRADICTION"]]),
                    label=self.id2label[predicted],
                )
            )
        return outputs

    def directional(
        self,
        premise: str,
        hypothesis: str,
        *,
        temperature: float = 1.0,
    ) -> NLIOutput:
        return self.probabilities(
            [(premise, hypothesis)],
            temperature=temperature,
            batch_size=1,
        )[0]

    def predict_labels(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 16,
        temperature: float = 1.0,
    ) -> list[str]:
        return [
            output.label
            for output in self.probabilities(
                pairs,
                batch_size=batch_size,
                temperature=temperature,
            )
        ]

    def label_index(self, label: Any) -> int:
        normalised = _normalise_nli_label(label)
        if normalised not in self.label2id:
            raise ValueError(f"Unsupported NLI label: {label!r}")
        return self.label2id[normalised]


class GateNLI(ThreeWayNLI):
    # gate pairs are shorter, 384 is enough here
    def __init__(self, model_name: str = NLI_MODEL, **kwargs: Any):
        kwargs.setdefault("max_length", 384)
        super().__init__(model_name, **kwargs)


class FactVerifier(ThreeWayNLI):
    # FV has evidence plus context, keep 512
    def __init__(self, model_name: str = NLI_MODEL, **kwargs: Any):
        kwargs.setdefault("max_length", 512)
        super().__init__(model_name, **kwargs)


class E5Encoder:
    def __init__(
        self,
        model_name: str = E5_MODEL,
        *,
        device: str | None = None,
        max_length: int = 256,
        fp16: bool = True,
    ):
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("torch and transformers are required.") from exc

        self.torch = torch
        self.functional = functional
        self.device = _resolve_device(torch, device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    def encode(
        self,
        texts: Sequence[str],
        *,
        prefix: str = "",
        batch_size: int = 32,
    ) -> Any:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + str(text) for text in texts[start:start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            # mean-pool real tokens only, no CLS pooling for E5
            mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(self.functional.normalize(pooled, p=2, dim=1).float().cpu())
        if not embeddings:
            return self.torch.empty((0, 0), dtype=self.torch.float32)
        return self.torch.cat(embeddings, dim=0)

    def encode_queries(self, texts: Sequence[str], *, batch_size: int = 32) -> Any:
        # E5 prefixes are part of the model input
        return self.encode(texts, prefix="query: ", batch_size=batch_size)

    def encode_passages(self, texts: Sequence[str], *, batch_size: int = 32) -> Any:
        return self.encode(texts, prefix="passage: ", batch_size=batch_size)

    def cosine(self, left: str, right: str) -> float:
        # embeddings are L2-normalised, dot product is cosine
        vectors = self.encode_queries([left, right], batch_size=2)
        return float((vectors[0] @ vectors[1]).item())


class BGEReranker:
    def __init__(
        self,
        model_name: str = BGE_RERANKER_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
    ):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("sentence-transformers is required.") from exc
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # downstream only needs relative scores, no sigmoid here
        self.model = CrossEncoder(model_name, device=resolved_device, max_length=max_length)

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> Any:
        return self.model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
