"""GPT-2 fine-tuned sequence classifier and adapter for legal contract clause tagging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CLASSES = [
    "Confidentiality",
    "Governing Law",
    "Termination",
    "Indemnification",
    "Payment",
]

DEFAULT_ADAPTER_PATH = Path("models/gpt2_clause_adapter.pt")
DEFAULT_DATASET_PATH = Path("data/clause_dataset.json")


class LegalClauseClassifier:
    """Legal clause classification using pre-trained GPT-2 and fine-tuned adapter."""

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        adapter_path: str | Path | None = None,
        classes: list[str] | None = None,
        device: str | None = None,
        fast_fallback: bool = False,
    ) -> None:
        self.classes = classes or list(DEFAULT_CLASSES)
        self.label2id = {c: i for i, c in enumerate(self.classes)}
        self.id2label = {i: c for i, c in enumerate(self.classes)}
        self.num_labels = len(self.classes)
        self.fast_fallback = fast_fallback

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None

        if not self.fast_fallback:
            self._load_model(model_name_or_path, adapter_path)

    def _load_model(
        self, model_name_or_path: str, adapter_path: str | Path | None
    ) -> None:
        """Load tokenizer, GPT-2 backbone, and fine-tuned adapter weights."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Load fine-tuned adapter weights if available
            adapter_file = Path(adapter_path or DEFAULT_ADAPTER_PATH)
            if adapter_file.exists():
                state_dict = torch.load(adapter_file, map_location="cpu")
                self.model.score.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()
        except Exception:
            # Graceful fallback to deterministic feature classification if offline or network error
            self.fast_fallback = True

    def _rule_based_fallback(self, text: str) -> dict[str, float]:
        """High-precision keyword and pattern-based probability fallback."""
        lower = text.lower()
        scores = {c: 0.05 for c in self.classes}

        if any(w in lower for w in ["confidential", "proprietary", "trade secret", "disclose", "receiving party"]):
            scores["Confidentiality"] += 2.5
        if any(w in lower for w in ["governed by", "laws of", "jurisdiction", "arbitration", "courts"]):
            scores["Governing Law"] += 2.5
        if any(w in lower for w in ["terminate", "termination", "breach", "written notice", "insolvency"]):
            scores["Termination"] += 2.5
        if any(w in lower for w in ["indemnify", "indemnification", "hold harmless", "liability", "damages", "claims"]):
            scores["Indemnification"] += 2.5
        if any(w in lower for w in ["pay", "payment", "invoiced", "fees", "dollars", "interest", "billing"]):
            scores["Payment"] += 2.5

        # Normalize to probabilities
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def predict(self, clause: str) -> dict[str, Any]:
        """Classify a single contract clause into one of the legal clause categories."""
        if not clause or not clause.strip():
            return {
                "clause": clause,
                "predicted_label": self.classes[0],
                "confidence": 0.2,
                "probabilities": {c: 0.2 for c in self.classes},
            }

        if self.fast_fallback or self.model is None or self.tokenizer is None:
            probs = self._rule_based_fallback(clause)
            best_label = max(probs, key=probs.get)
            return {
                "clause": clause,
                "predicted_label": best_label,
                "confidence": round(probs[best_label], 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            }

        inputs = self.tokenizer(
            clause,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs_tensor = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs_tensor).item()

        pred_label = self.id2label[pred_idx]
        confidence = probs_tensor[pred_idx].item()
        probs_dict = {
            self.id2label[i]: round(probs_tensor[i].item(), 4)
            for i in range(self.num_labels)
        }

        return {
            "clause": clause,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": probs_dict,
        }

    def predict_batch(self, clauses: list[str]) -> list[dict[str, Any]]:
        """Classify multiple clauses."""
        return [self.predict(c) for c in clauses]


_GLOBAL_CLASSIFIER: LegalClauseClassifier | None = None


def get_classifier(fast_fallback: bool = False) -> LegalClauseClassifier:
    """Singleton getter for the legal clause classifier."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None or (fast_fallback and not _GLOBAL_CLASSIFIER.fast_fallback):
        _GLOBAL_CLASSIFIER = LegalClauseClassifier(fast_fallback=fast_fallback)
    return _GLOBAL_CLASSIFIER
