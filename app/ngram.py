"""N-gram language model and next-word predictor built from scratch."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

from app.corpus import clean_text, tokenize


class NGramPredictor:
    """N-gram next-word predictor supporting bigrams, trigrams, and higher order n-grams.
    
    Includes Laplace (Add-k) smoothing, linear interpolation backoff, and text generation.
    """

    def __init__(
        self,
        n: int = 3,
        smoothing: str = "laplace",
        k: float = 1.0,
        lambdas: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize the predictor.
        
        Args:
            n: Order of n-gram (e.g. 2 for bigram, 3 for trigram).
            smoothing: Smoothing strategy ('none', 'laplace', 'add_k', 'interpolation').
            k: Smoothing constant for add-k / Laplace.
            lambdas: Weights for linear interpolation across orders.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.lambdas = lambdas or self._default_lambdas(n)

        # n-gram count tables: order -> context_tuple -> Counter of next_token
        self.ngram_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        # Context count tables: order -> context_tuple -> total occurrences
        self.context_counts: dict[int, Counter[tuple[str, ...]]] = defaultdict(Counter)
        # Unigram frequency table
        self.unigram_counts: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.total_tokens: int = 0

    @staticmethod
    def _default_lambdas(n: int) -> tuple[float, ...]:
        """Generate normalized descending weights giving higher weight to higher order n-grams."""
        raw = [2.0 ** i for i in range(n)]
        total = sum(raw)
        return tuple(r / total for r in raw)

    def fit(self, tokenized_sentences: list[list[str]]) -> NGramPredictor:
        """Train the n-gram model on tokenized sentences."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.unigram_counts.clear()
        self.vocab.clear()
        self.total_tokens = 0

        for sentence in tokenized_sentences:
            if not sentence:
                continue
            # Pad sentence with start tokens
            pad = ["<s>"] * (self.n - 1)
            padded = pad + sentence + ["</s>"]

            for token in sentence:
                self.vocab.add(token)
                self.unigram_counts[token] += 1
                self.total_tokens += 1
            self.vocab.add("</s>")
            self.unigram_counts["</s>"] += 1
            self.total_tokens += 1

            # Build counts for all orders 1..n
            for order in range(1, self.n + 1):
                ctx_len = order - 1
                for i in range(ctx_len, len(padded)):
                    context = tuple(padded[i - ctx_len : i])
                    target = padded[i]
                    self.ngram_counts[order][context][target] += 1
                    self.context_counts[order][context] += 1

        return self

    def _get_mle_prob(self, order: int, context: tuple[str, ...], word: str) -> float:
        """Maximum Likelihood Estimation probability P(word | context)."""
        ctx_count = self.context_counts[order][context]
        if ctx_count == 0:
            return 0.0
        return self.ngram_counts[order][context][word] / ctx_count

    def _get_laplace_prob(self, order: int, context: tuple[str, ...], word: str) -> float:
        """Laplace / Add-k smoothed probability."""
        ctx_count = self.context_counts[order][context]
        word_count = self.ngram_counts[order][context][word]
        vocab_size = max(len(self.vocab), 1)
        return (word_count + self.k) / (ctx_count + self.k * vocab_size)

    def _get_interpolated_prob(self, full_context: tuple[str, ...], word: str) -> float:
        """Linear interpolation across unigram, bigram, ..., n-gram models."""
        prob = 0.0
        for order in range(1, self.n + 1):
            ctx_len = order - 1
            ctx = full_context[len(full_context) - ctx_len :] if ctx_len > 0 else ()
            weight = self.lambdas[order - 1]
            if order == 1:
                # Unigram probability
                unigram_prob = (
                    (self.unigram_counts[word] + self.k) / (self.total_tokens + self.k * len(self.vocab))
                    if self.total_tokens > 0
                    else 1.0 / max(len(self.vocab), 1)
                )
                prob += weight * unigram_prob
            else:
                prob += weight * self._get_laplace_prob(order, ctx, word)
        return prob

    def get_next_word_probabilities(self, context: list[str] | str) -> dict[str, float]:
        """Compute the next-word probability distribution given a context."""
        if isinstance(context, str):
            tokens = tokenize(context, lowercase=True)
        else:
            tokens = [t.lower() for t in context]

        # Extract the relevant context of length (n-1)
        ctx_len = self.n - 1
        if len(tokens) < ctx_len:
            pad = ["<s>"] * (ctx_len - len(tokens))
            ctx_tuple = tuple(pad + tokens)
        else:
            ctx_tuple = tuple(tokens[-ctx_len:]) if ctx_len > 0 else ()

        probs: dict[str, float] = {}

        if self.smoothing == "interpolation":
            for word in self.vocab:
                if word == "<s>":
                    continue
                probs[word] = self._get_interpolated_prob(ctx_tuple, word)
        elif self.smoothing in ("laplace", "add_k"):
            for word in self.vocab:
                if word == "<s>":
                    continue
                probs[word] = self._get_laplace_prob(self.n, ctx_tuple, word)
        else:
            # MLE with fallback to unigram
            for word in self.vocab:
                if word == "<s>":
                    continue
                p = self._get_mle_prob(self.n, ctx_tuple, word)
                if p > 0:
                    probs[word] = p

        # Normalize probabilities so they sum to 1.0
        total = sum(probs.values())
        if total > 0:
            probs = {w: p / total for w, p in probs.items()}
        return probs

    def predict_next_word(
        self, context: list[str] | str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return the top-k most likely next words with their probabilities."""
        probs = self.get_next_word_probabilities(context)
        # Exclude internal boundary tokens from direct completion
        filtered = [(w, p) for w, p in probs.items() if w not in ("<s>", "</s>")]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered[:top_k]

    def generate(
        self,
        seed: str = "",
        max_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 5,
        seed_random: int | None = None,
    ) -> str:
        """Generate a coherent legal draft sentence starting from a seed phrase."""
        if seed_random is not None:
            random.seed(seed_random)

        seed_cleaned = clean_text(seed)
        tokens = tokenize(seed_cleaned, lowercase=True) if seed_cleaned else []
        if not tokens:
            # If no seed, pick from common legal sentence beginnings
            starters = ["this", "the", "either", "in", "upon", "all"]
            tokens = [random.choice([s for s in starters if s in self.vocab] or ["the"])]

        generated_tokens = list(tokens)

        for _ in range(max_tokens):
            candidates = self.predict_next_word(generated_tokens, top_k=top_k)
            if not candidates:
                break

            words, probs = zip(*candidates)

            if temperature <= 0.01:
                # Greedy selection
                next_word = words[0]
            else:
                # Temperature scaling
                scaled_logits = [math.log(max(p, 1e-9)) / temperature for p in probs]
                max_logit = max(scaled_logits)
                exp_logits = [math.exp(lg - max_logit) for lg in scaled_logits]
                total_exp = sum(exp_logits)
                adj_probs = [exp_l / total_exp for exp_l in exp_logits]
                next_word = random.choices(words, weights=adj_probs, k=1)[0]

            generated_tokens.append(next_word)

            # Check if sentence terminal punctuation condition met
            if next_word in ("agreement", "party", "notice", "damages", "provision", "months") and len(generated_tokens) >= 15:
                break

        # Capitalize and format output sentence
        sentence = " ".join(generated_tokens)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith("."):
                sentence += "."
        return sentence

    def perplexity(self, test_sentences: list[list[str]]) -> float:
        """Calculate language model perplexity on a test set."""
        log_prob_sum = 0.0
        n_tokens = 0

        for sentence in test_sentences:
            if not sentence:
                continue
            pad = ["<s>"] * (self.n - 1)
            padded = pad + sentence + ["</s>"]
            ctx_len = self.n - 1

            for i in range(ctx_len, len(padded)):
                ctx = tuple(padded[i - ctx_len : i])
                word = padded[i]
                prob = self._get_laplace_prob(self.n, ctx, word)
                log_prob_sum += math.log2(max(prob, 1e-12))
                n_tokens += 1

        if n_tokens == 0:
            return float("inf")
        avg_neg_log_prob = -log_prob_sum / n_tokens
        return 2.0 ** avg_neg_log_prob
