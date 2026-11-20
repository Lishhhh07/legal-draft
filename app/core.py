"""Your code. One stub per milestone; build them in order and keep each one small."""

from __future__ import annotations

from typing import Any
from app.corpus import clean_text, tokenize, split_sentences, prepare_corpus, DEFAULT_LEGAL_CORPUS

# --- Milestone 1: Prepare a small legal text corpus for n-gram training.
# Hint: Use text processing techniques from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_1(corpus_path: str | None = None) -> dict[str, Any]:
    """Prepare and tokenize the legal corpus for n-gram training.
    
    Processes the legal text corpus with normalization, tokenization,
    vocabulary extraction, and frequency counting.
    """
    processed = prepare_corpus(corpus_path or "data/legal_corpus.txt", lowercase=True)
    return {
        "sentence_count": len(processed["raw_sentences"]),
        "total_tokens": processed["total_tokens"],
        "vocab_size": processed["vocab_size"],
        "vocabulary": processed["vocabulary"],
        "tokenized_sentences": processed["tokenized_sentences"],
        "token_frequencies": processed["token_frequencies"],
    }



from app.ngram import NGramPredictor

# --- Milestone 2: Implement the n-gram next-word predictor.
# Hint: Apply the n-gram model building and prediction steps from the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_2(n: int = 3, smoothing: str = "laplace") -> dict[str, Any]:
    """Train an n-gram next-word predictor on the legal corpus and generate sample drafts.
    
    Returns the trained predictor model, next-word predictions for legal prompts,
    and generated legal sentences.
    """
    corpus_info = milestone_1()
    predictor = NGramPredictor(n=n, smoothing=smoothing, k=1.0)
    predictor.fit(corpus_info["tokenized_sentences"])

    prompts = [
        "This Agreement shall",
        "Either party may terminate",
        "The Receiving Party agrees to",
        "The Service Provider agrees to defend",
    ]

    predictions: dict[str, list[tuple[str, float]]] = {}
    generations: dict[str, str] = {}

    for prompt in prompts:
        predictions[prompt] = predictor.predict_next_word(prompt, top_k=5)
        generations[prompt] = predictor.generate(seed=prompt, max_tokens=15, temperature=0.7, seed_random=42)

    return {
        "predictor": predictor,
        "n": n,
        "smoothing": smoothing,
        "vocab_size": len(predictor.vocab),
        "prompts": prompts,
        "predictions": predictions,
        "generations": generations,
    }



from app.classifier import LegalClauseClassifier, get_classifier

# --- Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
# Hint: Follow the steps for loading a fine-tuned GPT-2 model from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_3(fast_fallback: bool = False) -> dict[str, Any]:
    """Load the pre-trained GPT-2 model with fine-tuned adapter for legal clause tagging.
    
    Classifies standard test legal contract clauses and returns predictions,
    confidence scores, and probability distributions.
    """
    classifier = get_classifier(fast_fallback=fast_fallback)
    test_clauses = [
        "The Receiving Party agrees to maintain all proprietary technical data in strict confidence.",
        "This Agreement shall be governed by and construed in accordance with the laws of New York.",
        "Either party may terminate this Agreement upon thirty days written notice.",
        "Service Provider shall defend, indemnify, and hold harmless Client against third-party claims.",
        "The Client shall pay all undisputed invoiced amounts within thirty days of invoice receipt.",
    ]

    results = classifier.predict_batch(test_clauses)
    return {
        "classifier": classifier,
        "classes": classifier.classes,
        "device": str(classifier.device) if hasattr(classifier, "device") else "cpu",
        "test_clauses": test_clauses,
        "predictions": results,
    }



# --- Milestone 4: Create a CLI interface to generate text and then classify the generated text.
# Hint: Integrate the n-gram predictor and the fine-tuned GPT-2 model into a command-line interface.
def milestone_4(
    seed: str = "This Agreement shall",
    n: int = 3,
    max_tokens: int = 15,
    temperature: float = 0.7,
    fast_fallback: bool = False,
) -> dict[str, Any]:
    """Generate legal text using n-gram predictor and instantly classify it with fine-tuned GPT-2."""
    corpus_info = milestone_1()
    predictor = NGramPredictor(n=n, smoothing="laplace")
    predictor.fit(corpus_info["tokenized_sentences"])

    generated_text = predictor.generate(
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        seed_random=42,
    )

    classifier = get_classifier(fast_fallback=fast_fallback)
    classification = classifier.predict(generated_text)

    return {
        "seed": seed,
        "n": n,
        "generated_text": generated_text,
        "predicted_label": classification["predicted_label"],
        "confidence": classification["confidence"],
        "probabilities": classification["probabilities"],
    }



from app.evaluate import evaluate_classifier

# --- Milestone 5: Evaluate classification accuracy on example clauses.
# Hint: Use evaluation methods for classification tasks, as demonstrated in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_5(
    fast_fallback: bool = False, dataset_path: str | None = None
) -> dict[str, Any]:
    """Evaluate classification accuracy, precision, recall, and baselines on example clauses."""
    return evaluate_classifier(
        dataset_path=dataset_path,
        split="test",
        fast_fallback=fast_fallback,
    )

