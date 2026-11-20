"""Comprehensive unit and integration tests for the legal drafting assistant."""
import pytest
from app.corpus import clean_text, load_legal_corpus, prepare_corpus, split_sentences, tokenize
from app.ngram import NGramPredictor
from app.classifier import LegalClauseClassifier, get_classifier
from app.evaluate import evaluate_classifier
from app.cli import build_parser, main


def test_corpus_edge_cases(tmp_path):
    # Empty string handling
    assert clean_text("") == ""
    assert tokenize("") == []
    assert split_sentences("") == []

    # Custom file loading
    custom_file = tmp_path / "custom.txt"
    custom_file.write_text("Clause A. Clause B.", encoding="utf-8")
    loaded = load_legal_corpus(str(custom_file))
    assert loaded == ["Clause A.", "Clause B."]

    # Non-existent file fallback
    fallback = load_legal_corpus(str(tmp_path / "non_existent.txt"))
    assert len(fallback) > 0

    # Corpus preparation with custom list
    prep = prepare_corpus(["Party A pays Party B.", "Confidential data shall be kept safe."])
    assert prep["total_tokens"] > 5
    assert "party" in prep["vocabulary"]


def test_ngram_smoothing_strategies():
    sentences = [
        ["the", "receiving", "party", "shall", "hold", "confidential", "information"],
        ["the", "disclosing", "party", "shall", "disclose", "confidential", "information"],
    ]

    # Test Laplace smoothing
    laplace_model = NGramPredictor(n=3, smoothing="laplace", k=0.5)
    laplace_model.fit(sentences)
    probs_laplace = laplace_model.get_next_word_probabilities(["the", "receiving"])
    assert "party" in probs_laplace
    assert probs_laplace["party"] > 0.1

    # Test Linear Interpolation smoothing
    interp_model = NGramPredictor(n=3, smoothing="interpolation")
    interp_model.fit(sentences)
    probs_interp = interp_model.get_next_word_probabilities(["the", "unknown_context"])
    assert len(probs_interp) > 0
    assert abs(sum(probs_interp.values()) - 1.0) < 1e-4

    # Test Greedy generation (temp <= 0.01)
    gen_greedy = laplace_model.generate(seed="the receiving", max_tokens=6, temperature=0.0, seed_random=1)
    assert gen_greedy.startswith("The receiving")


def test_ngram_invalid_order():
    with pytest.raises(ValueError):
        NGramPredictor(n=0)


def test_clause_classifier_batch_and_edge_cases():
    clf = LegalClauseClassifier(fast_fallback=True)

    # Empty clause
    res_empty = clf.predict("")
    assert res_empty["predicted_label"] == clf.classes[0]

    # Batch prediction
    clauses = [
        "In the event of default, either party may terminate this agreement upon written notice.",
        "Supplier shall defend and indemnify Customer against third party patent claims.",
        "Payment is due net thirty days from invoice transmission.",
    ]
    batch_res = clf.predict_batch(clauses)
    assert len(batch_res) == 3
    assert batch_res[0]["predicted_label"] == "Termination"
    assert batch_res[1]["predicted_label"] == "Indemnification"
    assert batch_res[2]["predicted_label"] == "Payment"


def test_evaluation_suite():
    res = evaluate_classifier(split="test", fast_fallback=True)
    assert "accuracy" in res
    assert "macro_f1" in res
    assert res["accuracy"] >= 0.8
    assert res["total_samples"] == 10
    assert len(res["sample_results"]) == 10

    # Ensure baseline calculation is correct
    assert res["baseline_random_accuracy"] == 0.20
    assert res["accuracy_gain_vs_random"] > 0.50


def test_cli_subparsers_and_execution(capsys):
    parser = build_parser()
    assert parser.prog == "legal-drafting-assistant"

    # Test evaluate CLI command
    main(["evaluate"])
    out = capsys.readouterr().out
    assert "Legal Clause Classification Evaluation" in out
    assert "Model Accuracy:" in out
    assert "Random Guess Baseline:" in out
