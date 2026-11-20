"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import math
import pytest

# Milestone 1: Prepare a small legal text corpus for n-gram training.
def test_milestone_1():
    from app.core import milestone_1
    from app.corpus import clean_text, tokenize, split_sentences

    # 1. Verify text cleaning
    raw = "  “Confidential   Information” — shall remain strictly private.  "
    cleaned = clean_text(raw)
    assert '"Confidential Information" - shall remain strictly private.' == cleaned

    # 2. Verify tokenization
    tokens = tokenize("This Agreement shall be governed by law.", lowercase=True)
    assert tokens == ["this", "agreement", "shall", "be", "governed", "by", "law"]

    # 3. Verify sentence splitting
    sentences = split_sentences("First sentence. Second sentence! Third sentence?")
    assert len(sentences) == 3

    # 4. Verify corpus preparation via milestone_1
    result = milestone_1()
    assert result["sentence_count"] >= 30
    assert result["total_tokens"] > 300
    assert result["vocab_size"] > 100
    assert len(result["tokenized_sentences"]) == result["sentence_count"]
    assert "agreement" in result["vocabulary"]
    assert "confidential" in result["vocabulary"]
    assert "shall" in result["vocabulary"]



# Milestone 2: Implement the n-gram next-word predictor.
def test_milestone_2():
    from app.core import milestone_2
    from app.ngram import NGramPredictor

    # 1. Direct NGramPredictor unit tests (bigram and trigram)
    sentences = [
        ["this", "agreement", "shall", "be", "governed", "by", "law"],
        ["this", "agreement", "may", "be", "terminated", "by", "notice"],
        ["either", "party", "may", "terminate", "this", "agreement"],
    ]

    bigram = NGramPredictor(n=2, smoothing="laplace", k=1.0)
    bigram.fit(sentences)

    # Context "this" -> "agreement" should have high probability
    preds = bigram.predict_next_word("this", top_k=3)
    words = [w for w, _ in preds]
    assert "agreement" in words

    # Test generation from seed
    draft = bigram.generate(seed="this agreement", max_tokens=8, seed_random=42)
    assert draft.startswith("This agreement")
    assert draft.endswith(".")

    # Test perplexity calculation
    ppl = bigram.perplexity(sentences)
    assert ppl > 0 and not math.isinf(ppl)

    # 2. Test milestone_2 end-to-end function
    res = milestone_2(n=3)
    assert "predictor" in res
    assert res["n"] == 3
    assert len(res["prompts"]) > 0
    assert "This Agreement shall" in res["generations"]
    assert len(res["predictions"]["This Agreement shall"]) > 0



# Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
def test_milestone_3():
    from app.core import milestone_3
    from app.classifier import LegalClauseClassifier

    # 1. Test classifier direct instantiation (fast fallback test)
    classifier = LegalClauseClassifier(fast_fallback=True)
    res = classifier.predict("The Receiving Party shall hold all Confidential Information in strict confidence.")
    assert res["predicted_label"] == "Confidentiality"
    assert res["confidence"] > 0.5
    assert "Confidentiality" in res["probabilities"]

    # Test governing law clause
    res_law = classifier.predict("This Agreement shall be governed by the laws of Delaware.")
    assert res_law["predicted_label"] == "Governing Law"

    # 2. Test milestone_3 end-to-end function
    m3_res = milestone_3(fast_fallback=True)
    assert "classifier" in m3_res
    assert len(m3_res["predictions"]) == 5
    predicted_labels = [p["predicted_label"] for p in m3_res["predictions"]]
    assert "Confidentiality" in predicted_labels
    assert "Governing Law" in predicted_labels
    assert "Termination" in predicted_labels
    assert "Indemnification" in predicted_labels
    assert "Payment" in predicted_labels



# Milestone 4: Create a CLI interface to generate text and then classify the generated text.
def test_milestone_4(capsys):
    from app.core import milestone_4
    from app.cli import main

    # 1. Test milestone_4 programmatic integration
    res = milestone_4(seed="The Receiving Party agrees to", n=3, fast_fallback=True)
    assert res["seed"] == "The Receiving Party agrees to"
    assert len(res["generated_text"]) > 10
    assert res["predicted_label"] in ["Confidentiality", "Governing Law", "Termination", "Indemnification", "Payment"]
    assert res["confidence"] > 0

    # 2. Test CLI commands via main()
    main(["draft", "--seed", "This Agreement shall", "--fast"])
    out = capsys.readouterr().out
    assert "Generated Legal Draft" in out
    assert "Clause Classification" in out

    main(["generate", "--seed", "Either party may terminate", "--n", "2"])
    out_gen = capsys.readouterr().out
    assert "Generated:" in out_gen

    main(["classify", "This Agreement shall be governed by New York law.", "--fast"])
    out_cls = capsys.readouterr().out
    assert "Predicted Type:   Governing Law" in out_cls



# Milestone 5: Evaluate classification accuracy on example clauses.
def test_milestone_5():
    from app.core import milestone_5
    from app.evaluate import evaluate_classifier

    # Run evaluation using fast fallback
    res = milestone_5(fast_fallback=True)

    # 1. Verify accuracy and sample counts
    assert res["total_samples"] == 10
    assert res["accuracy"] >= 0.80

    # 2. Verify baseline comparison
    assert res["baseline_random_accuracy"] == 0.20
    assert res["accuracy_gain_vs_random"] > 0.50

    # 3. Verify per-class metrics
    for cls_name in ["Confidentiality", "Governing Law", "Termination", "Indemnification", "Payment"]:
        assert cls_name in res["per_class_metrics"]
        metrics = res["per_class_metrics"][cls_name]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["support"] == 2

    # 4. Verify confusion matrix and limits
    assert len(res["confusion_matrix"]) == 5
    assert len(res["limits"]) > 0


