"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from typing import Sequence

from app.core import milestone_1, milestone_4, milestone_5
from app.classifier import get_classifier
from app.ngram import NGramPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-drafting-assistant",
        description="Paralegals get an assistant to draft simple legal sentences using n-gram prediction and instantly classify them into clause types with a fine-tuned GPT-2 model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Smoke / sanity command
    sub.add_parser("hello", help="check that everything is wired up")

    # Milestone 4 core command: draft and instantly classify
    draft_parser = sub.add_parser(
        "draft",
        help="Draft a legal sentence using n-gram predictor and instantly classify it into clause types",
    )
    draft_parser.add_argument(
        "--seed",
        type=str,
        default="This Agreement shall",
        help="Seed prompt or starting phrase for n-gram generation (e.g. 'The Receiving Party agrees')",
    )
    draft_parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="N-gram order (e.g. 2 for bigram, 3 for trigram)",
    )
    draft_parser.add_argument(
        "--max-tokens",
        type=int,
        default=18,
        help="Maximum number of tokens to generate",
    )
    draft_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation sampling temperature (0 for greedy)",
    )
    draft_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast rule-based classifier fallback instead of neural GPT-2",
    )

    # Standalone generator command
    gen_parser = sub.add_parser("generate", help="Generate text using the n-gram predictor")
    gen_parser.add_argument(
        "--seed",
        type=str,
        default="Either party may terminate",
        help="Seed context for generation",
    )
    gen_parser.add_argument("--n", type=int, default=3, help="N-gram order")
    gen_parser.add_argument("--max-tokens", type=int, default=18, help="Max tokens")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")

    # Standalone clause classifier command
    cls_parser = sub.add_parser("classify", help="Classify a legal clause into clause types")
    cls_parser.add_argument("text", type=str, help="Legal clause text to classify")
    cls_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast classifier fallback",
    )

    # Milestone 5 evaluation command
    sub.add_parser("evaluate", help="Evaluate clause classification accuracy and metrics on example clauses")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Drafting Assistant is ready. Open README.md for your milestones.")

    elif args.command == "draft":
        print(f"\n[Drafting with {args.n}-gram model from seed: '{args.seed}']")
        res = milestone_4(
            seed=args.seed,
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            fast_fallback=args.fast,
        )
        print("\n--- Generated Legal Draft ---")
        print(res["generated_text"])
        print("\n--- Clause Classification (Fine-Tuned GPT-2) ---")
        print(f"Predicted Clause Type: {res['predicted_label']}")
        print(f"Confidence Score:      {res['confidence'] * 100:.2f}%")
        print("\nClass Probabilities:")
        for label, prob in res["probabilities"].items():
            bar = "█" * int(prob * 25)
            print(f"  {label:<18}: {prob * 100:6.2f}% {bar}")
        print()

    elif args.command == "generate":
        corpus_data = milestone_1()
        predictor = NGramPredictor(n=args.n, smoothing="laplace")
        predictor.fit(corpus_data["tokenized_sentences"])
        generated = predictor.generate(
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nSeed:      {args.seed}")
        print(f"Generated: {generated}\n")

    elif args.command == "classify":
        classifier = get_classifier(fast_fallback=args.fast)
        result = classifier.predict(args.text)
        print("\n--- Clause Classification ---")
        print(f"Input Clause:     {result['clause']}")
        print(f"Predicted Type:   {result['predicted_label']}")
        print(f"Confidence:       {result['confidence'] * 100:.2f}%")
        print("\nProbability Distribution:")
        for label, prob in result["probabilities"].items():
            print(f"  {label:<18}: {prob * 100:6.2f}%")
        print()

    elif args.command == "evaluate":
        res = milestone_5(fast_fallback=args.fast if hasattr(args, "fast") else True)
        print("\n=== Legal Clause Classification Evaluation ===")
        print(f"Evaluated Test Samples:  {res['total_samples']}")
        print(f"Model Accuracy:          {res['accuracy'] * 100:.2f}%")
        print(f"Random Guess Baseline:   {res['baseline_random_accuracy'] * 100:.2f}%")
        print(f"Majority Class Baseline: {res['baseline_majority_accuracy'] * 100:.2f}%")
        print(f"Accuracy Gain:          +{res['accuracy_gain_vs_random'] * 100:.2f}%")
        print("\n--- Per-Class Performance ---")
        for cls_name, metrics in res["per_class_metrics"].items():
            print(f"  {cls_name:<18} | Precision: {metrics['precision'] * 100:5.1f}% | Recall: {metrics['recall'] * 100:5.1f}% | F1: {metrics['f1'] * 100:5.1f}%")
        print()


if __name__ == "__main__":
    main()

