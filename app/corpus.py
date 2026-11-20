"""Legal text corpus and preprocessing utilities for n-gram language modeling."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

# Default curated legal clauses covering key commercial contract categories:
# Confidentiality, Governing Law, Termination, Indemnification, Payment
DEFAULT_LEGAL_CORPUS = [
    # Confidentiality Clauses
    "The Receiving Party agrees to hold and maintain all Confidential Information in strict confidence.",
    "The Receiving Party shall not disclose any Confidential Information to any third party without prior written consent.",
    "Confidential Information shall include all proprietary, technical, and commercial data disclosed by the Disclosing Party.",
    "The Receiving Party shall protect Confidential Information with the same degree of care it uses for its own confidential data.",
    "The obligations of confidentiality shall survive the termination or expiration of this Agreement for five years.",
    "Confidential Information does not include information that is publicly known or independently developed without breach.",
    "Upon written request, the Receiving Party shall return or destroy all materials containing Confidential Information.",
    "The Disclosing Party retains all right, title, and interest in and to its Confidential Information.",

    # Governing Law & Jurisdiction Clauses
    "This Agreement shall be governed by and construed in accordance with the laws of the State of New York.",
    "This Agreement and all disputes arising hereunder shall be interpreted under the laws of the State of Delaware.",
    "The parties irrevocably consent to the exclusive jurisdiction and venue of the courts located in New York County.",
    "Any dispute arising out of or relating to this Agreement shall be submitted to binding arbitration.",
    "The prevailing party in any legal enforcement action shall be entitled to recover reasonable attorney fees.",
    "Each party waives any right to a trial by jury in any dispute arising out of this Agreement.",
    "The United Nations Convention on Contracts for the International Sale of Goods shall not apply to this Agreement.",

    # Termination Clauses
    "Either party may terminate this Agreement upon thirty days prior written notice to the other party.",
    "Either party may terminate this Agreement immediately if the other party materially breaches any provision.",
    "In the event of insolvency or bankruptcy of either party, this Agreement may be terminated with immediate effect.",
    "Upon termination of this Agreement, the Client shall promptly pay all outstanding invoices for completed services.",
    "Termination of this Agreement shall not affect any accrued rights or liabilities of either party prior to termination.",
    "The provisions regarding confidentiality, indemnification, and liability shall survive the termination of this Agreement.",
    "Neither party may terminate this Agreement without cause during the initial one year commitment period.",

    # Indemnification & Liability Clauses
    "The Service Provider agrees to defend, indemnify, and hold harmless the Client against any third-party claims.",
    "The Client agrees to indemnify and hold harmless the Service Provider from damages arising out of Client materials.",
    "In no event shall either party be liable for any indirect, incidental, special, or consequential damages.",
    "The total aggregate liability of either party under this Agreement shall not exceed the fees paid in the prior twelve months.",
    "The indemnified party shall promptly notify the indemnifying party in writing of any claim subject to indemnification.",
    "Neither party excludes or limits liability for gross negligence, willful misconduct, or fraud.",
    "Each party shall maintain adequate insurance coverage to support its indemnification obligations hereunder.",

    # Payment & Consideration Clauses
    "The Client shall pay all invoiced amounts within thirty days of the date of invoice.",
    "Payment shall be made in United States Dollars by electronic wire transfer or credit card.",
    "Late payments shall accrue interest at the rate of one and one-half percent per month or the maximum permitted by law.",
    "All fees payable under this Agreement are exclusive of applicable taxes, duties, and government assessments.",
    "The Client shall reimburse the Service Provider for all reasonable pre-approved travel and out-of-pocket expenses.",
    "Except as explicitly provided in this Agreement, all payment obligations are non-refundable and non-cancellable.",
    "In the event of a good faith billing dispute, the Client shall notify the Provider within fifteen days of invoice receipt."
]


def clean_text(text: str) -> str:
    """Normalize text: strip extra whitespace, normalize quotes, and remove strange control characters."""
    if not text:
        return ""
    # Standardize curly quotes and apostrophes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace newlines and excessive whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(
    sentence: str,
    lowercase: bool = True,
    add_boundaries: bool = False,
    preserve_punctuation: bool = False,
) -> list[str]:
    """Tokenize a sentence into words and optionally boundary tokens.
    
    Args:
        sentence: Raw sentence text.
        lowercase: Whether to convert tokens to lowercase.
        add_boundaries: Whether to prepend <s> and append </s> tokens.
        preserve_punctuation: Whether to keep punctuation as distinct tokens.
    """
    sentence = clean_text(sentence)
    if lowercase:
        sentence = sentence.lower()

    if preserve_punctuation:
        tokens = re.findall(r"\b[\w'-]+\b|[.,!?;]", sentence)
    else:
        # Standard word tokenization stripping punctuation
        tokens = re.findall(r"\b[a-zA-Z0-9'-]+\b", sentence)

    if add_boundaries:
        tokens = ["<s>"] + tokens + ["</s>"]

    return tokens


def split_sentences(text: str) -> list[str]:
    """Split a text block into individual sentences."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Split on sentence boundaries: period, question mark, exclamation followed by space/end
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def load_legal_corpus(filepath: str | Path | None = None) -> list[str]:
    """Load sentences from a file or return DEFAULT_LEGAL_CORPUS."""
    if filepath is None:
        return list(DEFAULT_LEGAL_CORPUS)
    
    path = Path(filepath)
    if not path.exists():
        return list(DEFAULT_LEGAL_CORPUS)
    
    content = path.read_text(encoding="utf-8")
    sentences = split_sentences(content)
    return sentences or list(DEFAULT_LEGAL_CORPUS)



def prepare_corpus(
    source: str | Path | list[str] | None = None,
    lowercase: bool = True,
    add_boundaries: bool = False,
) -> dict[str, Any]:
    """Prepare and preprocess a legal text corpus for n-gram modeling.
    
    Returns a dictionary containing:
        - raw_sentences: original list of sentence strings
        - tokenized_sentences: list of token lists per sentence
        - vocabulary: set of unique token types
        - total_tokens: total word count across corpus
        - vocab_size: count of unique words
        - token_frequencies: Counter of token occurrences
    """
    if isinstance(source, list):
        sentences = [clean_text(s) for s in source if clean_text(s)]
    elif isinstance(source, (str, Path)):
        sentences = load_legal_corpus(source)
    else:
        sentences = list(DEFAULT_LEGAL_CORPUS)

    tokenized_sentences: list[list[str]] = []
    freq_counter: Counter[str] = Counter()

    for sentence in sentences:
        tokens = tokenize(sentence, lowercase=lowercase, add_boundaries=add_boundaries)
        if tokens:
            tokenized_sentences.append(tokens)
            freq_counter.update(tokens)

    vocab = set(freq_counter.keys())

    return {
        "raw_sentences": sentences,
        "tokenized_sentences": tokenized_sentences,
        "vocabulary": vocab,
        "total_tokens": sum(freq_counter.values()),
        "vocab_size": len(vocab),
        "token_frequencies": dict(freq_counter.most_common()),
    }
