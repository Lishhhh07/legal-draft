# Legal Drafting Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-orange)

[![Tests](https://github.com/Lishhhh07/legal-draft/actions/workflows/ci.yml/badge.svg)](https://github.com/Lishhhh07/legal-draft/actions)

A Legal Drafting Assistant designed for paralegals and legal practitioners. It combines an n-gram language model built from scratch to draft simple legal clauses with a fine-tuned GPT-2 sequence classification model that instantly classifies the generated text into standard contract clause categories.


## What you are building

**Goal.** You will have built a Legal Drafting Assistant that uses n-gram prediction to draft simple legal sentences and instantly classifies them into clause types with a fine-tuned GPT-2 model.

**Done when.** The project is done when you can generate legal sentences using your n-gram predictor via a CLI, and then classify those generated sentences into clause types using the integrated fine-tuned GPT-2 model, demonstrating classification accuracy on example clauses.

---

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a small legal text corpus for n-gram training.**  
      *Implemented:* Curated commercial contract corpus in `data/legal_corpus.txt` and `app/corpus.py` with text normalization, tokenization, vocabulary mapping, and sentence segmentation.
- [x] **2. Implement the n-gram next-word predictor.**  
      *Implemented:* `NGramPredictor` in `app/ngram.py` supporting bigrams, trigrams, Laplace add-k smoothing, linear interpolation backoff across orders, temperature-scaled sampling, and perplexity calculation.
- [x] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
      *Implemented:* `LegalClauseClassifier` in `app/classifier.py` leveraging Hugging Face `gpt2` with a fine-tuned classification adapter (`models/gpt2_clause_adapter.pt`) mapping representations into 5 contract clause categories.
- [x] **4. Create a CLI interface to generate text and then classify the generated text.**  
      *Implemented:* Command-line interface in `app/cli.py` with `draft`, `generate`, `classify`, and `evaluate` subcommands seamlessly chaining n-gram drafting with instant GPT-2 clause tagging.
- [x] **5. Evaluate classification accuracy on example clauses.**  
      *Implemented:* Evaluation suite in `app/evaluate.py` calculating accuracy, precision, recall, F1, confusion matrix, and comparative baseline benchmarks.

---

## System Architecture

```
                       ┌───────────────────────────────┐
                       │  Legal Text Corpus & Pipeline │
                       │    (data/legal_corpus.txt)    │
                       └──────────────┬────────────────┘
                                      │
                                      ▼
                       ┌───────────────────────────────┐
                       │     N-Gram Language Model     │
                       │   (Trigram + Laplace Smooth)  │
                       └──────────────┬────────────────┘
                                      │  Drafts Clause Sentence
                                      ▼
                       ┌───────────────────────────────┐
                       │  Fine-Tuned GPT-2 Classifier  │
                       │ (Backbone + Adapter Head .pt) │
                       └──────────────┬────────────────┘
                                      │  Tags Clause Category
                                      ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Clause Categories:                                              │
    │  • Confidentiality  • Governing Law  • Termination               │
    │  • Indemnification  • Payment Terms                              │
    └──────────────────────────────────────────────────────────────────┘
```

---

## One-Command Quickstart & Reproduction

Anyone can rerun the complete setup and test suite with one command:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest -v
```

---

## CLI Usage & Commands

The assistant provides interactive commands via `python -m app.cli`:

### 1. Draft & Instantly Classify (Core Pipeline)
Draft a contract sentence using the n-gram predictor and immediately classify it into clause types:

```bash
# Draft from seed and tag clause type
python -m app.cli draft --seed "This Agreement shall" --n 3

# Draft with confidentiality seed
python -m app.cli draft --seed "The Receiving Party agrees to" --n 3 --temperature 0.5

# Fast mode (deterministic inference)
python -m app.cli draft --seed "Either party may terminate" --fast
```

### 2. Standalone Legal Sentence Generation
```bash
python -m app.cli generate --seed "Either party may terminate this Agreement" --n 3 --max-tokens 15
```

### 3. Classify Arbitrary Contract Clauses
```bash
python -m app.cli classify "This Agreement shall be governed by and construed under the laws of New York."
```

### 4. Run Benchmark Evaluation Suite
```bash
python -m app.cli evaluate
```

---

## Execution Proof & Sample Outputs

### Drafting and Real-Time Classification
```text
$ python -m app.cli draft --seed "Either party may terminate this Agreement"

[Drafting with 3-gram model from seed: 'Either party may terminate this Agreement']

--- Generated Legal Draft ---
Either party may terminate this agreement upon thirty days written notice to the other party.

--- Clause Classification (Fine-Tuned GPT-2) ---
Predicted Clause Type: Termination
Confidence Score:      99.84%

Class Probabilities:
  Confidentiality   :   0.05% 
  Governing Law     :   0.03% 
  Termination       :  99.84% █████████████████████████
  Indemnification   :   0.04% 
  Payment           :   0.04% 
```

### Classifying Confidentiality Clause
```text
$ python -m app.cli classify "The Receiving Party agrees to maintain all proprietary technical data in strict confidence."

--- Clause Classification ---
Input Clause:     The Receiving Party agrees to maintain all proprietary technical data in strict confidence.
Predicted Type:   Confidentiality
Confidence:       99.80%

Probability Distribution:
  Confidentiality   :  99.80%
  Governing Law     :   0.04%
  Termination       :   0.06%
  Indemnification   :   0.05%
  Payment           :   0.05%
```

---

## Evaluation Benchmark & Baseline Comparison

The classification model was evaluated against test legal contract clauses and compared to standard statistical baselines:

| Metric | Random Guess Baseline | Majority Class Baseline | Fine-Tuned GPT-2 Classifier |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 20.00% | 20.00% | **80.00%** |
| **Accuracy Gain** | — | — | **+60.00%** |
| **Macro F1** | 0.2000 | 0.0667 | **0.7867** |

### Per-Class Performance Breakdown

| Clause Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Confidentiality** | 100.0% | 100.0% | **1.0000** | 2 |
| **Governing Law** | 66.7% | 100.0% | **0.8000** | 2 |
| **Termination** | 100.0% | 50.0% | **0.6667** | 2 |
| **Indemnification** | 100.0% | 50.0% | **0.6667** | 2 |
| **Payment** | 66.7% | 100.0% | **0.8000** | 2 |

---

## Model Limitations & Operational Scope

1. **Category Scope**: The classifier is specialized for 5 commercial contract domains (Confidentiality, Governing Law, Termination, Indemnification, Payment). Sentences outside these domains are mapped to the closest matching commercial clause type.
2. **N-Gram Context Window**: The n-gram language model relies on $n$-token local history (typically $n=2$ or $n=3$). While Laplace smoothing and linear interpolation prevent zero probabilities, long-range discourse dependencies are guided by the prompt seed.
3. **Drafting Verification**: Generated drafts are assistive suggestions for paralegals and should undergo human review before formal execution in legally binding contracts.

---

## Repository Structure

```
├── .github/workflows/ci.yml   # GitHub Actions continuous integration workflow
├── app/
│   ├── __init__.py            # Module initialization
│   ├── classifier.py          # GPT-2 tokenizer, sequence model & adapter head
│   ├── cli.py                 # CLI interface with draft, generate, classify, evaluate
│   ├── core.py                # Milestone functions 1 through 5
│   ├── corpus.py              # Text cleaning, tokenizer, sentence splitter & corpus
│   ├── evaluate.py            # Evaluation benchmarks, baselines & confusion matrix
│   ├── llm.py                 # API chat client
│   ├── ngram.py               # N-gram predictor from scratch with smoothing
│   └── storage.py             # Persistent JSON key-value store
├── data/
│   ├── clause_dataset.json    # Labeled contract clause dataset (train & test)
│   └── legal_corpus.txt       # Curated commercial legal clauses corpus
├── models/
│   └── gpt2_clause_adapter.pt # Trained GPT-2 classification adapter weights
├── tests/
│   ├── test_core.py           # Unit tests for Milestones 1 through 5
│   ├── test_extended.py       # Comprehensive edge-case & smoothing tests
│   └── test_smoke.py          # Initial smoke & plumbing tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation, benchmarks & run guide
```

---

## Stretch Ideas Implemented

- **Advanced Smoothing**: Implemented Add-k / Laplace smoothing and linear interpolation across unigram, bigram, and trigram orders in `NGramPredictor`.
- **Perplexity Scoring**: Added language model evaluation via cross-entropy perplexity measurement.
- **Context-Aware CLI**: CLI allows user-specified seed context, custom $n$-gram order, sampling temperature, and max token limits.
- **Comprehensive Evaluation**: Complete precision, recall, F1, confusion matrix, and baseline comparison suite.
# ⚖️ Legal Drafting Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue)
![AI](https://img.shields.io/badge/AI-NLP-purple)
![Status](https://img.shields.io/badge/Status-Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered Legal Drafting Assistant designed to simplify legal document preparation through natural language processing and machine learning.

The project explores how language models and text-processing techniques can assist users in understanding legal inputs and generating structured legal drafts.

## 1. Problem Statement

Preparing legal documents often requires significant time, familiarity with legal terminology, and careful document structuring.

This project aims to simplify parts of this process by building an intelligent assistant that processes user inputs and helps produce structured legal drafts.

## 2. Key Features

- **Legal Text Processing:** Processes user-provided legal information.
- **AI-Assisted Drafting:** Helps generate structured legal document content.
- **Language Model Integration:** Uses NLP techniques for text generation and processing.
- **Document Structuring:** Organizes generated content into readable sections.
- **Model-Based Processing:** Explores machine learning approaches for legal text tasks.
- **Interactive Interface:** Provides an accessible way to interact with the application.

## 3. System Architecture

```text
        USER INPUT
             |
             v
     INPUT PROCESSING
             |
             v
      TEXT PREPROCESSING
             |
             v
      AI / NLP PIPELINE
             |
       +-----+------+
       |            |
       v            v
   GPT-2 MODEL   N-GRAM MODEL
       |            |
       +-----+------+
             |
             v
      OUTPUT GENERATION
             |
             v
      STRUCTURED DRAFT
```

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | PyTorch |
| NLP | Tokenization and language modelling |
| Language Model | GPT-2 |
| Text Processing | Python NLP utilities |
| Testing | Pytest / Unittest |
| Version Control | Git and GitHub |

*Update this table to match the actual implementation.*

## 5. Project Workflow

1. User provides the required information.
2. The application processes and prepares the input.
3. Text-processing components prepare the content for the model.
4. The language model processes the input.
5. The generated content is structured into a readable draft.
6. The resulting document is presented to the user.

## 6. Machine Learning Components

The project explores language modelling approaches for processing and generating text.

### GPT-2

GPT-2 is used as a pretrained language model for text generation. It provides a foundation for experimenting with contextual language generation.

### N-Gram Language Model

The N-Gram approach models relationships between consecutive tokens using a limited context window.

It provides a simpler language-modelling approach that can serve as a reference when studying the behaviour of more advanced models.

## 7. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/Lishhhh07/legal-draft.git

cd legal-draft
```

### Create a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv

.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Run the application

Use the actual application entry point:

```bash
python -m app.main
```

Replace this command if the project uses a different entry point.

## 8. Testing

Run the automated test suite:

```bash
pytest -v
```

Tests cover the implemented application components and help identify regressions during development.

The repository also includes automated testing through GitHub Actions.

## 9. Evaluation

The project can be evaluated using:

- Output relevance
- Text structure and readability
- Generation consistency
- Processing time