# Project Update Summary

This repository currently implements an end-to-end pipeline for evaluating AI support in German obstetric informed-consent communication.

## 1. Current Scope

The work is organized into three connected parts:
- Information extraction from medical consent documents (to create a ground truth list of points that are required to be discussed with the patient)
- Simulated doctor-patient conversations with retrieval and supervision (for evaluation of the chatbot performance, since we have no real data)
- Multi-metric evaluation of extraction and dialogue quality

## 2. Implemented System Components

### 2.1 Information Extraction
- Four extraction methods are implemented:
  - Naive prompting
  - Chain-of-thought prompting
  - Atomic fact extraction (two-stage)
  - Schema-guided extraction (UIE)
- Batch processing across models/documents is available via the extraction script.
- Outputs are stored in structured JSON format.
- Ground truth list of discussion points is defined.

### 2.2 Conversational Pipeline
- A retrieval-augmented doctor agent answers patient questions and includes source citations.
- A patient simulator supports personas and hidden-risk conditions.
- A dialogue manager controls turn flow and mandatory-question interventions.
- Dataset generation supports active/passive modes and resume-safe indexing.

### 2.3 Document Retrieval and Indexing
- Source documents are chunked, embedded, and stored in a vector store.
- Retrieval is performed per user turn and passed into response generation.

## 3. Implemented Evaluation Framework

The evaluation stack currently covers four main points:

### 3.1 Information Extraction (IE)
- Embedding-similarity metrics: Precision, Recall, F1
- SUSWIR metrics: SSF, RDF, REF, BAF, and combined SUSWIR score

### 3.2 Topic Coverage
Three approaches are implemented:
- Embedding + similarity with Hungarian one-to-one matching
- Bi-encoder retrieval + cross-encoder entailment scoring
- LLM-as-judge coverage scoring

Primary coverage metrics:
- hit_rate
- weighted_critical_recall

### 3.3 Citation
- LLM-as-judge only (because from the coverage evaluation I have experienced that embeddings and bi-encoders do not capture similarity between strings when they are paraphrased)
- Metrics include strict and relaxed citation precision, support coverage, and support distribution (full/partial/no support)

### 3.4 Mandatory Questions
- LLM-as-judge only (same reason as above)
- Metrics include question recall, strict/acceptable compliance, first mandatory-question turn, and judge failure rate

## 4. Data and Outputs

- Information extraction results
- Conversations (two datasets: 60 conversations with rather primitive patient personas, 216 conversations with more natural personas)
- Evaluation artifacts

The project contains scripts for reruns, threshold recalculation, and checkpoint-safe execution.

## 5. Observed Gaps / Maintenance Items

- Final plotting/reporting is partially prepared but not yet the primary finalized output layer.

## 6. Next Steps

- Finalize the thesis-facing results tables from existing evaluation outputs.
- Harmonize README and method documentation with the current code state.
- Run final end-to-end evaluation pass and freeze artifacts for writing.


## 7. Written work

- Introduction chapter is ready.
- Fundamentals and related work is ready.
- Methodology, experiments and results are in progress (waiting for the final evaluation to be done).
