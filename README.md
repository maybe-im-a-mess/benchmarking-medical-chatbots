# Benchmarking Information Extraction and Communication Quality in Medical Chatbots - Master Thesis

This repository contains the source code and configuration files for the Master thesis: **"Benchmarking Information Extraction and Communication Quality in Medical Chatbots"** by Olha Solodovnyk.

## Abstract

Large Language Models (LLMs) are widely deployed in healthcare applications; however, their reliability and safety in medical dialogue systems remain insufficiently understood. Hallucination, contextual degradation, and the omission of critical information pose significant risks in clinical settings. This project proposes and evaluates a comprehensive benchmarking pipeline for LLM-based medical chatbots, developed in the context of the Patient Information Assistant (PIA) project for prenatal care consultations. 

The framework addresses four evaluation targets: 
1. **Information extraction quality** (benchmarked using Naive, Atomic Fact, Chain-of-Thought, and Schema-Guided prompting across multiple language models, using semantic matching and the adapted SUSWIR metric).
2. **Topic coverage** (evaluated via SBERT semantics, Hungarian matching, and LLM-as-a-Judge based on synthetic datasets).
3. **Citation faithfulness** (assessed using two-level strict/relaxed precision framework).
4. **Mandatory question compliance** (comparing naive vs. supervised dialogue management frameworks).

## Project Structure

```text
.
├── chatbot/                   # Chatbot implementation & AI Agents
│   ├── dialogue_manager.py    # Manages conversation flow and modes
│   ├── doctor_agent.py        # RAG-enabled doctor persona
│   ├── embeddings.py          # Document chunking & embedding
│   ├── patient_agent.py       # Multi-persona synthetic patient
│   └── retrieval.py           # Vector store retrieval
├── data/                      # Datasets, evaluation results and models
│   ├── comprehension_questions/ # Patient understanding evaluation config
│   ├── conversations/         # Synthetic conversation logs
│   ├── evaluation_results/    # Comparison metrics and analysis
│   ├── ground_truth.json      # Gold standard annotations
│   ├── mandatory_questions.json # Required medical questions
│   ├── processed/             # Extraction results organized by model
│   ├── raw_md_files/          # Input: 6 German obstetrics consent documents
│   └── vector_store/          # Chroma/FAISS indexes for RAG
├── documentation/             # Detailed system prompts and documentation
│   ├── DIALOGUE_MANAGER_...   # Dialogue policy and scaffolding logic
│   ├── DOCTOR_AGENT_...       # Doctor persona instructions
│   ├── PATIENT_AGENT_...      # Patient multi-persona generation logic
│   └── evaluation_metrics_taxonomy.md # Overview of evaluation frameworks
├── evaluation/                # Automated evaluation scripts
│   ├── evaluate_citation.py   # Checks LLM citation accuracy
│   ├── evaluate_coverage.py   # Full coverage evaluation pipeline
│   ├── evaluate_mandatory_q.py# Verifies mandatory questions
│   └── ...                    # Other metric calculation scripts
├── information_extraction/    # Extraction method implementations
│   ├── naive_llm.py           # Baseline: Direct LLM prompting
│   ├── atomic_fact_extraction.py # Two-stage extraction with facts
│   ├── cot_extraction.py      # Chain-of-Thought paradigm 
│   └── uie.py                 # Schema-guided structural extraction
├── scripts/                   # Workflow and execution scripts
│   ├── generate_conversations.py # Synthetic dataset generation
│   ├── plot_coverage_results.py  # Visualizations
│   ├── run_extraction.py      # Batch extraction runner
│   └── test_doctor_patient.py # Interaction testing
└── utils/                     # Utilities and helpers
    ├── compare_results.py     # Analysis and comparison utilities
    └── llm_config.py          # Model configurations
```

## Setup

### Requirements
- Python 3.8+
- Local LLM server (e.g., LM Studio) running on `http://127.0.0.1:1234` or API keys for remote models.
- Packages: `requests`, and other ML specific tools. Install requirements via `pip install -r requirements.txt`.

### Configuration

Edit `utils/llm_config.py` to add/modify models configured for information extraction or chat completions.

## Extraction Methods

### 1. Naive LLM (`naive`)
- **Approach**: Direct prompting for discussion points.
- **Output**: `{point, rationale}` pairs.
- **Use case**: Baseline for comparison.

### 2. Atomic Fact Extraction (`atomic`)
- **Approach**: Two-stage extraction. 
  - Stage 1: Break text into atomic facts (limits to essential fragments).
  - Stage 2: Synthesize into final discussion points.
- **Use case**: High granularity with synthesis.

### 3. Chain-of-Thought (`cot`)
- **Approach**: Forces reasoning before output generation.
- **Output**: Thinking process + JSON.
- **Use case**: Explainable extraction with implicit reasoning trace.

### 4. Schema-Guided (`uie`)
- **Approach**: Type-constrained with predefined categorization schema (RISK, INSTRUCTION, PREREQUISITE, GENERAL_INFO).
- **Use case**: Structured, typed extraction prioritizing clinical risks.

## Usage

### 1. Run Extraction Pipeline

Process all documents with all methods and models:
```bash
python scripts/run_extraction.py
```
This processes all 6 documents in `data/raw_md_files/`, applies all 4 extraction methods, tests all configured models based on `llm_config.py`, and saves results to `data/processed/{model_name}/`.

### 2. Analyze Results

Compare extraction methods across models:
```bash
python utils/compare_results.py
```
This loads all extraction results, prints summary statistics, and can be used to export a comparison CSV to `data/evaluation_results/`.

### 3. Generate and Evaluate Conversations
Generate synthetic conversations by running the agents via:
```bash
python scripts/generate_conversations.py
```

Run evaluation scripts (coverage, citation, mandatory questions) from the `evaluation/` directory:
```bash
python evaluation/evaluate_coverage.py
python evaluation/evaluate_citation.py
python evaluation/evaluate_mandatory_q.py
```

## Input Documents
The framework uses six German obstetrics consent documents outlining prenatal care rules:
1. **Äußere Wendung** - External cephalic version
2. **Geburtseinleitung** - Labor induction
3. **Geburtshilfliche Maßnahmen** - Obstetric procedures
4. **Kaiserschnitt** - Cesarean section
5. **Narkose** - Anesthesia
6. **DRK Geburtshilfe Infos** - Birthing facility information

## Project Phases

### Phase 1: Information Extraction
- Implemented 4 extraction methods (Naive, Auto-Fact, CoT, UIE).
- Processed 6 obstetric documents & evaluated against standard human-annotated `ground_truth.json`.

### Phase 2: Evaluation System
- Developed Coverage algorithms (SBERT, Hungarian matching, LLM-as-a-Judge, and SUSWIR metric variation).
- Designed citation verification algorithms using exact and relaxed string checks.
- Formulated mandatory queries validation.

### Phase 3: Synthetic Patients & Conversations
- Patient agent dynamically powered by multiple distinct clinical personas.
- Built interactive testing protocols bridging Patient module with Chatbot systems.

### Phase 4: Dialogue Management
- Supervised constraints enforcing mandatory questioning logic seamlessly injected into discussions.
- Turn-based scaffolding enhancing compliance to 100% completion in strict checks.

### Phase 5: Chatbot Implementation
- Fully functional Retrieval-Augmented Generation (RAG) agent (`doctor_agent.py`, `retrieval.py`).
- Integrated chunking mechanisms ensuring the chatbot uses medical documentation practically.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
