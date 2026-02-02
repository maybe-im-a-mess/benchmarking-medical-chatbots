# System Architecture Diagram

## Overall Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THESIS PROJECT PHASES                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 1: Information Extraction (COMPLETED)                         │
│  ────────────────────────────────────────────                        │
│                                                                       │
│  PDF Documents → Markdown → [4 Extraction Methods]                   │
│                              ↓                                        │
│                        Extracted Statements                           │
│                        (Ground Truth)                                 │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 2: Conversational System (IMPLEMENTED)                        │
│  ─────────────────────────────────────────────                       │
│                                                                       │
│  ┌────────────┐         ┌─────────────┐         ┌──────────────┐   │
│  │  Patient   │ ──────→ │   Doctor    │ ──────→ │  Dialogue    │   │
│  │   Agent    │ ←────── │    Agent    │ ←────── │   Manager    │   │
│  │            │         │   (RAG)     │         │              │   │
│  └────────────┘         └─────────────┘         └──────────────┘   │
│        │                       │                        │            │
│        │                       ↓                        │            │
│        │              ┌─────────────────┐              │            │
│        │              │  Embeddings +   │              │            │
│        │              │    Retrieval    │              │            │
│        │              └─────────────────┘              │            │
│        │                       │                        │            │
│        └───────────────────────┴────────────────────────┘           │
│                                 ↓                                     │
│                          Conversation Log                            │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 3: Evaluation (READY)                                         │
│  ────────────────────────────                                        │
│                                                                       │
│  Conversations + Ground Truth → [Evaluation Metrics]                 │
│                                        ↓                              │
│                              ┌───────────────────┐                   │
│                              │ • Sender Coverage │                   │
│                              │ • Citation Acc.   │                   │
│                              │ • Conciseness     │                   │
│                              │ • Comprehension   │                   │
│                              └───────────────────┘                   │
│                                        ↓                              │
│                                  Thesis Results                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Doctor Agent (RAG Pipeline)

```
                    Patient Question
                           ↓
                  ┌────────────────┐
                  │ Query Embedding│
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │   Document     │
                  │   Retrieval    │ ←── InMemoryDocumentStore
                  │   (Top-K=5)    │     (6 Medical Documents)
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │  Retrieved     │
                  │   Chunks       │ (with citations)
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │ Prompt Builder │
                  │   + System     │
                  │   Message      │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │ LLM Generator  │ (GPT-4o-mini)
                  │  (Temperature  │
                  │     0.3)       │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │    Answer      │ + Citations
                  └────────────────┘
```

## Patient Agent (Simulator)

```
                     Procedure Name
                           ↓
                  ┌────────────────┐
                  │   Question     │
                  │  Generation    │ ←── Previous Answers
                  │   (LLM Call)   │     (Context)
                  └────────┬───────┘
                           ↓
                     Patient Question
                           ↓
                  ┌────────────────┐
                  │ Doctor Agent   │
                  │   Responds     │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │   Receive      │
                  │  Information   │
                  └────────┬───────┘
                           ↓
                  ┌────────────────┐
                  │ Comprehension  │
                  │   Evaluation   │ ←── Test Questions
                  │   (LLM Call)   │
                  └────────┬───────┘
                           ↓
                  Understanding Score
```

## Dialogue Manager (State Machine)

```
     Conversation Start
            ↓
    ┌──────────────┐
    │ Initialize   │
    │    State     │
    └──────┬───────┘
           ↓
    ┌──────────────────────────────────────┐
    │  Turn Loop (max 10 turns)            │
    │                                       │
    │  1. Increment turn counter            │
    │  2. Check: Should ask mandatory Q?    │
    │     ├─Yes→ Ask mandatory question     │
    │     └─No → Continue information       │
    │                exchange                │
    │  3. Record topic covered              │
    │  4. Update state                      │
    │                                       │
    │  Exit condition:                      │
    │  • All mandatory Q's asked            │
    │  • Min 5 turns                        │
    │  • Min 2 topics covered               │
    └──────┬───────────────────────────────┘
           ↓
    ┌──────────────┐
    │  Conclusion  │
    │   + Consent  │
    └──────────────┘
```

## Evaluation Pipeline

```
    Synthetic Conversations
            ↓
    ┌──────────────────────────────────┐
    │  Load Ground Truth               │
    │  (Extracted Statements)          │
    └──────┬───────────────────────────┘
           ↓
    ┌──────────────────────────────────┐
    │  Calculate Metrics               │
    │                                   │
    │  1. Sender Coverage:              │
    │     Match statements → responses  │
    │                                   │
    │  2. Citation Accuracy:            │
    │     Analyze retrieval stats       │
    │                                   │
    │  3. Conciseness:                  │
    │     Count words/characters        │
    │                                   │
    │  4. Comprehension:                │
    │     Patient agent answers Q's     │
    └──────┬───────────────────────────┘
           ↓
    ┌──────────────────────────────────┐
    │  Aggregate Results               │
    │  • Per conversation              │
    │  • Per procedure                  │
    │  • Overall averages               │
    └──────┬───────────────────────────┘
           ↓
      JSON + Analysis
```

## Data Flow

```
Offline Preparation:
───────────────────
data/raw_md_files/*.md
        ↓
[Document Processor]
        ↓
embeddings + chunks
        ↓
InMemoryDocumentStore
        ↓
(Ready for retrieval)


Runtime Conversation:
────────────────────
Patient Question
        ↓
[Retrieval] → Retrieved Chunks
        ↓
[LLM Generator] → Doctor Answer
        ↓
Conversation Log
        ↓
data/conversations/*.json


Evaluation:
──────────
Conversation Log + Ground Truth
        ↓
[Metric Calculators]
        ↓
Evaluation Results
        ↓
data/conversations/*_evaluation.json
```

## Technology Stack

```
┌─────────────────────────────────────────┐
│          Application Layer              │
├─────────────────────────────────────────┤
│  • Doctor Agent (RAG)                   │
│  • Patient Agent (Simulator)            │
│  • Dialogue Manager (State)             │
│  • Evaluation Scripts                   │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│         Framework Layer                 │
├─────────────────────────────────────────┤
│  Haystack: RAG pipeline                 │
│  sentence-transformers: Embeddings      │
│  OpenAI: LLM generation                 │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│           Data Layer                    │
├─────────────────────────────────────────┤
│  • Markdown documents (6)               │
│  • Extracted statements (JSON)          │
│  • Document embeddings (vectors)        │
│  • Conversation logs (JSON)             │
│  • Evaluation results (JSON/CSV)        │
└─────────────────────────────────────────┘
```

## File Organization

```
code/
├── chatbot/              ← NEW: Conversational components
│   ├── doctor_agent.py       (RAG pipeline)
│   ├── patient_agent.py      (Simulator)
│   ├── dialogue_manager.py   (State machine)
│   ├── embeddings.py         (Document processing)
│   └── retrieval.py          (Citation retrieval)
│
├── scripts/              ← NEW: Executable workflows
│   ├── run_chatbot.py        (Interactive demo)
│   ├── generate_conversations.py
│   ├── evaluate_conversations.py
│   └── [extraction scripts moved here]
│
├── information_extraction/   ← Existing: 4 methods
│   ├── naive_llm.py
│   ├── atomic_fact_extraction.py
│   ├── cot_extraction.py
│   └── uie.py
│
├── utils/                ← Updated: Config + analysis
│   ├── llm_config.py     (Added Haystack support)
│   └── compare_results.py
│
└── data/
    ├── raw_md_files/     ← 6 consent documents
    ├── processed/        ← Extraction results
    ├── vector_store/     ← NEW: Embeddings
    └── conversations/    ← NEW: Dialogues + evaluations
```
