# Dialogue Manager Documentation

## 3.4 Dialogue State Management

The Dialogue Manager is the central orchestrator of the conversation between the patient and doctor agents. Two configurations of the chatbot are compared in this system: a naive baseline with no structural enforcement, and a supervised version with explicit mandatory question tracking.

### 3.4.1 Naive Dialogue Management Baseline

In the passive or naive mode, the doctor agent operates solely based on its static system prompt, the retrieved medical context, and the conversation history.
- No external state is actively tracked during the conversation.
- There is no dynamic enforcement of mandatory questions; they are simply appended to the system prompt as an "ideal" checklist (e.g., "Folgende Pflichtfragen sollten idealerweise im Gespräch abgedeckt werden...").
- This serves as a meaningful baseline because it reflects a typical, standard LLM chatbot deployment where the model is expected to handle conversational goals autonomously without external scaffolding.

### 3.4.2 Supervised Dialogue Management

In the active mode, the Dialogue Manager acts as a supervisor, actively tracking the conversational state and intervening when necessary.

**Architecture of the State Tracker**
- Mandatory questions for specific medical procedures (e.g., "Narkose", "Kaiserschnitt") are defined in an external JSON file (`data/mandatory_questions.json`) and loaded at initialization.
- The manager maintains a `conversation_history`, the current `turn_count`, and a `mandatory_asked` set that stores the IDs of questions that have already been triggered and asked.

**Detection Mechanism**
- The system determines whether to intervene and inject a mandatory question using an embedding-based semantic check.
- It uses the `paraphrase-multilingual-mpnet-base-v2` model from `sentence-transformers` to encode both the pending mandatory questions and the patient's latest utterance.
- If the cosine similarity between the patient's question and any pending mandatory question exceeds a defined threshold (`similarity_threshold = 0.55`), a match is detected.

**Intervention Logic**
When an intervention is triggered, it happens in one of two ways:
1. **Contextual Intervention:** If a strong semantic match is found between the patient's input and a pending question, the system seamlessly injects the best-matching mandatory question so that the doctor asks it naturally in the context of the current topic.
2. **Safety Net Intervention:** If the conversation is nearing its maximum limit (`max_turns - turn_count <= len(pending_questions)`), the manager forces the remaining mandatory questions to ensure they are covered before the conversation inevitably ends.

When a mandatory question is selected, the manager explicitly instructs the doctor agent by injecting a dynamic trigger prompt (as defined in `DIALOGUE_MANAGER_PROMPTS.md`):
`"Bitte stelle die folgende Pflichtfrage auf natürliche Weise im Verlauf deiner Antwort: {specific_mandatory_question}"`

**Wrapping the Doctor Agent**
This supervised management wraps around the doctor agent as an external layer. It intercepts the user (patient) input and dynamically manipulates the doctor's system instructions (`extra_system_instructions`) for that specific turn. The core response generation and retrieval logic of the doctor agent remains unchanged; it is simply guided at runtime to fulfill necessary procedural requirements seamlessly.