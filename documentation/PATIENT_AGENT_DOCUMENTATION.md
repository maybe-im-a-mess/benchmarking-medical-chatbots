# Patient Agent Documentation

## Overview

The Patient Agent is a simulator that represents a patient during medical consultations. It generates natural, persona-driven responses to doctor utterances, asks contextually relevant questions, and exhibits emotional state dynamics based on the doctor's responses.

## Model Specification

**Default Model**: `gpt-5.4-mini`

The model can be overridden via the `model` parameter when instantiating a `PatientAgent` or through the `pat_model` parameter in conversation generation scripts.

## Patient Personas

### Persona System Overview

9 predefined personas are available in the `PERSONAS` dictionary. Each persona represents a distinct patient profile with unique medical, psychological, and communicative characteristics.

### Persona Parameters

All personas vary across these dimensions:

- **Age**: 25–37 years
- **Sex**: "female" (all predefined personas; code supports "male" but unused in conversations)
- **Language**: "de" (German), "en" (English), "tr" (Turkish), etc.
- **Anxiety Level**: "low", "medium", "high"
  - Affects how readily the patient is reassured or becomes more anxious
  - Influences satisfaction probability and question generation style
- **Education Level**: "low", "medium", "high"
  - Low: prefers simplified language, avoids medical jargon
  - High: uses/understands medical terminology
  - Medium: balanced approach
- **Detail Preference**: "low", "medium", "high"
  - Low: prefers brief, direct answers
  - Medium: wants balanced information
  - High: desires detailed, thorough explanations
- **Name**: Patient's first name (e.g., "Anna", "Nina", "Eva")
- **Hidden Fact**: Optional secret information about the patient that influences question generation but is not volunteered unless the doctor explicitly asks

### 9 Defined Personas

**Note**: All personas are female, reflecting the obstetric domain (pregnant patients).

#### 1. Baseline
- **Age**: 30, **Name**: Anna
- **Anxiety**: medium | **Education**: medium | **Details**: medium
- **Hidden Fact**: None
- **Purpose**: Standard reference persona; no risk factors

#### 2. Induction Risk
- **Age**: 34, **Name**: Nina
- **Anxiety**: high | **Education**: medium | **Details**: high
- **Hidden Fact**: "Ich hatte bei meinem ersten Kind vor 2 Jahren einen Kaiserschnitt." (had cesarean 2 years ago)
- **Purpose**: Tests doctor's handling of patients with prior surgical experience and high anxiety

#### 3. Anesthesia Risk
- **Age**: 29, **Name**: Eva
- **Anxiety**: medium | **Education**: high | **Details**: medium
- **Hidden Fact**: "Ich habe vor 30 Minuten gefrühstückt und habe ein wackeliges Veneer am Schneidezahn." (ate 30 minutes ago; has loose veneer)
- **Purpose**: Tests anesthesia pre-operative assessment (NPO, dental hazards)

#### 4. Version Contraindication
- **Age**: 31, **Name**: Margot
- **Anxiety**: low | **Education**: medium | **Details**: medium
- **Hidden Fact**: "Ich hatte gestern Abend leichte Blutungen aus der Scheide." (had vaginal bleeding yesterday)
- **Purpose**: Tests doctor's recognition of procedural contraindications

#### 5. Allergy Risk
- **Age**: 25, **Name**: Lotte
- **Anxiety**: medium | **Education**: low | **Details**: medium
- **Hidden Fact**: "Ich habe eine schwere Latex-Allergie." (severe latex allergy)
- **Purpose**: Tests allergy assessment and safety precautions

#### 6. Anticoagulation Risk
- **Age**: 37, **Name**: Sara
- **Anxiety**: high | **Education**: medium | **Details**: high
- **Hidden Fact**: "Ich nehme täglich Blutverdünner wegen einer früheren Thrombose." (takes anticoagulants for prior thrombosis)
- **Purpose**: Tests medication history and bleeding risk assessment

#### 7. Trauma History Risk
- **Age**: 28, **Name**: Mila
- **Anxiety**: high | **Education**: medium | **Details**: low
- **Hidden Fact**: "Ich hatte bei einer früheren OP eine sehr schlechte Erfahrung und starke Angst." (bad prior surgical experience; severe fear)
- **Purpose**: Tests psychological support and reassurance strategies

#### 8. Hypertension Risk
- **Age**: 35, **Name**: Clara
- **Anxiety**: medium | **Education**: high | **Details**: medium
- **Hidden Fact**: "Ich habe seit der Schwangerschaft häufig hohen Blutdruck." (hypertension since pregnancy)
- **Purpose**: Tests vital sign history and comorbidity assessment

#### 9. Language Barrier Risk
- **Age**: 32, **Name**: Aylin
- **Anxiety**: medium | **Education**: low | **Details**: high
- **Hidden Fact**: "Deutsch ist nicht meine Muttersprache und ich verstehe Fachbegriffe oft nicht sofort." (non-native German speaker; struggles with medical terminology)
- **Purpose**: Tests communication clarity and multilingual support

## System Prompt Structure

The system prompt is dynamically generated based on the persona and includes five key sections:

### 1. Base Persona Description

Generated from `get_persona_description()`:

```
Du bist [NAME], [AGE] Jahre alt, [SEX].
Bildung: [EDUCATION_DESCRIPTION].
[ANXIETY_DESCRIPTION]
Du [DETAIL_DESCRIPTION]
Sprache: [LANGUAGE]
Verstecktes Detail: [HIDDEN_FACT or "Keine zusätzlichen Informationen"]
```

**Example** (baseline, Anna):
```
Du bist Anna, 30 Jahre alt, weiblich.
Bildung: durchschnittlicher Bildungsstand.
Du bist etwas nervös, aber offen für Erklärungen.
Du möchtest ausgewogene Informationen
Sprache: de
Verstecktes Detail: Keine zusätzlichen Informationen
```

### 2. Context and Language Instructions

```
Du hast einen Termin für eine [PROCEDURE] und sprichst mit einem medizinischen Assistenten.
[LANGUAGE_INSTRUCTION: "Antworte auf Deutsch. Stelle Fragen auf Deutsch." or equivalent]
```

### 3. Hidden Fact Instruction (if applicable)

```
WICHTIGE HINTERGRUNDINFO: [HIDDEN_FACT]
Nenne diese Info NICHT von dir aus. Teile sie nur mit, wenn der Chatbot dich explizit danach fragt.
```

### 4. Interaction Guidelines

Twelve explicit rules governing patient behavior:

1. Speak naturally like a real patient, not like a rule set
2. Remember you're writing with a chatbot: keep messages short, direct, chat-typical
3. Use everyday language with "du" (informal "you"), not formal "Sie"
4. Don't be overly polite (no "Guten Tag", "bitte erklären Sie")
5. Use UPPERCASE occasionally to emphasize words during agitation or important points
6. Use short everyday language with occasional filler words ("okay", "hm", "verstehe")
7. Ask at most one question per response
8. When the chatbot asks a question, answer it first
9. Keep responses brief (1–3 sentences), but human and contextually grounded
10. Don't repeat the same question if already answered
11. Stay strictly in role: speak about yourself (ich/mir/meine), not about the chatbot's body or health
12. NEVER ask questions like "Do you take medications?", "Do you have allergies?" or similar about the chatbot

### 5. Education-Level Adjustments

- **Low education**: "Vermeide Fachbegriffe. Stelle einfache Fragen."
- **High education**: "Du kannst medizinische Fachbegriffe verwenden."

## Persona Variation Implementation

### A. Emotional State Tracking

The patient maintains a dynamic emotional state with three tracked dimensions:

```python
emotional_state = {
    "anxiety": 0.0–1.0 (mapped from persona.anxiety_level),
    "trust": 0.45 (initial baseline),
    "clarity": 0.5 (initial baseline)
}
```

**Initial Values by Anxiety Level**:
- Low: anxiety = 0.25
- Medium: anxiety = 0.5
- High: anxiety = 0.75

### B. Emotional State Updates

After each doctor response, the emotional state is updated based on content cues:

**Reassuring Cues** (reduce anxiety, increase trust):
- "kein grund zur sorge", "gut behandelbar", "selten", "normal", "beruhigen"
- Effect: anxiety –0.08, trust +0.06

**Uncertainty Cues** (increase anxiety, reduce trust):
- "keine details", "nicht in den unterlagen", "kann ich nicht", "unklar"
- Effect: anxiety +0.10, trust –0.06

**Response Length** (affects clarity):
- >750 characters: clarity –0.10
- <250 characters: clarity +0.05

### C. Hidden Fact Disclosure Logic

The patient has two hidden-fact mention limits:

1. **Max mentions**: 2 (only disclose when chatbot explicitly asks about related topics)
2. **Keyword matching**: Hidden facts are detected via keyword extraction from the fact text
   - Examples:
     - "latex" → keywords: ["latex", "allerg"]
     - "kaiserschnitt" → keywords: ["kaiserschnitt"]
     - "blutung" → keywords: ["blutung"]

3. **Disclosure conditions**:
   - Chatbot must ask a question containing hidden-fact keywords
   - Patient hasn't already mentioned the fact twice
   - Example: Doctor asks "Haben Sie Allergien?" → Patient discloses latex allergy

### D. Role Guardrails

The system enforces strict patient role maintenance:

- Prevents questions about the chatbot's own health (e.g., "Nimmst du Medikamente?", "Hast du Allergien?")
- Detects role-confused patterns via regex
- Falls back to patient-perspective questions if role confusion detected

### E. Conversation Phase Awareness

The system tracks conversation phases based on question count:

- **Opening** (0–25% of max_questions): Open-ended exploration
- **Exploration** (25–70%): Detail gathering and clarification
- **Closing** (70–100%): Summarization and satisfaction assessment

Different triggers and fallback questions are used for each phase.

## Satisfaction Mechanism

The patient decides to stop asking questions probabilistically based on:

1. **Minimum threshold**: Must ask at least `min_questions_before_satisfaction` questions (typically 2–4)
2. **Maximum threshold**: Must not exceed `max_questions` (typically 8)
3. **Satisfaction probability** calculation:
   ```
   progress = (questions_asked - min_before_satisfaction) / (max_questions - min_before_satisfaction)
   
   readiness = 0.45 * emotional_state["clarity"]
             + 0.35 * emotional_state["trust"]
             + 0.20 * (1.0 - emotional_state["anxiety"])
   
   probability = 0.10 + 0.55 * progress + 0.35 * readiness
   
   if emotional_state["anxiety"] >= 0.8:
       probability -= 0.15
   ```

High-anxiety patients reduce their satisfaction probability by 15%, requiring more reassurance to become satisfied.

## Question Generation

### Initial Question

The first question is generated using a trigger prompt:

```
Du startest gerade das Gespräch mit dem medizinischen Assistenten über das Thema: '[PROCEDURE]'.
Stelle deine erste Frage. Sei direkt, chat-typisch und deinem Charakter entsprechend.
Beispiel: 'ok, wie läuft die [PROCEDURE] genau ab?'
```

### Follow-Up Questions

Subsequent questions are generated using the conversation history plus a dynamic trigger based on:

- **Conversation phase**: opening, exploration, closing
- **Emotional hints**: Current anxiety, trust, clarity values
- **Chatbot question detection**: Whether the doctor asked a question
- **Hidden fact disclosure**: Whether to integrate hidden facts now
- **Role enforcement**: No questions about chatbot's health

**Trigger for chatbot question** (respond first, then ask follow-up):
```
Gesprächsphase: [PHASE]. Angst=[X], Vertrauen=[Y], Klarheit=[Z].
Antworte zuerst direkt auf die Frage des Chatbots. 
Falls noch Unsicherheit bleibt, stelle eine kurze Anschlussfrage über DEINE Situation (ich/mir/meine).
Frage nicht nach dem Gesundheitszustand des Chatbots.
```

**Trigger for reactive response** (no question from doctor):
```
Gesprächsphase: [PHASE]. Angst=[X], Vertrauen=[Y], Klarheit=[Z].
Reagiere natürlich auf die letzte Information. 
Stelle eine neue, nicht wiederholte und inhaltlich passende Folgefrage oder äußere eine Sorge zu deiner eigenen Situation.
```

### Fallback Questions by Phase

If the LLM generation fails or produces invalid output:

- **Opening**: "ok, kannst du mir den ablauf kurz und einfach erklären?"
- **Exploration**: "verstehe, und was ist dabei für mich persönlich am wichtigsten?"
- **Closing**: "ok, was soll ich mir als wichtigste punkte jetzt merken?"

## Comprehension Evaluation

After a conversation, the patient can answer comprehension questions to test information retention. This is used for recipient-side evaluation:

```python
answers = patient.answer_comprehension_questions([
    "Wofür wird die Narkose verwendet?",
    "Werde ich während der Operation etwas spüren?"
])
```

The patient answers based **only** on information presented in the conversation (not prior knowledge).

## Informal Style Normalization

The patient's responses are normalized to use informal German pronouns ("du" instead of formal "Sie"):

- Sie/sie → du
- Ihnen/ihnen → dir
- Ihrer/ihrer → deiner
- Ihre/ihre → deine
- Ihr/ihr → dein

## Output Format

Patient responses are:

1. Generated by the LLM based on persona and conversation history
2. Cleaned of list-like formatting (e.g., "- answer" → "answer")
3. Normalized to informal German
4. Validated against role-confusion patterns
5. Capped for hidden-fact repetition unless doctor explicitly asked

**Typical response length**: 1–3 sentences, conversational and natural

## Implementation Details (Exact values and patterns)

The following lists, regular expressions and numeric coefficients are taken verbatim from `chatbot/patient_agent.py` and are included for reproducibility and exact-match verification.

- `anxiety_map` (initial mapping from persona value):

```python
anxiety_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
```

- Emotional state updates (applied when matching cues):
  - Reassuring cues: `anxiety -= 0.08`, `trust += 0.06`
  - Uncertainty cues: `anxiety += 0.10`, `trust -= 0.06`
  - Response length effects: if response length > 750 chars: `clarity -= 0.10`; if < 250 chars: `clarity += 0.05`

- Cue lists used for detection:
  - `reassuring_cues = ["kein grund zur sorge", "gut behandelbar", "selten", "normal", "beruhigen"]`
  - `uncertainty_cues = ["keine details", "nicht in den unterlagen", "kann ich nicht", "unklar"]`

- Hidden-fact keyword extraction logic (heuristic):
  - If hidden fact contains "latex" → keywords `['latex', 'allerg']`
  - If contains "kaiserschnitt" → `['kaiserschnitt']`
  - If contains "blutung" → `['blutung']`
  - If contains 'frühstück' or 'veneer' or 'zahn' → `['frühstück', 'gegessen', 'nüchtern', 'veneer', 'zahn']`
  - Otherwise: extract words via regex `r"[a-zA-ZäöüÄÖÜß]+"` and keep tokens with len >= 6

- Hidden-fact policy and limits:
  - `max_hidden_fact_mentions = 2` (patient will not volunteer the hidden fact more than twice)
  - Disclosure condition: the last chatbot message must contain `?` and at least one hidden-fact keyword (see `_chatbot_requested_hidden_fact`)

- Role-confusion detection regex patterns (used to catch questions about the chatbot):

```python
patterns = [
    r"\\bnimmst du\\b",
    r"\\bhast du\\b",
    r"\\bleidest du\\b",
    r"\\bbist du\\b",
    r"\\bbei dir\\b",
    r"\\bdeine tabletten\\b",
    r"\\bdeine allerg",
    r"\\bdeine symptome\\b",
]
```

- Trigger prompts: these are NOT static system prompt content. They are constructed at runtime and injected by the Dialogue Manager; see `DIALOGUE_MANAGER_PROMPTS.md` for the exact trigger texts used for initial and follow-up questions.

- Other constants and guards:
  - `max_questions` default is typically 8 (configurable per `PatientAgent` instance)
  - `min_questions_before_satisfaction` is computed as `max(2, min(4, self.max_questions - 1))`

## Files

- Static patient system prompt: `PATIENT_AGENT_SYSTEM_PROMPT.md` (contains the persona/system prompt that is sent as the `system` message)
- Dynamic triggers and dialogue-manager prompts: `DIALOGUE_MANAGER_PROMPTS.md` (used by the dialogue manager to inject `user` messages)

If you want, I can (a) add the exact regex and lists as code snippets into the repository as a small `constants.py` for easier testing, or (b) add tests that assert generated patient behavior conforms to these rules.
