# Dialogue Manager Prompts

This file contains dynamic trigger prompts used by the dialogue manager when invoking the doctor and patient agents. These prompts are NOT part of static system prompts; they are injected as `user` or `assistant` messages during runtime to guide question generation and interventions.

## Patient Agent Triggers

### Initial Question Trigger

```
Du startest gerade das Gespräch mit dem medizinischen Assistenten über das Thema: '[PROCEDURE]'.
Stelle deine erste Frage. Sei direkt, chat-typisch und deinem Charakter entsprechend.
Beispiel: 'ok, wie läuft die [PROCEDURE] genau ab?'
```

### Follow-Up Trigger — When Doctor Asked a Question

```
Gesprächsphase: [PHASE]. Angst=[X], Vertrauen=[Y], Klarheit=[Z].
Antworte zuerst direkt auf die Frage des Chatbots.
Falls noch Unsicherheit bleibt, stelle eine kurze Anschlussfrage über DEINE Situation (ich/mir/meine).
Frage nicht nach dem Gesundheitszustand des Chatbots.
```

### Follow-Up Trigger — When Doctor Did Not Ask a Question

```
Gesprächsphase: [PHASE]. Angst=[X], Vertrauen=[Y], Klarheit=[Z].
Reagiere natürlich auf die letzte Information.
Stelle eine neue, nicht wiederholte und inhaltlich passende Folgefrage oder äußere eine Sorge zu deiner eigenen Situation.
```

### Hidden-Fact Disclosure Addendum

When the dialogue manager decides to allow hidden-fact disclosure, append this sentence to the patient trigger:

```
Integriere jetzt diese persönliche Info unauffällig in deine Antwort: '[HIDDEN_FACT]'.
```

## Doctor Agent Triggers

### Mandatory Question Injection (Active Mode with Intervention)

```
Bitte stelle die folgende Pflichtfrage auf natürliche Weise im Verlauf deiner Antwort:
{specific_mandatory_question}
```

### Retrieval/Context Injection

When calling the doctor agent, the dialogue manager injects the `MEDIZINISCHER KONTEXT` block as described in the doctor system prompt. This is NOT part of the static system prompt file; it is added per response.

## Usage Notes

- Keep triggers short and deterministic where possible.
- Only the Dialogue Manager should use this file; system prompts remain static in their respective files.
