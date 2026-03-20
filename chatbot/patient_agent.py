from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Optional


load_dotenv()

class PatientPersona:
    """
    Defines a persona for the patient agent.
    """
    def __init__(self,
                 age: int,
                 sex: str,
                 language: str = "de",
                 anxiety_level: str = "medium",
                 education_level: str = "medium",
                 detail_preference: str = "medium",
                 name: str = "Himeno",
                 hidden_fact: Optional[str] = None):
        """
        Parameters:
            :param age: Patient age
            :param sex: 'male' or 'female'
            :param language: 'de' (German), 'en' (English), 'tr' (Turkish), etc.
            :param anxiety_level: 'low', 'medium', 'high' - how anxious the patient is about medical procedures
            :param education_level: 'low', 'medium', 'high'
            :param detail_preference: 'low' (brief answers), 'medium' (default), 'high' (wants details)
            :param name: Patient name
            :param hidden_fact: An optional secret fact about the patient that is not revealed to the chatbot but can influence their questions (e.g., "has a family history of anesthesia complications")
        """
        self.age = age
        self.sex = sex
        self.language = language
        self.anxiety_level = anxiety_level
        self.education_level = education_level
        self.detail_preference = detail_preference
        self.name = name or self._generate_name()
        self.hidden_fact = hidden_fact

    def _generate_name(self) -> str:
        """Return a stable fallback name when none is provided."""
        if self.sex == "male":
            return "Max"
        return "Anna"
    
    def get_persona_description(self) -> str:
        """Return a textual description of the patient's persona."""
        sex_de = "männlich" if self.sex == "male" else "weiblich"
        anxiety_map = {
            "low": "Du bist entspannt und vertraust dem medizinischen Personal.",
            "medium": "Du bist etwas nervös, aber offen für Erklärungen.",
            "high": "Du bist ängstlich und brauchst viel Beruhigung."
        }
        
        education_map = {
            "low": "einfacher Bildungsstand, sprichst einfaches Deutsch",
            "medium": "durchschnittlicher Bildungsstand",
            "high": "hoher Bildungsstand, verstehst medizinische Fachbegriffe"
        }
        
        detail_map = {
            "low": "möchtest kurze, direkte Antworten",
            "medium": "möchtest ausgewogene Informationen",
            "high": "möchtest detaillierte, gründliche Erklärungen"
        }
        
        return f"""Du bist {self.name}, {self.age} Jahre alt, {sex_de}.
Bildung: {education_map[self.education_level]}.
{anxiety_map[self.anxiety_level]}
Du {detail_map[self.detail_preference]}
Sprache: {self.language}
Verstecktes Detail: {self.hidden_fact if self.hidden_fact else 'Keine zusätzlichen Informationen'}"""


class PatientAgent:
    """
    Simulates a patient asking questions about a medical procedure.
    """
    PERSONAS = {
        "baseline": PatientPersona(
            age=30, sex="female", language="de", 
            education_level="medium", detail_preference="medium",
            anxiety_level="medium", hidden_fact=None, name="Anna"
        ),
        "induction_risk": PatientPersona(
            age=34, sex="female", language="de", anxiety_level="high",
            education_level="medium", detail_preference="high",
            hidden_fact="Ich hatte bei meinem ersten Kind vor 2 Jahren einen Kaiserschnitt.",
            name="Nina"
        ),
        "anesthesia_risk": PatientPersona(
            age=29, sex="female", language="de", anxiety_level="medium",
            education_level="high", detail_preference="medium",
            hidden_fact="Ich habe vor 30 Minuten gefrühstückt und habe ein wackeliges Veneer am Schneidezahn.",
            name="Eva"
        ),
        "version_contraindication": PatientPersona(
            age=31, sex="female", language="de", anxiety_level="low",
            education_level="medium", detail_preference="medium",
            hidden_fact="Ich hatte gestern Abend leichte Blutungen aus der Scheide.",
            name="Margot"
        ),
        "allergy_risk": PatientPersona(
            age=25, sex="female", language="de", anxiety_level="medium",
            education_level="low", detail_preference="medium",
            hidden_fact="Ich habe eine schwere Latex-Allergie.",
            name="Lotte"
        ),
    }
    
    def __init__(self,
                procedure_name: str = "Narkose",
                persona: Optional[PatientPersona] = None,
                persona_type: Optional[str] = None,
                model: str = "gpt-5-mini",
                max_questions: int = 8,
                temperature: float = 0.8):
        """
        Parameters:
            :param procedure_name: A procedure the patient agent is discussing
            :param persona: PatientPersona object; if None, a random persona is chosen
            :param persona_type: Optional type of predefined persona to use from PERSONAS
            :param model: OpenAI model
            :param max_questions: Maximum number of questions the patient will ask
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.procedure_name = procedure_name
        self.max_questions = max_questions
        self.temperature = temperature

        # Set persona
        if persona:
            self.persona = persona
        elif persona_type and persona_type in self.PERSONAS:
            self.persona = self.PERSONAS[persona_type]
        else:
            # Default persona
            self.persona = PatientPersona(age=27, sex="male", language="de", education_level="high", detail_preference="medium", name="Johan")

        # Conversation state
        self.conversation_history = []
        self.questions_asked = 0
        self.hidden_fact_disclosed = False

        anxiety_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
        self.emotional_state = {
            "anxiety": anxiety_map.get(self.persona.anxiety_level, 0.5),
            "trust": 0.45,
            "clarity": 0.5,
        }

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _contains_question(self, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        question_starters = [
            "haben sie", "sind sie", "nehmen sie", "kennen sie", "wissen sie",
            "können sie", "möchten sie", "wie", "was", "wann", "warum"
        ]
        return "?" in text or any(text_lower.strip().startswith(qs) for qs in question_starters)

    def _update_emotional_state(self, chatbot_response: str) -> None:
        """Update emotional state from the assistant response style/content."""
        if not chatbot_response:
            return

        text = chatbot_response.lower()
        reassuring_cues = ["kein grund zur sorge", "gut behandelbar", "selten", "normal", "beruhigen"]
        uncertainty_cues = ["keine details", "nicht in den unterlagen", "kann ich nicht", "unklar"]

        if any(cue in text for cue in reassuring_cues):
            self.emotional_state["anxiety"] = self._clamp01(self.emotional_state["anxiety"] - 0.08)
            self.emotional_state["trust"] = self._clamp01(self.emotional_state["trust"] + 0.06)

        if any(cue in text for cue in uncertainty_cues):
            self.emotional_state["anxiety"] = self._clamp01(self.emotional_state["anxiety"] + 0.10)
            self.emotional_state["trust"] = self._clamp01(self.emotional_state["trust"] - 0.06)

        if len(chatbot_response) > 750:
            self.emotional_state["clarity"] = self._clamp01(self.emotional_state["clarity"] - 0.10)
        elif len(chatbot_response) < 250:
            self.emotional_state["clarity"] = self._clamp01(self.emotional_state["clarity"] + 0.05)

    def _should_disclose_hidden_fact(self, last_chatbot_msg: Optional[str]) -> bool:
        """Decide when a hidden fact should be disclosed naturally."""
        if self.hidden_fact_disclosed or not self.persona.hidden_fact:
            return False

        # Mention medically relevant facts when prompted by related content,
        # or organically if concern remains high after a few turns.
        text = (last_chatbot_msg or "").lower()
        relevance_keywords = [
            "allerg", "vorerkrank", "operation", "medik", "nüchtern", "essen", "getrunken",
            "zahn", "krone", "piercing", "blutung", "kaiserschnitt", "risiko", "komplikation"
        ]

        if any(k in text for k in relevance_keywords):
            return True

        return self.questions_asked >= 3 and self.emotional_state["anxiety"] >= 0.7

    def _conversation_phase(self) -> str:
        if self.max_questions <= 0:
            return "exploration"
        ratio = self.questions_asked / max(1, self.max_questions)
        if ratio < 0.25:
            return "opening"
        if ratio < 0.7:
            return "exploration"
        return "closing"

    def _normalize_informal_style(self, text: str) -> str:
        """Convert formal German pronouns to informal chat style where possible."""
        replacements = [
            (r"\bSie\b", "du"),
            (r"\bsie\b", "du"),
            (r"\bIhnen\b", "dir"),
            (r"\bihnen\b", "dir"),
            (r"\bIhrer\b", "deiner"),
            (r"\bihrer\b", "deiner"),
            (r"\bIhre\b", "deine"),
            (r"\bihre\b", "deine"),
            (r"\bIhr\b", "dein"),
            (r"\bihr\b", "dein"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    def _get_system_prompt(self) -> str:
        """Generate system prompt based on persona."""
        language_instructions = {
            "de": "Antworte auf Deutsch. Stelle Fragen auf Deutsch.",
            "en": "Respond in English. Ask questions in English.",
            "tr": "Türkçe cevap ver. Türkçe soru sor.",
        }
        
        # 1. Base Persona Description
        base_description = self.persona.get_persona_description()

        # 2. Hidden Fact Instruction
        hidden_instruction = ""
        if self.persona.hidden_fact:
            hidden_instruction = (
                f"\nWICHTIGE HINTERGRUNDINFO: {self.persona.hidden_fact}\n"
                "Bringe diese Info natürlich ein, sobald sie medizinisch relevant wird oder zu deiner Sorge passt."
            )
            
        # 3. Construct the Full Prompt
        full_prompt = f"""{base_description}

Du hast einen Termin für eine {self.procedure_name} und sprichst mit einem medizinischen Assistenten.
{language_instructions.get(self.persona.language, language_instructions["de"])}

{hidden_instruction}

Deine Interaktions-Richtlinien:
1. Sprich natürlich wie ein echter Patient, nicht wie ein Regelwerk
2. Du weißt, dass du mit einem Chatbot schreibst: schreibe kurz, direkt und chat-typisch
3. Verwende Alltagssprache mit "du", nicht formelles "Sie"
4. Sei nicht übertrieben höflich (keine Floskeln wie "Guten Tag", "bitte erklären Sie")
5. Nutze bei Aufregung oder wichtigen Punkten gelegentlich GROSSBUCHSTABEN zur Betonung einzelner Wörter
6. Verwende kurze Alltagssprache, gelegentlich mit spontanen Füllwörtern wie "okay", "hm", "verstehe"
7. Stelle pro Antwort maximal eine Frage
8. Wenn der Chatbot dir eine Frage stellt, antworte zuerst darauf
9. Halte deine Antworten knapp (1-3 Sätze), aber menschlich und kontextbezogen
10. Wiederhole nicht dieselbe Frage, wenn sie schon beantwortet wurde
"""
        
        # Add education/age adjustments
        if self.persona.education_level == "low":
            full_prompt += "\nVermeide Fachbegriffe. Stelle einfache Fragen."
        elif self.persona.education_level == "high":
            full_prompt += "\nDu kannst medizinische Fachbegriffe verwenden."
        
        return full_prompt
    
    def _get_initial_question(self) -> str:
        """
        Dynamically generate the opening question based on the procedure and persona.
        This allows the same agent to be used for any document.
        """
        system_prompt = self._get_system_prompt()
        
        # Context for the opening move
        trigger_prompt = (
            f"Du startest gerade das Gespräch mit dem medizinischen Assistenten über das Thema: '{self.procedure_name}'.\n"
            "Stelle deine erste Frage. Sei direkt, chat-typisch und deinem Charakter entsprechend.\n"
            f"Beispiel: 'ok, wie läuft die {self.procedure_name} genau ab?'"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": trigger_prompt}
                ],
                temperature=self.temperature
            )

            question = response.choices[0].message.content.strip()
            question = re.sub(r"^[\-\*\d\.]\s+", "", question, flags=re.MULTILINE).strip()
            
            # Validation: ensure not empty
            if not question or len(question) < 5:
                print(f"Warning: Empty initial question generated. Using fallback.")
                return f"ok, wie läuft die {self.procedure_name} genau ab?"
            
            return question
            
        except Exception as e:
            print(f"Error generating initial question: {e}")
            # Fallback just in case
            return f"kannst du mir kurz sagen, wie die {self.procedure_name} abläuft?"
    
    def ask_question(self, chatbot_response: Optional[str] = None) -> str:
        """
        Generate the next patient question based on conversation history.
        
        Args:
            chatbot_response: The chatbot's previous answer (optional)
        
        Returns:
            str: The patient's question
        """
        # Add chatbot response to history
        if chatbot_response:
            self.conversation_history.append({
                "role": "assistant",
                "content": chatbot_response
            })
            self._update_emotional_state(chatbot_response)
        
        # First question
        if self.questions_asked == 0:
            question = self._get_initial_question()
        else:
            # Generate follow-up question
            question = self._generate_question()
        
        # Update history and state
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.questions_asked += 1
        
        return question
    
    def _generate_question(self) -> str:
        """Generate a question using the LLM."""
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(self.conversation_history)

        # Check if chatbot asked a question in last response
        last_chatbot_msg = None
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_chatbot_msg = msg["content"]
                break

        phase = self._conversation_phase()
        asked_by_chatbot = self._contains_question(last_chatbot_msg or "")
        disclose_hidden = self._should_disclose_hidden_fact(last_chatbot_msg)

        emotional_hint = (
            f"Aktueller innerer Zustand: Angst={self.emotional_state['anxiety']:.2f}, "
            f"Vertrauen={self.emotional_state['trust']:.2f}, Klarheit={self.emotional_state['clarity']:.2f}."
        )

        if asked_by_chatbot:
            trigger = (
                f"Gesprächsphase: {phase}. {emotional_hint} "
                "Antworte zuerst direkt auf die Frage des Chatbots. "
                "Falls noch Unsicherheit bleibt, stelle eine kurze Anschlussfrage."
            )
        else:
            trigger = (
                f"Gesprächsphase: {phase}. {emotional_hint} "
                "Reagiere natürlich auf die letzte Information. "
                "Stelle eine neue, nicht wiederholte und inhaltlich passende Folgefrage oder äußere eine Sorge."
            )

        if disclose_hidden and self.persona.hidden_fact:
            trigger += (
                f" Integriere jetzt diese persönliche Info unauffällig in deine Antwort: '{self.persona.hidden_fact}'."
            )
            self.hidden_fact_disclosed = True
        
        messages.append({
            "role": "user",
            "content": trigger
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            text = response.choices[0].message.content.strip()
            if not text:
                return "ich bin noch unsicher, kannst du das bitte einfacher sagen?"

            # Soft cleanup in case the model drifts into list-like formatting.
            text = re.sub(r"^[\-\*\d\.]\s+", "", text, flags=re.MULTILINE).strip()
            text = self._normalize_informal_style(text)
            return text
        
        except Exception as e:
            print(f"Error generating patient response: {e}")
            fallbacks = [
                "kannst du das nochmal in einfachen worten sagen?",
                "ok, was bedeutet das jetzt konkret für mich?",
                "ich bin noch unsicher, was ist jetzt der WICHTIGSTE punkt?"
            ]
            return fallbacks[self.questions_asked % len(fallbacks)]
    
    def is_satisfied(self) -> bool:
        """Check if patient has asked enough questions."""
        return self.questions_asked >= self.max_questions
    
    def answer_comprehension_questions(self, test_questions: List[str]) -> Dict[str, str]:
        """
        Answer questions to test comprehension (for recipient-side evaluation).
        
        Args:
            test_questions: List of questions to test patient understanding
        
        Returns:
            dict: Answers to comprehension questions
        """
        system_prompt = f"""{self.persona.get_persona_description()}

Du hattest gerade ein Aufklärungsgespräch über {self.procedure_name}.
Beantworte die folgenden Fragen basierend NUR auf den Informationen, die du im Gespräch erhalten hast.
Wenn du etwas nicht weißt, sage das ehrlich."""
        
        # Build conversation context
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Chatbot'}: {msg['content']}"
            for msg in self.conversation_history
        ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Hier ist das Gespräch:\n\n{conversation_text}"}
        ]
        
        answers = {}
        for question in test_questions:
            messages.append({"role": "user", "content": question})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            answer = response.choices[0].message.content.strip()
            answers[question] = answer
            messages.append({"role": "assistant", "content": answer})
        
        return answers
    
    def get_conversation_log(self) -> List[Dict]:
        """Return the full conversation history."""
        return self.conversation_history
    
    def reset(self):
        """Reset the patient for a new conversation."""
        self.conversation_history = []
        self.questions_asked = 0
        self.hidden_fact_disclosed = False

        anxiety_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
        self.emotional_state = {
            "anxiety": anxiety_map.get(self.persona.anxiety_level, 0.5),
            "trust": 0.45,
            "clarity": 0.5,
        }


# Convenience function to create patients with predefined personas
def create_patient(persona_type: str = "baseline", **kwargs) -> PatientAgent:
    """
    Factory function to create a patient with a predefined persona.
    
    Args:
        persona_type: One of the predefined persona names
        **kwargs: Additional arguments passed to PatientAgent
    
    Returns:
        PatientAgent instance
    """
    return PatientAgent(persona_type=persona_type, **kwargs)


if __name__ == "__main__":
    # Test different personas
    print("=== Testing Patient Agent with Different Personas ===\n")
    
    # Test 1: Baseline patient
    print("--- Persona: Baseline ---")
    patient1 = create_patient("baseline", procedure_name="Narkose")
    q1 = patient1.ask_question()
    print(f"Patient ({patient1.persona.name}, {patient1.persona.age}): {q1}\n")
    
    # Test 2: Elderly patient
    print("--- Persona: Anaesthesia Risk ---")
    patient2 = create_patient("anesthesia_risk", procedure_name="Narkose")
    q2 = patient2.ask_question()
    print(f"Patient ({patient2.persona.name}, {patient2.persona.age}): {q2}\n")
    
    
    # Test 4: Comprehension evaluation
    print("--- Testing Comprehension Evaluation ---")
    patient4 = create_patient("baseline")
    patient4.ask_question()
    patient4.ask_question("Die Narkose wird verwendet, um Schmerzen während der Operation zu verhindern. Sie werden schlafen und nichts spüren.")
    
    test_questions = [
        "Wofür wird die Narkose verwendet?",
        "Werde ich während der Operation etwas spüren?"
    ]
    
    comprehension = patient4.answer_comprehension_questions(test_questions)
    for q, a in comprehension.items():
        print(f"Q: {q}")
        print(f"A: {a}\n")