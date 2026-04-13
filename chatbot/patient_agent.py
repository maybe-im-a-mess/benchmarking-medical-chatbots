from openai import OpenAI
import os
import re
import random
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
        "anticoagulation_risk": PatientPersona(
            age=37, sex="female", language="de", anxiety_level="high",
            education_level="medium", detail_preference="high",
            hidden_fact="Ich nehme täglich Blutverdünner wegen einer früheren Thrombose.",
            name="Sara"
        ),
        "trauma_history_risk": PatientPersona(
            age=28, sex="female", language="de", anxiety_level="high",
            education_level="medium", detail_preference="low",
            hidden_fact="Ich hatte bei einer früheren OP eine sehr schlechte Erfahrung und starke Angst.",
            name="Mila"
        ),
        "hypertension_risk": PatientPersona(
            age=35, sex="female", language="de", anxiety_level="medium",
            education_level="high", detail_preference="medium",
            hidden_fact="Ich habe seit der Schwangerschaft häufig hohen Blutdruck.",
            name="Clara"
        ),
        "language_barrier_risk": PatientPersona(
            age=32, sex="female", language="de", anxiety_level="medium",
            education_level="low", detail_preference="high",
            hidden_fact="Deutsch ist nicht meine Muttersprache und ich verstehe Fachbegriffe oft nicht sofort.",
            name="Aylin"
        ),
    }
    
    def __init__(self,
                procedure_name: str = "Narkose",
                persona: Optional[PatientPersona] = None,
                persona_type: Optional[str] = None,
                model: str = "gpt-5-mini",
                max_questions: int = 8,
                temperature: float = 0.8,
                random_seed: Optional[int] = None):
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
        self.random_seed = random_seed

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
        self.hidden_fact_mentions = 0
        self.max_hidden_fact_mentions = 2

        anxiety_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
        self.emotional_state = {
            "anxiety": anxiety_map.get(self.persona.anxiety_level, 0.5),
            "trust": 0.45,
            "clarity": 0.5,
        }
        self.min_questions_before_satisfaction = max(2, min(4, self.max_questions - 1))
        self.rng = random.Random(random_seed)

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
        """Disclose hidden fact only when the chatbot explicitly asks for relevant details."""
        if not self.persona.hidden_fact:
            return False

        if self.hidden_fact_mentions >= self.max_hidden_fact_mentions:
            return False

        return self._chatbot_requested_hidden_fact(last_chatbot_msg)

    def _hidden_fact_keywords(self) -> List[str]:
        hidden = (self.persona.hidden_fact or "").lower()
        if not hidden:
            return []

        if "latex" in hidden:
            return ["latex", "allerg"]
        if "kaiserschnitt" in hidden:
            return ["kaiserschnitt"]
        if "blutung" in hidden:
            return ["blutung"]
        if "frühstück" in hidden or "veneer" in hidden or "zahn" in hidden:
            return ["frühstück", "gegessen", "nüchtern", "veneer", "zahn"]

        return [w for w in re.findall(r"[a-zA-ZäöüÄÖÜß]+", hidden) if len(w) >= 6]

    def _question_mentions_hidden_fact(self, text: str) -> bool:
        if not text or not self.persona.hidden_fact:
            return False
        low = text.lower()
        return any(k in low for k in self._hidden_fact_keywords())

    def _chatbot_requested_hidden_fact(self, last_chatbot_msg: Optional[str]) -> bool:
        if not last_chatbot_msg:
            return False
        low = last_chatbot_msg.lower()
        if "?" not in low:
            return False
        return any(k in low for k in self._hidden_fact_keywords())

    def _fallback_question_without_hidden_fact(self) -> str:
        phase = self._conversation_phase()
        if phase == "opening":
            return "ok, kannst du mir den ablauf kurz und einfach erklären?"
        if phase == "exploration":
            return "verstehe, und was ist dabei für mich persönlich am wichtigsten?"
        return "ok, was soll ich mir als wichtigste punkte jetzt merken?"

    def _satisfaction_probability(self) -> float:
        # Readiness to stop rises with trust/clarity, lower anxiety, and later phase.
        progress = (self.questions_asked - self.min_questions_before_satisfaction + 1) / max(
            1, self.max_questions - self.min_questions_before_satisfaction + 1
        )
        progress = self._clamp01(progress)

        readiness = (
            0.45 * self.emotional_state["clarity"]
            + 0.35 * self.emotional_state["trust"]
            + 0.20 * (1.0 - self.emotional_state["anxiety"])
        )

        probability = 0.10 + 0.55 * progress + 0.35 * readiness
        if self.emotional_state["anxiety"] >= 0.8:
            probability -= 0.15

        return self._clamp01(probability)

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
                "Nenne diese Info NICHT von dir aus. Teile sie nur mit, wenn der Chatbot dich explizit danach fragt."
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
11. Bleibe strikt in der Rolle als Patientin: sprich über dich selbst (ich/mir/meine), nicht über den Körper oder Gesundheitszustand des Chatbots
12. Stelle NIEMALS Fragen wie "Nimmst du Medikamente?", "Hast du Allergien?" oder ähnliche Fragen über den Chatbot
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
            # Enforce strict hidden-fact policy also for the opening turn.
            if self._question_mentions_hidden_fact(question):
                question = f"ok, wie läuft die {self.procedure_name} genau ab?"
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
                "Falls noch Unsicherheit bleibt, stelle eine kurze Anschlussfrage über DEINE Situation (ich/mir/meine). "
                "Frage nicht nach dem Gesundheitszustand des Chatbots."
            )
        else:
            trigger = (
                f"Gesprächsphase: {phase}. {emotional_hint} "
                "Reagiere natürlich auf die letzte Information. "
                "Stelle eine neue, nicht wiederholte und inhaltlich passende Folgefrage oder äußere eine Sorge zu deiner eigenen Situation."
            )

        if disclose_hidden and self.persona.hidden_fact:
            trigger += (
                f" Integriere jetzt diese persönliche Info unauffällig in deine Antwort: '{self.persona.hidden_fact}'."
            )

        if self.persona.hidden_fact and self.hidden_fact_mentions >= self.max_hidden_fact_mentions:
            trigger += (
                " Deine persönliche Risikoinfo wurde schon genannt. Wiederhole sie jetzt NICHT erneut, "
                "außer der Chatbot fragt dich explizit direkt danach."
            )
        
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

            # Cap repeated hidden-fact mentions unless the chatbot explicitly asked for it.
            if self._question_mentions_hidden_fact(text):
                allow_repeat = self._chatbot_requested_hidden_fact(last_chatbot_msg)
                allow_repeat = allow_repeat and (self.hidden_fact_mentions < self.max_hidden_fact_mentions)

                if allow_repeat:
                    self.hidden_fact_mentions += 1
                    self.hidden_fact_disclosed = True
                else:
                    text = self._fallback_question_without_hidden_fact()

            # Guardrail: keep strict patient role and prevent questions about chatbot's own health.
            if self._is_role_confused_question(text):
                text = self._fallback_patient_perspective_question()

            return text
        
        except Exception as e:
            print(f"Error generating patient response: {e}")
            fallbacks = [
                "kannst du das nochmal in einfachen worten sagen?",
                "ok, was bedeutet das jetzt konkret für mich?",
                "ich bin noch unsicher, was ist jetzt der WICHTIGSTE punkt?"
            ]
            return fallbacks[self.questions_asked % len(fallbacks)]

    def _is_role_confused_question(self, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        patterns = [
            r"\bnimmst du\b",
            r"\bhast du\b",
            r"\bleidest du\b",
            r"\bbist du\b",
            r"\bbei dir\b",
            r"\bdeine tabletten\b",
            r"\bdeine allerg",
            r"\bdeine symptome\b",
        ]
        return any(re.search(p, low) for p in patterns)

    def _fallback_patient_perspective_question(self) -> str:
        phase = self._conversation_phase()
        if phase == "opening":
            return "ok, was ist für mich jetzt der wichtigste punkt vor dem eingriff?"
        if phase == "exploration":
            return "kannst du mir sagen, was ich in meiner situation konkret beachten muss?"
        return "ok, kannst du mir kurz zusammenfassen, was ich mir für mich merken sollte?"
    
    def is_satisfied(self) -> bool:
        """Probabilistically decide if the patient is satisfied enough to stop."""
        if self.questions_asked >= self.max_questions:
            return True
        if self.questions_asked < self.min_questions_before_satisfaction:
            return False

        probability = self._satisfaction_probability()
        return self.rng.random() < probability
    
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
        self.hidden_fact_mentions = 0

        anxiety_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
        self.emotional_state = {
            "anxiety": anxiety_map.get(self.persona.anxiety_level, 0.5),
            "trust": 0.45,
            "clarity": 0.5,
        }
        self.rng = random.Random(self.random_seed)


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