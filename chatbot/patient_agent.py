from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import random


load_dotenv()

class PatientPersona:
    """
    Defines a persona for the patient agent.
    """
    def __init__(self,
                 age: int,
                 sex: str,
                 language: str = "de",
                 education_level: str = "medium",
                 detail_preference: str = "medium",
                 name: Optional[str] = None):
        """
        Parameters:
            :param age: Patient age (e.g., 25, 40, 75)
            :param sex: 'male' or 'female'
            :param language: 'de' (German), 'en' (English), 'tr' (Turkish), etc.
            :param education_level: 'low', 'medium', 'high'
            :param detail_preference: 'low' (brief answers), 'medium' (default), 'high' (wants details)
            :param name: Optional patient name
        """
        self.age = age
        self.sex = sex
        self.language = language
        self.education_level = education_level
        self.detail_preference = detail_preference
        self.name = name or self._generate_name()

    def _generate_name(self) -> str:
        """Generate a random name"""
        names = {
            "male": ["Max", "Lukas", "Leon", "Finn", "Elias"],
            "female": ["Emma", "Mia", "Hannah", "Sophia", "Lea"]
        }
        return random.choice(names.get(self.sex, ["Patient"]))
    
    def get_persona_description(self) -> str:
        """Return a textual description of the patient's persona."""
        sex_de = "männlich" if self.sex == "male" else "weiblich"
        
        education_map = {
            "low": "einfacher Bildungsstand, verwendet einfache Sprache",
            "medium": "durchschnittlicher Bildungsstand",
            "high": "hoher Bildungsstand, versteht medizinische Fachbegriffe"
        }
        
        detail_map = {
            "low": "möchte kurze, direkte Antworten",
            "medium": "möchte ausgewogene Informationen",
            "high": "möchte detaillierte, gründliche Erklärungen"
        }
        
        return f"""Du bist {self.name}, {self.age} Jahre alt, {sex_de}.
Bildung: {education_map[self.education_level]}
Informationsbedarf: {detail_map[self.detail_preference]}"""


class PatientAgent:
    """
    Simulates a patient asking questions about a medical procedure.
    """
    PERSONAS = {
        "young_educated": PatientPersona(
            age=28, sex="female", education_level="high", 
            detail_preference="high", name="Sarah"
        ),
        "elderly_medium": PatientPersona(
            age=72, sex="male", education_level="medium", 
            detail_preference="medium", name="Herr Schmidt"
        ),
        "middle_aged": PatientPersona(
            age=45, sex="male", education_level="medium", 
            detail_preference="medium", name="Thomas"
        ),
        "low_education": PatientPersona(
            age=55, sex="female", education_level="low", 
            detail_preference="low", name="Frau Müller"
        ),
        "detail_oriented": PatientPersona(
            age=38, sex="female", education_level="high", 
            detail_preference="high", name="Dr. Wagner"
        ),
    }
    
    def __init__(self,
                procedure_name: str = "Narkose",
                persona: Optional[PatientPersona] = None,
                persona_type: Optional[str] = None,
                model: str = "gpt-5-mini",
                max_questions: int = 8):
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
        self.topics_covered = set()

    def _get_system_prompt(self) -> str:
        """Generate system prompt based on persona."""
        language_instructions = {
            "de": "Antworte auf Deutsch. Stelle Fragen auf Deutsch.",
            "en": "Respond in English. Ask questions in English.",
            "tr": "Türkçe cevap ver. Türkçe soru sor.",
        }
        
        base_prompt = f"""{self.persona.get_persona_description()}

Du hast einen Termin für eine {self.procedure_name} und sprichst mit einem medizinischen Assistenten.
{language_instructions.get(self.persona.language, language_instructions["de"])}

Deine Interaktions-Richtlinien:
1. Belehre nicht: Sprich wie ein unbedarfter Patient. Verwende kurze Sätze.
2. Simuliere Missverständnisse: Wenn die KI komplexe Fachbegriffe ohne Erklärung verwendet, drücke Verwirrung aus (z. B. "Was bedeutet 'Laparotomie'?").
3. Fordere die KI heraus: Wenn die KI vage bleibt, bitte um Klarstellung (z. B. "Aber wie sehr genau wird es wehtun?").
4. Ziel: Dein Ziel ist es, dich sicher und informiert zu fühlen. Beende das Gespräch erst, wenn du das Gefühl hast, dass deine Fragen beantwortet wurden."""
        # Add persona-specific adjustments
        if self.persona.education_level == "low":
            base_prompt += "\nVermeide Fachbegriffe. Stelle einfache, direkte Fragen."
        elif self.persona.education_level == "high":
            base_prompt += "\nDu verstehst medizinische Begriffe und kannst spezifische Fragen stellen."
        
        if self.persona.age > 65:
            base_prompt += f"\nAls {self.persona.age}-jährige Person interessierst du dich besonders für altersbedingte Risiken."
        
        return base_prompt
    
    def _get_initial_question(self) -> str:
        """Generate the first question based on persona."""
        initial_questions = {
            "de": {
                "low": f"Was passiert bei der {self.procedure_name}?",
                "medium": f"Können Sie mir erklären, wie die {self.procedure_name} abläuft?",
                "high": f"Ich hätte gerne detaillierte Informationen über den Ablauf der {self.procedure_name}. Können Sie mir das erklären?"
            },
            "en": {
                "low": f"What happens during {self.procedure_name}?",
                "medium": f"Can you explain how {self.procedure_name} works?",
                "high": f"I would like detailed information about the {self.procedure_name} procedure. Could you explain?"
            }
        }
        
        lang = self.persona.language
        level = self.persona.education_level
        
        return initial_questions.get(lang, initial_questions["de"]).get(level, 
               initial_questions["de"]["medium"])
    
    def ask_question(self, doctor_response: Optional[str] = None) -> str:
        """
        Generate the next patient question based on conversation history.
        
        Args:
            doctor_response: The doctor's previous answer (optional)
        
        Returns:
            str: The patient's question
        """
        # Add doctor's response to history
        if doctor_response:
            self.conversation_history.append({
                "role": "doctor",
                "content": doctor_response
            })
        
        # First question
        if self.questions_asked == 0:
            question = self._get_initial_question()
        else:
            # Generate follow-up question
            question = self._generate_question()
        
        # Update history and state
        self.conversation_history.append({
            "role": "patient",
            "content": question
        })
        self.questions_asked += 1
        
        return question
    
    def _generate_question(self) -> str:
        """Generate a question using the LLM."""
        
        # Format conversation history
        conv_summary = self._format_conversation_for_context()
        
        system_prompt_with_context = f"""{self._get_system_prompt()}

    BISHERIGE KONVERSATION:
    {conv_summary}

    Basierend auf diesem Gespräch, stelle jetzt eine logische Folgefrage oder ein neues relevantes Thema."""

        messages = [{"role": "system", "content": system_prompt_with_context}]
        
        # Don't add history again since it's in system prompt
        messages.append({
            "role": "user", 
            "content": "Stelle jetzt deine nächste Frage."
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content.strip()
    
    def _format_conversation_for_context(self) -> str:
        """Format conversation history for context awareness."""
        if not self.conversation_history:
            return "Dies ist das erste Gespräch."
        
        formatted = []
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "Du" if msg["role"].lower() == "patient" else "Doctor"
            content = msg["content"][:200]  # Truncate long messages
            formatted.append(f"{role}: {content}...")
        
        return "\n".join(formatted)

    
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
            f"{'Patient' if msg['role'] == 'patient' else 'Doctor'}: {msg['content']}"
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
        self.topics_covered = set()


# Convenience function to create patients with predefined personas
def create_patient(persona_type: str = "middle_aged", **kwargs) -> PatientAgent:
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
    
    # Test 1: Young educated patient
    print("--- Persona: Young, Educated ---")
    patient1 = create_patient("young_educated", procedure_name="Narkose")
    q1 = patient1.ask_question()
    print(f"Patient ({patient1.persona.name}, {patient1.persona.age}): {q1}\n")
    
    # Test 2: Elderly patient
    print("--- Persona: Elderly ---")
    patient2 = create_patient("elderly_medium", procedure_name="Narkose")
    q2 = patient2.ask_question()
    print(f"Patient ({patient2.persona.name}, {patient2.persona.age}): {q2}\n")
    
    # Test 3: Custom persona (Turkish speaker)
    print("--- Persona: Custom (Turkish speaker) ---")
    turkish_persona = PatientPersona(
        age=35, 
        sex="female", 
        language="tr",
        education_level="medium",
        name="Ayşe"
    )
    patient3 = PatientAgent(
        procedure_name="Anestezi",
        persona=turkish_persona
    )
    q3 = patient3.ask_question()
    print(f"Patient ({patient3.persona.name}, {patient3.persona.age}): {q3}\n")
    
    # Test 4: Comprehension evaluation
    print("--- Testing Comprehension Evaluation ---")
    patient4 = create_patient("middle_aged")
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