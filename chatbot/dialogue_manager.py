import json
from datetime import datetime
from typing import List, Dict, Optional


class DialogueManager:
    """
    Manages the conversation flow between doctor and patient agents.
    Tracks mandatory questions, conversation state, and generates conversation logs.
    """
    # Mandatory questions that should be asked during consent conversation
    MANDATORY_QUESTIONS = [
        "Haben Sie Allergien oder Unverträglichkeiten?",
        "Nehmen Sie regelmäßig Medikamente ein?",
        "Haben Sie Vorerkrankungen, die für den Eingriff relevant sein könnten?",
        "Gab es in Ihrer Familie Komplikationen bei ähnlichen Eingriffen?",
        "Wie ist Ihr aktueller Gesundheitszustand?",
    ]
    def __init__(self, 
                 doctor_agent, 
                 patient_agent, 
                 max_turns: int = 10, 
                 min_turns: int = 5,
                 procedure_name: str = "Narkose"):
        """
        Parameters:
            doctor_agent: The DoctorAgent instance
            patient_agent: The PatientAgent instance
            max_turns: Maximum conversation turns
            min_turns: Minimum turns before ending
            procedure_name: Medical procedure being discussed
        """
        self.doctor = doctor_agent
        self.patient = patient_agent
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.procedure_name = procedure_name

        # Conversation state
        self.conversation_history = []
        self.mandatory_asked = set()
        self.topics_covered = set()
        self.turn_count = 0

    def should_end_conversation(self) -> bool:
        """
        Determine if conversation should end.
        
        Should end when:
        - Max turns reached, OR
        - Min turns reached AND patient is satisfied
        """
        if self.turn_count >= self.max_turns:
            return True
        
        if self.turn_count >= self.min_turns and self.patient.is_satisfied():
            return True
        
        return False
    
    def run_conversation(self) -> List[Dict]:
        """
        Run the complete conversation between the doctor and the patient agents.

        Returns:
            dict: Complete conversation log with metadata
        """
        print(f"\n{'='*60}")
        print(f"Starting conversation about {self.procedure_name}")
        print(f"Patient: {self.patient.persona.name}, {self.patient.persona.age} years old")
        print(f"{'='*60}\n")

        while not self.should_end_conversation():
            self.turn_count += 1
            print(f"--- Turn {self.turn_count}/{self.max_turns} ---")

            # Get previous doctor response from history (if any)
            previous_response = None
            if self.conversation_history:
                previous_response = self.conversation_history[-1]["doctor_response"]


            #Patient asks a question
            patient_question = self.patient.ask_question(previous_response)
            print(f"Patient: {patient_question}\n")

            # TODO: check topic coverage

            # Doctor responds
            doctor_response_data = self.doctor.respond(patient_question)
            doctor_response = doctor_response_data["response"]
            retrieved_chunks = doctor_response_data["retrieved_chunks"]
            
            print(f"Doctor: {doctor_response}\n")
            print(f"[Retrieved {len(retrieved_chunks)} chunks]\n")

            # Log the turn
            turn_data = {
                "turn": self.turn_count,
                "patient_question": patient_question,
                "doctor_response": doctor_response,
                "retrieved_chunks": [
                    {
                        "content": chunk["content"],
                        "score": chunk["score"],
                        "meta": chunk["meta"]
                    }
                    for chunk in retrieved_chunks
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            self.conversation_history.append(turn_data)

        # Conversation end
        print(f"\n{'='*60}")
        print(f"Conversation ended after {self.turn_count} turns")
        print(f"Topics covered: {', '.join(self.topics_covered)}")
        print(f"{'='*60}\n")
        
        # Generate final conversation log
        conversation_log = self._generate_conversation_log()
        
        return conversation_log
    
    def _generate_conversation_log(self) -> Dict:
        """Generate structured conversation log for evaluation."""
        return {
            "metadata": {
                "procedure": self.procedure_name,
                "patient_persona": {
                    "name": self.patient.persona.name,
                    "age": self.patient.persona.age,
                    "sex": self.patient.persona.sex,
                    "education_level": self.patient.persona.education_level,
                    "language": self.patient.persona.language
                },
                "doctor_model": self.doctor.model,
                "patient_model": self.patient.model,
                "total_turns": self.turn_count,
                "topics_covered": list(self.topics_covered),
                "conversation_date": datetime.now().isoformat()
            },
            "conversation": self.conversation_history,
            "summary": {
                "total_patient_questions": len(self.conversation_history),
                "total_doctor_responses": len(self.conversation_history),
                "avg_response_length": sum(
                    len(turn["doctor_response"]) 
                    for turn in self.conversation_history
                ) / len(self.conversation_history) if self.conversation_history else 0,
                "topics_covered_count": len(self.topics_covered)
            }
        }
    
    def save_conversation(self, filepath: str):
        """Save conversation log to JSON file."""
        conversation_log = self._generate_conversation_log()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_log, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Conversation saved to {filepath}")


if __name__ == "__main__":
    # Test the dialogue manager
    from embeddings import load_document_store
    from doctor_agent import DoctorAgent
    from patient_agent import create_patient
    
    print("Loading document store...")
    doc_store = load_document_store()
    
    print("Initializing agents...")
    doctor = DoctorAgent(doc_store, model="gpt-5-mini")
    patient = create_patient("middle_aged", procedure_name="Narkose")
    
    # Run conversation
    manager = DialogueManager(doctor, patient, max_turns=3, min_turns=1)
    conversation = manager.run_conversation()
    
    # Save conversation
    manager.save_conversation("data/conversations/test_conversation.json")
    
    # Print summary
    print("\n=== Conversation Summary ===")
    print(f"Total turns: {conversation['summary']['total_patient_questions']}")
    print(f"Topics: {', '.join(conversation['metadata']['topics_covered'])}")
    print(f"Avg response length: {conversation['summary']['avg_response_length']:.0f} chars")
