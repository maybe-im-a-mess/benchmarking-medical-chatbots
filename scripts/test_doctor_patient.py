import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


from chatbot.doctor_agent import DoctorAgent
from chatbot.patient_agent import PatientAgent, PatientPersona, create_patient
from chatbot.embeddings import load_document_store


if __name__ == "__main__":
    # Test run the conversation
    doc_store = load_document_store()
    doctor = DoctorAgent(doc_store, model="gpt-5-mini")
    patient = create_patient("low_education", procedure_name="Narkose", model="gpt-5-mini")

    print("=== Doctor-Patient Conversation Test ===\n")
    
    question = patient.ask_question()
    print(f"Patient ({patient.persona.name}, {patient.persona.age}): {question}\n")
    
    response = doctor.respond(question)
    print(f"Doctor: {response['response']}\n")

    