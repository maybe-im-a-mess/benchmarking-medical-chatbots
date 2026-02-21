import json
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pathlib import Path


class DialogueManager:
    """
    Manages the conversation flow between chatbot and patient agents.
    Tracks mandatory questions, conversation state, and generates conversation logs.
    """
    def __init__(self, 
                 chatbot_agent, 
                 patient_agent, 
                 max_turns: int = 10, 
                 min_turns: int = 5,
                 procedure_name: str = "Narkose",
                 mandatory_questions_path: str = "data/mandatory_questions.json",
                 mode: str = "active"):
        """
        Parameters:
            chatbot_agent: The chatbot agent instance
            patient_agent: The PatientAgent instance
            max_turns: Maximum conversation turns
            min_turns: Minimum turns before ending
            procedure_name: Medical procedure being discussed
        """
        self.chatbot = chatbot_agent
        self.patient = patient_agent
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.procedure_name = procedure_name
        self.mandatory_questions_path = Path(mandatory_questions_path)
        self.mode = mode

        # Conversation state
        self.conversation_history = []
        self.mandatory_asked = set()
        self.turn_count = 0
        self.last_mandatory_turn = 0

        # Embedding model for semantic matching
        self.embedding_model_name = "paraphrase-multilingual-mpnet-base-v2"
        self.similarity_threshold = 0.55
        self.embedder = SentenceTransformer(self.embedding_model_name)

        # Load mandatory questions
        self.mandatory_questions = self._load_mandatory_questions()
        self._init_mandatory_question_embeddings()

    def _get_question_key(self, q: Dict) -> str:
        return q.get("question_id") or q.get("content") or ""

    def _init_mandatory_question_embeddings(self) -> None:
        self.mandatory_question_keys = []
        self.mandatory_question_texts = []
        for q in self.mandatory_questions:
            key = self._get_question_key(q)
            text = q.get("content") or ""
            if key and text:
                self.mandatory_question_keys.append(key)
                self.mandatory_question_texts.append(text)

        self.question_key_to_index = {
            key: idx for idx, key in enumerate(self.mandatory_question_keys)
        }

        if not self.mandatory_question_texts:
            self.question_embeddings = None
            return

        self.question_embeddings = self.embedder.encode(
            self.mandatory_question_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def _normalize_procedure_key(self, key: str, candidates: List[str]) -> Optional[str]:
        """Match procedure name to a key in a dict (case-insensitive, with/without .md)."""
        if not key:
            return None
        key_md = key if key.endswith(".md") else f"{key}.md"
        for cand in candidates:
            if cand.lower() == key_md.lower():
                return cand
        for cand in candidates:
            if cand.lower().replace(".md", "") == key.lower():
                return cand
        return None

    def _load_mandatory_questions(self) -> List[Dict]:
        """Load mandatory questions for the current procedure."""
        if not self.mandatory_questions_path.exists():
            return []
        with open(self.mandatory_questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        proc_key = self._normalize_procedure_key(self.procedure_name, list(data.keys()))
        if not proc_key:
            return []
        entries = data.get(proc_key, [])
        if not entries:
            return []
        questions = []
        for entry in entries:
            for q in entry.get("questions", []):
                questions.append(q)
        return questions

    def _best_semantic_match(self, patient_question: str, pending_questions: List[Dict]) -> Dict:
        """Return best pending question and similarity score using embeddings."""
        if self.question_embeddings is None:
            return {"best_question": None, "best_score": 0.0}

        pending_indices = []
        pending_list = []
        for q in pending_questions:
            key = self._get_question_key(q)
            idx = self.question_key_to_index.get(key)
            if idx is not None:
                pending_indices.append(idx)
                pending_list.append(q)

        if not pending_indices:
            return {"best_question": None, "best_score": 0.0}

        query_vec = self.embedder.encode(
            [patient_question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        candidates = self.question_embeddings[pending_indices]
        scores = np.dot(candidates, query_vec)
        best_idx = int(np.argmax(scores))
        return {"best_question": pending_list[best_idx], "best_score": float(scores[best_idx])}

    def _should_ask_mandatory_now(self, patient_question: str) -> Optional[str]:
        """Decide whether to ask a mandatory question this turn.

        Returns:
            "contextual" | "safety_net" | None
        """
        pending = [q for q in self.mandatory_questions if (q.get("question_id") or q.get("content")) not in self.mandatory_asked]
        if not pending:
            return None

        # Contextual intervention when semantic match is strong
        semantic = self._best_semantic_match(patient_question, pending)
        if semantic["best_score"] >= self.similarity_threshold:
            return "contextual"

        # Safety net intervention near the end
        remaining_turns = max(0, self.max_turns - self.turn_count)
        if remaining_turns <= len(pending):
            return "safety_net"

        return None

    def _format_mandatory_questions_for_prompt(self) -> str:
        if not self.mandatory_questions:
            return ""
        lines = []
        for q in self.mandatory_questions:
            q_id = q.get("question_id") or ""
            content = q.get("content") or ""
            if q_id:
                lines.append(f"- ({q_id}) {content}")
            else:
                lines.append(f"- {content}")
        return "\n".join(lines)

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
            if self.mode == "active" and len(self.mandatory_asked) < len(self.mandatory_questions):
                return False
            return True
        
        return False
    
    def run_conversation(self) -> List[Dict]:
        """
        Run the complete conversation between the chatbot and the patient agents.

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

            # Get previous chatbot response from history (if any)
            previous_response = None
            if self.conversation_history:
                previous_response = self.conversation_history[-1]["chatbot_response"]


            #Patient asks a question
            patient_question = self.patient.ask_question(previous_response)
            print(f"Patient: {patient_question}\n")

            # Ask mandatory question naturally when it fits
            pending_questions = []
            for q in self.mandatory_questions:
                q_id = q.get("question_id") or q.get("content")
                if q_id not in self.mandatory_asked:
                    pending_questions.append(q)

            supervisor_triggered = False
            mandatory_question_asked = None
            mandatory_instruction = ""
            intervention_type = None

            if self.mode == "active" and pending_questions:
                intervention_type = self._should_ask_mandatory_now(patient_question)

            if self.mode == "active" and pending_questions and intervention_type:
                # Prefer semantic match for contextual, else first pending for safety net
                if intervention_type == "contextual":
                    semantic = self._best_semantic_match(patient_question, pending_questions)
                    best_q = semantic["best_question"] or pending_questions[0]
                else:
                    best_q = pending_questions[0]

                q_id = best_q.get("question_id") or best_q.get("content")
                mandatory_question_asked = {
                    "question_id": q_id,
                    "content": best_q.get("content"),
                    "reason": best_q.get("reason")
                }
                if q_id:
                    self.mandatory_asked.add(q_id)
                self.last_mandatory_turn = self.turn_count
                supervisor_triggered = True

                mandatory_instruction = (
                    "Bitte stelle die folgende Pflichtfrage auf natürliche Weise im Verlauf deiner Antwort: "
                    f"{best_q.get('content')}"
                )

            # Build supervision layer
            mandatory_list = self._format_mandatory_questions_for_prompt()
            if self.mode == "passive":
                extra_system_instructions = (
                    "Folgende Pflichtfragen sollten idealerweise im Gespräch abgedeckt werden, "
                    "aber entscheide selbst, wann sie passen:\n"
                    f"{mandatory_list}"
                )
            else:
                if intervention_type:
                    extra_system_instructions = mandatory_instruction
                else:
                    extra_system_instructions = f"Pflichtfragen-Liste:\n{mandatory_list}" if mandatory_list else ""

            # Chatbot responds
            chatbot_response_data = self.chatbot.respond(
                patient_question,
                extra_system_instructions=extra_system_instructions
            )
            chatbot_response = chatbot_response_data["response"]
            retrieved_chunks = chatbot_response_data["retrieved_chunks"]
            citations = chatbot_response_data.get("citations", {})
            
            print(f"Chatbot: {chatbot_response}\n")
            print(f"[Retrieved {len(retrieved_chunks)} chunks]\n")

            # Log the turn
            turn_data = {
                "turn": self.turn_count,
                "patient_question": patient_question,
                "chatbot_response": chatbot_response,
                "mandatory_question": mandatory_question_asked,
                "supervisor_triggered": supervisor_triggered,
                "intervention_type": intervention_type,
                "citations": citations,
                "prompt_messages": chatbot_response_data.get("metadata", {}).get("prompt_messages"),
                "retrieved_chunks": [
                    {
                        "content": chunk["content"],
                        "score": chunk["score"],
                        "meta": chunk["meta"],
                        "id": chunk.get("id")
                    }
                    for chunk in retrieved_chunks
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            self.conversation_history.append(turn_data)

        # Conversation end
        print(f"\n{'='*60}")
        print(f"Conversation ended after {self.turn_count} turns")
        print(f"{'='*60}\n")
        
        # Generate final conversation log
        conversation_log = self._generate_conversation_log()
        
        return conversation_log
    
    def _generate_conversation_log(self) -> Dict:
        """Generate structured conversation log for evaluation."""
        total_citations = sum(
            turn.get("citations", {}).get("total_citations", 0)
            for turn in self.conversation_history
        )
        turns_missing_citations = sum(
            1 for turn in self.conversation_history
            if turn.get("citations", {}).get("total_citations", 0) == 0
        )
        return {
            "metadata": {
                "procedure": self.procedure_name,
                "mandatory_questions_total": len(self.mandatory_questions),
                "mandatory_questions_asked": len(self.mandatory_asked),
                "mode": self.mode,
                "patient_persona": {
                    "name": self.patient.persona.name,
                    "age": self.patient.persona.age,
                    "sex": self.patient.persona.sex,
                    "education_level": self.patient.persona.education_level,
                    "language": self.patient.persona.language
                },
                "chatbot_model": self.chatbot.model,
                "patient_model": self.patient.model,
                "total_turns": self.turn_count,
                "conversation_date": datetime.now().isoformat()
            },
            "conversation": self.conversation_history,
            "summary": {
                "total_patient_questions": len(self.conversation_history),
                "total_chatbot_responses": len(self.conversation_history),
                "avg_response_length": sum(
                    len(turn["chatbot_response"]) 
                    for turn in self.conversation_history
                ) / len(self.conversation_history) if self.conversation_history else 0,
                "total_citations": total_citations,
                "turns_missing_citations": turns_missing_citations
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
    chatbot = DoctorAgent(doc_store, model="gpt-5-mini")
    patient = create_patient("middle_aged", procedure_name="Narkose")
    
    # Run conversation
    manager = DialogueManager(chatbot, patient, max_turns=3, min_turns=1)
    conversation = manager.run_conversation()
    
    # Save conversation
    manager.save_conversation("data/conversations/test_conversation.json")
    
    # Print summary
    print("\n=== Conversation Summary ===")
    print(f"Total turns: {conversation['summary']['total_patient_questions']}")
    print(f"Mandatory asked: {conversation['metadata']['mandatory_questions_asked']}/{conversation['metadata']['mandatory_questions_total']}")
    print(f"Avg response length: {conversation['summary']['avg_response_length']:.0f} chars")
