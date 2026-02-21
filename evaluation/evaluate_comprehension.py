import json
from pathlib import Path
from typing import Dict, List
from patient_agent import PatientAgent, PatientPersona
from openai import OpenAI
import os

class ComprehensionEvaluator:
    """
    Evaluates patient comprehension after a conversation.
    Separate from conversation generation for flexibility.
    """
    
    def __init__(self, 
                 comprehension_questions_path: str = "data/comprehension_questions"):
        self.questions_path = Path(comprehension_questions_path)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _load_comprehension_questions(self, procedure_name: str) -> List[str]:
        """Load procedure-specific comprehension questions."""
        question_file = self.questions_path / f"{procedure_name}.json"
        
        if question_file.exists():
            with open(question_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("questions", [])
        
        # Default fallback questions
        return [
            f"Was ist der Zweck von {procedure_name}?",
            "Welche Risiken wurden erwähnt?",
            f"Was passiert während {procedure_name}?",
            f"Was sollten Sie nach {procedure_name} beachten?"
        ]
    
    def _recreate_patient_from_log(self, conversation_log: Dict) -> PatientAgent:
        """Recreate the patient agent from conversation metadata."""
        metadata = conversation_log["metadata"]["patient_persona"]
        
        persona = PatientPersona(
            age=metadata["age"],
            sex=metadata["sex"],
            language=metadata.get("language", "de"),
            education_level=metadata["education_level"],
            detail_preference=metadata.get("detail_preference", "medium"),
            name=metadata["name"]
        )
        
        patient = PatientAgent(
            procedure_name=conversation_log["metadata"]["procedure"],
            persona=persona,
            model=conversation_log["metadata"]["patient_model"]
        )
        
        # Restore conversation history
        for turn in conversation_log["conversation"]:
            patient.conversation_history.append({
                "role": "user",
                "content": turn["patient_question"]
            })
            patient.conversation_history.append({
                "role": "assistant", 
                "content": turn["chatbot_response"]
            })
        
        return patient
    
    def _score_comprehension_answer(
        self, 
        question: str,
        patient_answer: str,
        ground_truth_topics: List[str]
    ) -> Dict:
        """
        Use LLM to score how well the patient understood the topic.
        
        Returns:
            {
                "score": float (0-1),
                "rationale": str
            }
        """
        prompt = f"""Du bist ein medizinischer Ausbilder, der das Verständnis eines Patienten bewertet.

FRAGE AN DEN PATIENTEN:
{question}

ANTWORT DES PATIENTEN:
{patient_answer}

RELEVANTE INFORMATIONEN AUS DEM AUFKLÄRUNGSGESPRÄCH:
{chr(10).join(f"- {topic}" for topic in ground_truth_topics)}

Bewerte auf einer Skala von 0.0 bis 1.0:
- 1.0 = Vollständig verstanden, korrekt wiedergegeben
- 0.7 = Größtenteils verstanden, kleinere Lücken
- 0.4 = Teilweise verstanden, wichtige Details fehlen
- 0.0 = Nicht verstanden oder falsch

Antworte im JSON-Format:
{{
    "score": <0.0-1.0>,
    "rationale": "<Kurze Begründung>"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def evaluate_conversation_file(self, conversation_file: str) -> Dict:
        """
        Evaluate patient comprehension for a saved conversation.
        
        Returns:
            {
                "conversation_id": str,
                "comprehension_test": {
                    "questions": List[str],
                    "answers": Dict[str, str],
                    "scores": Dict[str, float],
                    "average_score": float
                }
            }
        """
        # Load conversation log
        with open(conversation_file, "r", encoding="utf-8") as f:
            conversation_log = json.load(f)
        
        procedure = conversation_log["metadata"]["procedure"]
        
        # Load comprehension questions
        questions = self._load_comprehension_questions(procedure)
        
        # Recreate patient with conversation history
        patient = self._recreate_patient_from_log(conversation_log)
        
        # Get patient's answers
        answers = patient.answer_comprehension_questions(questions)
        
        # Extract topics from conversation for scoring
        topics_covered = self._extract_topics_from_conversation(conversation_log)
        
        # Score each answer
        scores = {}
        for question, answer in answers.items():
            score_result = self._score_comprehension_answer(
                question, 
                answer,
                topics_covered
            )
            scores[question] = score_result
        
        # Calculate average
        avg_score = sum(s["score"] for s in scores.values()) / len(scores) if scores else 0.0
        
        return {
            "conversation_id": Path(conversation_file).stem,
            "procedure": procedure,
            "patient_persona": conversation_log["metadata"]["patient_persona"]["name"],
            "comprehension_test": {
                "questions": questions,
                "answers": answers,
                "scores": scores,
                "average_score": avg_score,
                "pass_threshold": 0.7,
                "passed": avg_score >= 0.7
            },
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "conversation_turns": conversation_log["metadata"]["total_turns"]
            }
        }
    
    def _extract_topics_from_conversation(self, conversation_log: Dict) -> List[str]:
        """Extract topics mentioned in the conversation (simple extraction)."""
        topics = []
        for turn in conversation_log["conversation"]:
            # Simple: each chatbot response is a potential topic source
            topics.append(turn["chatbot_response"])
        return topics
    
    def evaluate_batch(self, conversation_dir: str, output_dir: str):
        """Evaluate all conversations in a directory."""
        conv_dir = Path(conversation_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for conv_file in conv_dir.glob("*.json"):
            print(f"Evaluating {conv_file.name}...")
            
            result = self.evaluate_conversation_file(str(conv_file))
            results.append(result)
            
            # Save individual result
            output_file = out_dir / f"{conv_file.stem}_comprehension.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save aggregate results
        summary = {
            "total_conversations": len(results),
            "average_comprehension_score": sum(r["comprehension_test"]["average_score"] for r in results) / len(results),
            "pass_rate": sum(1 for r in results if r["comprehension_test"]["passed"]) / len(results),
            "results": results
        }
        
        summary_file = out_dir / "comprehension_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Evaluated {len(results)} conversations")
        print(f"Average comprehension score: {summary['average_comprehension_score']:.2%}")
        print(f"Pass rate (>70%): {summary['pass_rate']:.2%}")
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from datetime import datetime
    
    # Example usage
    evaluator = ComprehensionEvaluator()
    
    # Option 1: Evaluate single conversation
    result = evaluator.evaluate_conversation_file(
        "data/conversations/conv_001.json"
    )
    evaluator.save_results(
        result,
        "data/evaluations/comprehension/conv_001_comprehension.json"
    )
    
    # Option 2: Evaluate all conversations
    summary = evaluator.evaluate_batch(
        conversation_dir="data/conversations",
        output_dir="data/evaluations/comprehension"
    )
    
    print("\n=== Comprehension Evaluation Complete ===")