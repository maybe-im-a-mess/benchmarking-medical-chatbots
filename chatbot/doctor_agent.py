from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator  # or HuggingFaceAPIGenerator
from haystack.utils import Secret
import os
from dotenv import load_dotenv
from chatbot.retrieval import DocumentRetriever

load_dotenv()


class DoctorAgent:
    """
    RAG based conversational agent that answers patient questions.
        - Answers patient questions using retrieved medical documents
        - Tracks conversation history
        - Asks mandatory consent questions at appropriate moments
        - Provides consistent, contextual responses
    """
    def __init__(self, document_store, model="gpt-5-mini"):
        self.document_store = document_store
        self.model = model
        
        # Conversation state
        self.conversation_history = []
        self.mandatory_questions_asked = set()
        self.topics_discussed = set()
        self.turn_count = 0
        
        # Initialize retriever
        self.retriever = DocumentRetriever(document_store, top_k=3)
        
        # Build RAG pipeline
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the complete RAG pipeline: retrieval + prompt + generation"""
        
        # System prompt for medical context
        prompt_template = """Du bist ein erfahrener medizinischer Assistent. Dein Ziel ist es, eine genaue, streng kontextbasierte Patientenaufklärung zu bieten und sicherzustellen, dass der Patient den medizinischen Eingriff vollständig versteht.

REGELN:
- Beantworte die Patientenfrage basierend auf dem medizinischen Kontext
- Sei einfühlsam und verwende verständliche Sprache
- Wenn du etwas bereits erklärt hast, verweise kurz darauf statt es zu wiederholen
- Halte deine Antwort fokussiert (max 300 Wörter)
- Sei präzise und medizinisch korrekt

WICHTIG - QUELLENANGABEN:
- Wenn du Informationen aus dem Kontext verwendest, füge eine Referenz ein: [Quelle X]
- Nutze die Quellennummern aus dem Kontext unten
- Beispiel: "Die Narkose dauert etwa 2-3 Stunden [Quelle 1]. Dabei werden Sie kontinuierlich überwacht [Quelle 2]."
- Gib NUR Quellen an, die du tatsächlich verwendet hast

MEDIZINISCHER KONTEXT MIT QUELLEN:
{% for doc in documents %}
[Quelle {{ loop.index }}]:
{{ doc.content }}
Dokument: {{ doc.meta.file_path if doc.meta.file_path else "Unbekannt" }}
---
{% endfor %}

{% if conversation_history %}
BISHERIGE KONVERSATION:
{{ conversation_history }}
{% endif %}

PATIENTENFRAGE: {{ question }}

ANTWORT (mit Quellenangaben [Quelle X]):"""
        
        pipeline = Pipeline()
        
        # Prompt builder
        pipeline.add_component("prompt_builder", PromptBuilder(
            template=prompt_template,
            required_variables=["documents", "question"]
        ))
        
        # LLM Generator
        pipeline.add_component("generator", OpenAIGenerator(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            model=self.model,
            generation_kwargs={
                "max_completion_tokens": 2000
            }
        ))
        
        pipeline.connect("prompt_builder", "generator")
        
        return pipeline
    
    def _extract_citations(self, response_text: str) -> dict:
        """
        Extract citation information from the response.
        
        Returns:
            dict with citation_count, cited_sources list, and response_with_citations
        """
        import re
        
        # Find all [Quelle X] patterns
        citation_pattern = r'\[Quelle (\d+)\]'
        citations = re.findall(citation_pattern, response_text)
        
        # Convert to integers and get unique citations
        cited_sources = sorted(list(set(int(c) for c in citations)))
        
        return {
            "citation_count": len(citations),
            "unique_sources_cited": len(cited_sources),
            "cited_source_indices": cited_sources,
            "response_with_citations": response_text
        }
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for prompt."""
        formatted = []
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = msg["role"]
            content = msg["content"][:200]  # Truncate long messages
            formatted.append(f"{role.upper()}: {content}")
        return "\n".join(formatted)
    
    def respond(self, patient_message):
        """
        Generate response to patient question.
        
        Args:
            patient_message: The patient's question
        
        Returns:
            dict with 'response', 'retrieved_chunks', 'citations', 'metadata'
        """
        self.turn_count += 1

        try:
            # Add patient message to history
            self.conversation_history.append({
                "role": "Patient",
                "content": patient_message
            })
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(patient_message)
            
            # Step 2: Format conversation history
            conv_history = self._format_conversation_history()
            
            # Step 3: Generate response with full context
            result = self.pipeline.run({
                "prompt_builder": {
                    "documents": retrieved_docs,
                    "question": patient_message,
                    "conversation_history": conv_history,
                }
            })
            
            response = result["generator"]["replies"][0]

            # Step 4: Extract citation information
            citation_info = self._extract_citations(response) 
            
            # Step 5: Map citations to actual documents
            citation_mapping = []
            for idx in citation_info["cited_source_indices"]:
                if idx <= len(retrieved_docs):
                    doc = retrieved_docs[idx - 1]  #0-indexed
                    citation_mapping.append({
                        "citation_index": idx,
                        "document_id": doc.id,
                        "file_path": doc.meta.get("file_path", "Unknown"),
                        "content_preview": doc.content[:100] + "...",
                        "score": doc.score
                    })
            
            # Add doctor response to history
            self.conversation_history.append({
                "role": "Doctor",
                "content": response
            })
            
            # Return response + metadata for evaluation
            return {
                "response": response,
                "retrieved_chunks": [
                    {
                        "content": doc.content,
                        "score": doc.score,
                        "meta": doc.meta,
                        "id": doc.id
                    }
                    for doc in retrieved_docs
                ],
                "citations": {
                    "total_citations": citation_info["citation_count"],
                    "unique_sources": citation_info["unique_sources_cited"],
                    "citation_mapping": citation_mapping
                },
                "metadata": {
                    "turn": self.turn_count,
                    "num_chunks": len(retrieved_docs),
                    "top_score": retrieved_docs[0].score if retrieved_docs else None,
                }
            }
        
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": "Entschuldigung, ich hatte ein technisches Problem. Können Sie Ihre Frage bitte wiederholen?",
                "retrieved_chunks": [],
                "citations": {
                    "total_citations": 0,
                    "unique_sources": 0,
                    "citation_mapping": []
                },
                "metadata": {
                    "turn": self.turn_count,
                    "num_chunks": 0,
                    "top_score": None,
                    "error": str(e)
                }
            }
        
    def reset(self):
        """Reset conversation state for new dialogue."""
        self.conversation_history = []
        self.mandatory_questions_asked = set()
        self.topics_discussed = set()
        self.turn_count = 0
    

if __name__ == "__main__":
    # Test the doctor agent
    from embeddings import load_document_store
    
    doc_store = load_document_store()
    doctor = DoctorAgent(doc_store)
    
    # Test conversation
    question = "Was sind die Risiken einer Narkose?"
    print(f"\nPatient: {question}\n")
    
    response = doctor.respond(question)
    
    print(f"Doctor: {response['response']}\n")
    print(f"--- Citation Analysis ---")
    print(f"Total citations: {response['citations']['total_citations']}")
    print(f"Unique sources cited: {response['citations']['unique_sources']}")
    print(f"\nCitation mapping:")
    for citation in response['citations']['citation_mapping']:
        print(f"  [Quelle {citation['citation_index']}]: {citation['file_path']} (score: {citation['score']:.3f})")