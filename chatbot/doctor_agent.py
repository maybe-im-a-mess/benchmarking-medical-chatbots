from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
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
        self.turn_count = 0
        self.history_max_messages = 20
        
        # Initialize retriever
        self.retriever = DocumentRetriever(document_store, top_k=3)
        
        # Build generator
        self.generator = self._build_generator()
        self.core_rules = (
            "Du bist ein erfahrener medizinischer Assistent. Dein Ziel ist es, eine genaue, "
            "streng kontextbasierte Patientenaufklärung zu bieten und sicherzustellen, dass "
            "der Patient den medizinischen Eingriff vollständig versteht.\n\n"
            "REGELN:\n"
            "- Beantworte die Patientenfrage basierend auf dem medizinischen Kontext\n"
            "- Sei einfühlsam und verwende verständliche Sprache\n"
            "- Halte deine Antwort fokussiert (max 300 Wörter)\n"
            "- Sei präzise und medizinisch korrekt\n\n"
            "ZITIERREGELN:\n"
            "1. Bevorzuge Informationen aus dem 'MEDIZINISCHER KONTEXT' und zitiere sie mit [Quelle X]\n"
            "2. Für allgemeine medizinische Fragen (z.B. 'Tut es weh?', 'Wie lange dauert es?') "
            "darfst du OHNE Zitat antworten, wenn die Info im Kontext fehlt\n"
            "3. Für spezifische Details (Risiken, Ablauf, Komplikationen) MUSS eine [Quelle X] vorhanden sein\n"
            "4. Erfinde KEINE spezifischen medizinischen Fakten\n"
            "5. Bei fehlenden Informationen sage: 'Dazu habe ich keine Details in den Unterlagen. "
            "Der Chatbot kann das nicht weiter ausführen.'"
        )
    
    def _build_generator(self):
        """Build the chat generator."""
        return OpenAIChatGenerator(
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            model=self.model,
        )

    def _format_documents(self, documents) -> str:
        parts = ["MEDIZINISCHER KONTEXT:"]
        for idx, doc in enumerate(documents, 1):
            parts.append("---")
            parts.append(f"[Quelle {idx}]")
            parts.append(f"(Datei: {doc.meta.get('file_path', 'Unknown')})")
            parts.append("INHALT:")
            parts.append(doc.content)
            parts.append("---")
        return "\n".join(parts)

    def _build_messages(self, patient_message: str, retrieved_docs, extra_system_instructions: str) -> list:
        messages = []

        # Layer 1 — core identity
        messages.append(ChatMessage.from_system(self.core_rules))

        # Layer 2 — experiment condition
        if extra_system_instructions:
            messages.append(ChatMessage.from_system(extra_system_instructions))
            
        # Layer 3 — RAG context
        messages.append(ChatMessage.from_system(self._format_documents(retrieved_docs)))

        # Layer 4 — conversation history
        for msg in self.conversation_history[-self.history_max_messages:]:
            if msg["role"] == "user":
                messages.append(ChatMessage.from_user(msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(ChatMessage.from_assistant(msg["content"]))

        # Layer 5 — new question
        messages.append(ChatMessage.from_user(patient_message))

        return messages
    
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
    
    def respond(self, patient_message, extra_system_instructions: str = ""):
        """
        Generate response to patient question.
        
        Args:
            patient_message: The patient's question
        
        Returns:
            dict with 'response', 'retrieved_chunks', 'citations', 'metadata'
        """
        self.turn_count += 1

        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(patient_message)
            
            # Step 2: Build layered messages
            messages = self._build_messages(patient_message, retrieved_docs, extra_system_instructions)

            # Step 3: Generate response with full context
            result = self.generator.run(messages=messages)
            response_message = result["replies"][0]
            response_text = response_message.text

            # Step 4: Extract citation information
            citation_info = self._extract_citations(response_text) 
            
            # Step 5: Map citations to actual documents
            citation_mapping = []
            for idx in citation_info["cited_source_indices"]:
                if idx <= len(retrieved_docs):
                    doc = retrieved_docs[idx - 1]
                    citation_mapping.append({
                        "citation_index": idx,
                        "document_id": doc.id,
                        "file_path": doc.meta.get("file_path", "Unknown"),
                        "score": doc.score
                    })
            
            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": patient_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Return response + metadata for evaluation
            return {
                "response": response_text,
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
                    "citation_missing": citation_info["citation_count"] == 0,
                    "prompt_messages": [m.to_dict() for m in messages]
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
        self.turn_count = 0
    

if __name__ == "__main__":
    # Test the chatbot agent
    from embeddings import load_document_store
    
    doc_store = load_document_store()
    chatbot = DoctorAgent(doc_store)
    
    # Test conversation
    question = "Was sind die Risiken einer Narkose?"
    print(f"\nPatient: {question}\n")
    
    response = chatbot.respond(question)
    
    print(f"Chatbot: {response['response']}\n")
    print(f"--- Citation Analysis ---")
    print(f"Total citations: {response['citations']['total_citations']}")
    print(f"Unique sources cited: {response['citations']['unique_sources']}")
    print(f"\nCitation mapping:")
    for citation in response['citations']['citation_mapping']:
        print(f"  [Quelle {citation['citation_index']}]: {citation['file_path']} (score: {citation['score']:.3f})")