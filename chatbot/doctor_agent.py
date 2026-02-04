from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator  # or HuggingFaceAPIGenerator
from haystack.utils import Secret
import os
from dotenv import load_dotenv
from .retrieval import DocumentRetriever

load_dotenv()


class DoctorAgent:
    """
    RAG based conversational agent that answers patient questions.
    Uses DocumentRetriever or retrieval and an LLM for response generation.
    """
    def __init__(self, document_store, model="gpt-5-mini"):
        self.document_store = document_store
        self.model = model
        
        # Initialize retriever
        self.retriever = DocumentRetriever(document_store, top_k=3)
        
        # Build RAG pipeline
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the complete RAG pipeline: retrieval + prompt + generation"""
        
        # System prompt for medical context
        prompt_template = """Du bist ein medizinischer Fachmann, der Patienten über medizinische Eingriffe aufklärt.
Beantworte die Patientenfrage kurz und klar basierend auf diesem Kontext:

Kontext:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Patientenfrage: {{ question }}

Antwort:"""
        
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
    
    def respond(self, patient_message, conversation_history=None):
        """
        Generate response to patient question.
        
        Args:
            patient_message: The patient's question
            conversation_history: Optional list of previous messages for context
        
        Returns:
            dict with 'response' and 'retrieved_chunks' (for evaluation)
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(patient_message)
        
        # Step 2: Generate response using retrieved context
        result = self.pipeline.run({
            "prompt_builder": {
                "documents": retrieved_docs,
                "question": patient_message
            }
        })
        
        # Return response + metadata for evaluation
        return {
            "response": result["generator"]["replies"][0],
            "retrieved_chunks": retrieved_docs,  # For sender coverage evaluation
            "metadata": {
                "num_chunks": len(retrieved_docs),
                "top_score": retrieved_docs[0].score if retrieved_docs else None
            }
        }
    
    def respond_simple(self, patient_message):
        """Simplified interface that just returns the text response."""
        return self.respond(patient_message)["response"]
    

if __name__ == "__main__":
    # Test the doctor agent
    from embeddings import load_document_store
    
    doc_store = load_document_store()
    doctor = DoctorAgent(doc_store)
    
    # Test conversation
    question = "Was sind die Risiken einer Narkose?"
    response = doctor.respond(question)
    
    print(f"Patient: {question}\n")
    print(f"Doctor: {response['response']}\n")
    print(f"Retrieved {response['metadata']['num_chunks']} chunks")
    print(f"Top relevance score: {response['metadata']['top_score']:.4f}")