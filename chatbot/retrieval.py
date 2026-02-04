from haystack import Pipeline
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import Secret

import os
from dotenv import load_dotenv

load_dotenv()

class DocumentRetriever:
    """
    Retrieval component to fetch relevant document chunks based on user queries.
    """
    def __init__(self, document_store, top_k=3):
        self.document_store = document_store
        self.top_k = top_k
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        pipeline = Pipeline()

        pipeline.add_component("text_embedder", HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                              api_params={"model": "Qwen/Qwen3-Embedding-8B"},
                                              token=Secret.from_token(os.getenv("HF_TOKEN"))))
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=self.top_k))
        
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        return pipeline
    
    def retrieve(self,query):
        """
        Retrieve relevant document chunks given a query.
        Returns:
            list of Documents with content and metadata
        """
        result = self.pipeline.run({"text_embedder": {"text": query}})
        return result['retriever']['documents']
    
    def retrieve_with_scores(self, query):
        """
        Retrieve relevant documents with their relevance scores.
        Returns:
            list of tuples (Document, score)
        """
        documents = self.retrieve(query)
        return [
            {
                "content": doc.content,
                "score": doc.score
            }
            for doc in documents
        ]



if __name__ == "__main__":
    # Test the retriever
    from embeddings import load_document_store
    document_store = load_document_store()
    retriever = DocumentRetriever(document_store, top_k=3)

    query = "Wofür wird die Narkose bei einer Operation eingesetzt?"
    results = retriever.retrieve_with_scores(query)

    print(f"Query: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i} (Score: {result['score']:.4f}):")
        print(f"{result['content'][:200]}...\n")
