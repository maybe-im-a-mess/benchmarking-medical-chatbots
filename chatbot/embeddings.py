from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret

import json

import os
from dotenv import load_dotenv


load_dotenv()

def create_document_store(documents_dir="data/raw_md_files", store_path="data/vector_store/embedded_docs.json"):
    """
    Create a store for documents with embeddings.
    Populate it with the embeddings of markdown files found in documents_dir.
    """
    # Get the files
    file_names = []
    for file in os.listdir(documents_dir):
        if file.endswith(".md"):
            file_names.append(os.path.join(documents_dir, file))

    if not file_names:
        raise ValueError(f"No .md files found in {documents_dir}")

    # Initialize a document store
    document_store = InMemoryDocumentStore()

    # Build a pipeline for processing and embedding documents
    pipeline = Pipeline()

    pipeline.add_component("converter", MarkdownToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=100))
    pipeline.add_component("embedder", HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                              api_params={"model": "Qwen/Qwen3-Embedding-8B"},
                                              token=Secret.from_token(os.getenv("HF_TOKEN"))))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # Run the pipeline to process and embed documents
    print(f"Processing and embedding {len(file_names)} documents...")
    pipeline.run({"converter": {"sources": file_names}})

    # Save the document store
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    all_docs = list(document_store.storage.values())
    docs_data = []
    for doc in all_docs:
        docs_data.append({
            "id": doc.id,
            "content": doc.content,
            "embedding": doc.embedding,
            "meta": doc.meta
        })
    
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)
    
    print(f"Document store saved to {store_path}")
    return document_store

def load_document_store(store_path="data/vector_store/embedded_docs.json"):
    """
    Load a document store with embeddings.
    """
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Document store not found at {store_path}. Run create_document_store() first.")

    with open(store_path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    
    # Recreate document store
    document_store = InMemoryDocumentStore()
    
    # Recreate documents
    documents = []
    for doc_data in docs_data:
        doc = Document(
            id=doc_data["id"],
            content=doc_data["content"],
            embedding=doc_data["embedding"],
            meta=doc_data["meta"]
        )
        documents.append(doc)
    
    # Write documents to store
    document_store.write_documents(documents)
    return document_store

if __name__ == "__main__":
    # This script is only run once to create the embeddings
    create_document_store()










