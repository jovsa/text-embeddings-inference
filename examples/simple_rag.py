#!/usr/bin/env python3
"""
Simple RAG Example with Text Embeddings Inference
=================================================

This script demonstrates how to build a simple Retrieval-Augmented Generation (RAG)
pipeline using TEI for the "Retrieval" part.

Flow:
1. INDEXING: Embed a small knowledge base using TEI.
2. RETRIEVAL: Embed the user's query and find the most similar doc.
3. GENERATION: Construct a prompt for an LLM (Mocked).

Usage:
    python examples/simple_rag.py
"""

import requests
import numpy as np
from typing import List, Dict

TEI_URL = "http://127.0.0.1:8080"


class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs: List[str]):
        """Embed documents and store them."""
        print(f"🔄 Indexing {len(docs)} documents...")
        
        # Call TEI to get embeddings
        response = requests.post(
            f"{TEI_URL}/embed",
            json={"inputs": docs},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to embed documents: {response.text}")
            
        new_embeddings = response.json()
        
        self.documents.extend(docs)
        if not self.embeddings:
            self.embeddings = np.array(new_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        print("✅ Indexing complete.\n")

    def retrieve(self, query: str, k: int = 1) -> List[Dict]:
        """Find top-k most similar documents."""
        # Embed the query
        response = requests.post(
            f"{TEI_URL}/embed",
            json={"inputs": query},
            headers={"Content-Type": "application/json"}
        )
        query_embedding = np.array(response.json()[0])

        # Calculate cosine similarity
        # Sim(A, B) = dot(A, B) / (norm(A) * norm(B))
        # Note: BGE models often produce normalized embeddings, but we'll normalize to be safe
        norm_query = np.linalg.norm(query_embedding)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        
        scores = np.dot(self.embeddings, query_embedding) / (norm_docs * norm_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "score": scores[idx],
                "text": self.documents[idx]
            })
            
        return results


def mock_llm_generate(prompt: str) -> str:
    """
    In a real app, this would call OpenAI, Anthropic, or a local LLM.
    Here strictly for demonstration.
    """
    return "Based on the context provided, I can explain that..."


def main():
    # 1. Define Knowledge Base
    knowledge_base = [
        "Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models.",
        "TEI enables high-performance extraction for popular models, including FlagEmbedding, Ember, GTE and E5.",
        "Features include Token based dynamic batching and Flash Attention optimization.",
        "TEI supports distributed tracing with Open Telemetry and Prometheus metrics.",
        "You can use Docker to deploy TEI on CPU, Turing, Ampere, Ada Lovelace and Hopper architectures."
    ]

    # 2. Indexing
    store = VectorStore()
    try:
        store.add_documents(knowledge_base)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to TEI at {TEI_URL}")
        print("   Please start the server first.")
        return

    # 3. User Query
    query = "What architectures does TEI support?"
    print(f"❓ Query: {query}")

    # 4. Retrieval
    results = store.retrieve(query, k=2)
    
    print("\n🔍 Retrieved Context:")
    for i, res in enumerate(results):
        print(f"   {i+1}. [Score: {res['score']:.4f}] {res['text']}")

    # 5. Generation (Prompt Construction)
    context_str = "\n".join([f"- {r['text']}" for r in results])
    prompt = f"""
Use the following context to answer the question.

Context:
{context_str}

Question: {query}

Answer:"""

    print("\n🤖 Constructed Prompt for LLM:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # 6. Mock Generation
    print("\n(This prompt would now be sent to your LLM...)")


if __name__ == "__main__":
    main()
