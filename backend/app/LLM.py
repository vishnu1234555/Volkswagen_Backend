"""
VOLKSWAGEN PRODUCT INTELLIGENCE - RAG INFERENCE ENGINE
VERSION: 1.0.0
DESCRIPTION: Production-grade retrieval-augmented generation using Groq Llama-3 and Qdrant.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_QDRANT_URL = (
    "https://0c6c9160-9894-4ee5-b55a-74979039be06.europe-west3-0.gcp.cloud.qdrant.io"
)


class VolkswagenRAG:
    """
    Main controller for the Volkswagen RAG pipeline.
    Handles semantic retrieval from Qdrant and synthesis via Groq.
    """

    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        collection_name = os.getenv("COLLECTION_NAME", "volkswagen_newsroom")
        llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is required (set in environment or .env file at repo root)."
            )
        if qdrant_api_key is None:
            raise ValueError(
                "QDRANT_API_KEY is required (set in environment or .env file)."
            )
        if groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY is required (set in environment or .env file)."
            )

        self.collection_name = collection_name
        self.llm_model = llm_model

        ollama_base = os.getenv("OLLAMA_BASE_URL")
        emb_kwargs: dict = {"model": "nomic-embed-text"}
        if ollama_base:
            emb_kwargs["base_url"] = ollama_base
        self.embeddings = OllamaEmbeddings(**emb_kwargs)

        self.q_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.vector_store = QdrantVectorStore(
            client=self.q_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        self.groq_client = Groq(api_key=groq_api_key)

    def retrieve_context(self, user_query: str) -> str:
        """Return concatenated top-k chunk text for UI / debugging."""
        docs = self.vector_store.similarity_search(user_query, k=4)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, user_query: str) -> str:
        """Full RAG cycle: search -> augment -> generate."""
        retrieved_context = self.retrieve_context(user_query)

        system_prompt = f"""
        ROLE: Professional Volkswagen Product Specialist.
        TASK: Answer the question using ONLY the provided official technical context.
        CONTEXT:
        {retrieved_context}
        """

        completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            model=self.llm_model,
            temperature=0.1,
        )
        return completion.choices[0].message.content


if __name__ == "__main__":
    rag_engine = VolkswagenRAG()
    print("\n[VWRAG-v1] Engine Initialized. Type 'exit' to terminate session.")

    while True:
        user_input = input("\nPLEASE ENTER PROMPT >> ").strip()

        if user_input.lower() in ("exit", "quit"):
            logger.info("Session terminated by user.")
            break

        if user_input:
            try:
                print("\n[SYSTEM] Retrieving context and generating specialist response...")
                response = rag_engine.generate_response(user_input)
                print(f"\nVW_SPECIALIST: {response}\n")
                print("-" * 50)
            except Exception as e:
                logger.error(f"Pipeline Error: {e}")
