"""
Vector store module using ChromaDB.
Provides persistence and similarity search with metadata.
"""
import os
from hashlib import sha256
import time
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Wraps ChromaDB for document storage and retrieval."""
    EMBEDDING_BATCH_SIZE = 80
    RATE_LIMIT_WAIT_SECONDS = 60
    MAX_RATE_LIMIT_RETRIES = 3

    @staticmethod
    def has_documents(
        persist_dir: str = "./data/chroma",
        collection_name: str = "rag_documents",
    ) -> bool:
        """Check whether a persisted collection contains indexed chunks."""
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_or_create_collection(collection_name)
            return (
                collection.count() > 0
                and bool((collection.metadata or {}).get("index_complete", False))
            )
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "rag_documents",
        embedding_model: str = "models/gemini-embedding-001",
        api_key: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("Set GOOGLE_API_KEY in your .env file before building an index.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
        )
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Resume indexing missing chunks and mark the index complete on success."""
        if not docs:
            return

        ids = []
        texts = []
        metadatas = []
        for document in docs:
            metadata = {
                "source": str(document.get("source", "unknown")),
                "page": int(document.get("page", 0)),
                "location": str(
                    document.get("location", f"Page {document.get('page', 0)}")
                ),
            }
            text = str(document["text"])
            ids.append(
                sha256(
                    f"{metadata['source']}\0{metadata['page']}\0{text}".encode("utf-8")
                ).hexdigest()
            )
            texts.append(text)
            metadatas.append(metadata)

        collection = self.client.get_or_create_collection(self.collection_name)
        collection.modify(metadata={**(collection.metadata or {}), "index_complete": False})

        existing_ids = set(collection.get(ids=ids, include=[])["ids"])
        missing_indexes = [i for i, document_id in enumerate(ids) if document_id not in existing_ids]
        for start in range(0, len(missing_indexes), self.EMBEDDING_BATCH_SIZE):
            batch_indexes = missing_indexes[start : start + self.EMBEDDING_BATCH_SIZE]
            batch_ids = [ids[i] for i in batch_indexes]
            batch_texts = [texts[i] for i in batch_indexes]
            batch_metadata = [metadatas[i] for i in batch_indexes]

            for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    embeddings = self.embeddings.embed_documents(batch_texts)
                    break
                except Exception as error:
                    message = str(error).upper()
                    rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                    if not rate_limited or attempt == self.MAX_RATE_LIMIT_RETRIES:
                        raise
                    time.sleep(self.RATE_LIMIT_WAIT_SECONDS)

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadata,
            )

        expected_ids = set(ids)
        count = collection.count()
        stored_ids = []
        for offset in range(0, count, 1000):
            stored_ids.extend(
                collection.get(limit=min(1000, count - offset), offset=offset, include=[])[
                    "ids"
                ]
            )
        stale_ids = [document_id for document_id in stored_ids if document_id not in expected_ids]
        for start in range(0, len(stale_ids), 1000):
            collection.delete(ids=stale_ids[start : start + 1000])

        collection.modify(
            metadata={**(collection.metadata or {}), "index_complete": True}
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search and return chunks with metadata."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        collection = self.client.get_or_create_collection(self.collection_name)
        count = collection.count()
        if count == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "page": results["metadatas"][0][i].get("page", 0),
                "distance": results["distances"][0][i],
            })
        return chunks

    def clear(self) -> None:
        """Delete all documents from the collection."""
        collection = self.client.get_or_create_collection(self.collection_name)
        collection.modify(
            metadata={**(collection.metadata or {}), "index_complete": False}
        )
        while collection.count():
            ids = collection.get(limit=min(collection.count(), 1000), include=[])["ids"]
            if not ids:
                raise RuntimeError(
                    f"Could not clear Chroma collection {self.collection_name!r}."
                )
            collection.delete(ids=ids)

    def close(self) -> None:
        """Release the persistent Chroma client when it is no longer needed."""
        close = getattr(self.client, "close", None)
        if close is not None:
            close()