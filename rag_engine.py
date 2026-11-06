"""
Basic RAG engine.
Retrieves relevant chunks and generates answers with source citations.
"""
import os
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


def response_to_text(response: Any) -> str:
    """Extract text from either string or block-based LangChain message content."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        if text_parts:
            return "".join(text_parts)
        raise ValueError("The language model returned no text content.")
    raise TypeError(
        f"Expected language-model content to be text or text blocks, got {type(content).__name__}."
    )


class RAGEngine:
    """Retrieval-Augmented Generation engine with citations."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "rag_documents",
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        k: int = 5,
        api_key: Optional[str] = None,
        chroma_client: Optional[Any] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.k = k

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("Set GOOGLE_API_KEY in your .env file before asking questions.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=api_key
        )

        chroma_options: Dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_function": self.embeddings,
        }
        if chroma_client is None:
            chroma_options["persist_directory"] = persist_dir
        else:
            chroma_options["client"] = chroma_client
        self.vectorstore = Chroma(**chroma_options)

        self.prompt = PromptTemplate(
            template=self._prompt_template(),
            input_variables=["context", "question"],
        )

    def _prompt_template(self) -> str:
        return """You are a helpful assistant that answers questions using ONLY the provided document excerpts.
You must cite the source document and page number for every claim you make.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the provided context.
- If the answer is not in the context, say so clearly.
- Cite every claim with the source document name and page number in brackets, e.g. [source.pdf, page 3].
- Be concise and accurate.
"""

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a context string with citations."""
        formatted = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", 0)
            location = d.metadata.get("location", f"Page {page}")
            formatted.append(f"[{source}, {location}]\n{d.page_content}")
        return "\n\n".join(formatted)

    def _format_retrieved(self, chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for c in chunks:
            location = c.get("location", f"Page {c['page']}")
            formatted.append(f"[{c['source']}, {location}]\n{c['text']}")
        return "\n\n".join(formatted)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        k = self.k if k is None else k
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        results = self.vectorstore.similarity_search(query, k=k)
        chunks = []
        for d in results:
            chunks.append({
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", 0),
                "location": d.metadata.get(
                    "location", f"Page {d.metadata.get('page', 0)}"
                ),
            })
        return chunks

    def generate(self, query: str, context: str) -> str:
        """Generate an answer given a query and formatted context."""
        formatted_prompt = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)
        return response_to_text(response)

    def query(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve -> generate -> return answer + sources."""
        chunks = self.retrieve(query, k=k)
        context = self._format_retrieved(chunks)
        answer = self.generate(query, context)
        return {
            "answer": answer,
            "sources": chunks,
            "context": context,
        }