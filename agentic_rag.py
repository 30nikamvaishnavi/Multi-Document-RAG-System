"""
Agentic RAG layer built with LangGraph.

The agent loop:
1. question   -> user query
2. rewrite    -> rewrite query for better retrieval (if needed)
3. retrieve  -> fetch relevant chunks
4. generate   -> produce an answer from retrieved context
5. ground_check -> verify the answer is grounded in the retrieved context
6. If grounded -> return answer; else -> rewrite and retry (up to max_retries)
"""
import os
from typing import List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from rag_engine import RAGEngine, response_to_text
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    grounded: Optional[bool]
    retry_count: int
    retrieval_used: bool


# --- LLM for agent decisions ------------------------------------------------
def _make_llm(
    model: str = "gemini-3.8-flash",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("Set GOOGLE_API_KEY in your .env file before starting the agent.")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


# --- Prompts ----------------------------------------------------------------
ROUTE_PROMPT = PromptTemplate(
    template="""Decide whether answering this user message requires searching their uploaded documents.
Answer "yes" when the user asks about document contents, uploaded files, or asks a question
that should be answered from those documents. Answer "no" for greetings, general conversation,
or questions that do not depend on the uploaded documents.

User message: {question}

Return only "yes" or "no".""",
    input_variables=["question"],
)

REWRITE_PROMPT = PromptTemplate(
    template="""You are a query rewriting assistant. Given a user question and possibly some prior context,
rewrite the question to be more specific and self-contained so that a document retrieval system can find the best passages.

If the question is already clear and specific, return it unchanged.

Original question: {question}
{context_hint}

Return ONLY the rewritten question, no extra text.""",
    input_variables=["question", "context_hint"],
)

GROUND_CHECK_PROMPT = PromptTemplate(
    template="""You are a strict fact-checker. Determine whether the following answer is FULLY grounded in the provided context.

Context:
{context}

Answer:
{answer}

Rules:
- The answer is grounded ONLY if every factual claim in the answer can be directly supported by the context.
- If the answer contains any claim, number, name, or detail NOT present in the context, it is NOT grounded.
- If the answer says it cannot be determined, that counts as grounded.

Return ONLY one word: "yes" or "no".""",
    input_variables=["context", "answer"],
)

DIRECT_ANSWER_PROMPT = PromptTemplate(
    template="""Respond helpfully to the user's general message. Do not claim that you searched
or used their uploaded documents.

User message: {question}""",
    input_variables=["question"],
)


# --- Agentic RAG graph -------------------------------------------------------
class AgenticRAG:
    """LangGraph agent that routes, rewrites, retrieves, and checks grounding."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "rag_documents",
        model: str = "gemini-3.8-flash",
        k: int = 5,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        chroma_client: Optional[Any] = None,
    ):
        self.rag = RAGEngine(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model=model,
            k=k,
            api_key=api_key,
            chroma_client=chroma_client,
        )
        self.llm = _make_llm(model=model, api_key=api_key)
        self.max_retries = max_retries
        self.graph = self._build_graph()

    # --- Node implementations ------------------------------------------------
    def _route_node(self, state: AgentState) -> Dict[str, Any]:
        """Decide whether this question needs evidence from the user's documents."""
        decision = self.llm.invoke(
            ROUTE_PROMPT.format(question=state["question"])
        )
        decision = response_to_text(decision).strip().lower()
        return {"retrieval_used": not decision.startswith("no")}

    def _should_retrieve(self, state: AgentState) -> str:
        return "retrieve" if state["retrieval_used"] else "direct"

    def _rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite the query if it's the first pass, or refine after a failed grounding check."""
        if state["retry_count"] == 0:
            question = state["question"]
            context_hint = ""
        else:
            question = state["rewritten_question"]
            context_hint = f"Previous attempt failed grounding check. Previous answer: {state['answer']}"

        prompt = REWRITE_PROMPT.format(question=question, context_hint=context_hint)
        rewritten = response_to_text(self.llm.invoke(prompt)).strip()
        return {"rewritten_question": rewritten or state["question"]}

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant chunks using the (possibly rewritten) query."""
        query = state["rewritten_question"] or state["question"]
        chunks = self.rag.retrieve(query, k=self.rag.k)
        return {"retrieved_chunks": chunks}

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate an answer from the retrieved context."""
        context = self.rag._format_retrieved(state["retrieved_chunks"])
        answer = self.rag.generate(state["rewritten_question"] or state["question"], context)
        return {"answer": answer}

    def _ground_check_node(self, state: AgentState) -> Dict[str, Any]:
        """Check whether the answer is grounded in the retrieved context."""
        context = self.rag._format_retrieved(state["retrieved_chunks"])
        prompt = GROUND_CHECK_PROMPT.format(context=context, answer=state["answer"])
        result = response_to_text(self.llm.invoke(prompt)).strip().lower()
        grounded = result.startswith("yes")
        return {"grounded": grounded}

    def _direct_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Answer non-document questions without running retrieval."""
        prompt = DIRECT_ANSWER_PROMPT.format(question=state["question"])
        answer = response_to_text(self.llm.invoke(prompt)).strip()
        return {"answer": answer, "grounded": None}

    def _record_retry_node(self, state: AgentState) -> Dict[str, Any]:
        return {"retry_count": state["retry_count"] + 1}

    def _should_retry(self, state: AgentState) -> str:
        """Retry after a failed grounding check, up to the configured retry limit."""
        if state["grounded"]:
            return "end"
        if state["retry_count"] >= self.max_retries:
            return "end"
        return "retry"

    # --- Graph construction --------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("route", self._route_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("ground_check", self._ground_check_node)
        builder.add_node("record_retry", self._record_retry_node)
        builder.add_node("direct_answer", self._direct_answer_node)

        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            self._should_retrieve,
            {"retrieve": "rewrite", "direct": "direct_answer"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "ground_check")
        builder.add_conditional_edges(
            "ground_check",
            self._should_retry,
            {"retry": "record_retry", "end": END},
        )
        builder.add_edge("record_retry", "rewrite")
        builder.add_edge("direct_answer", END)

        return builder.compile()

    # --- Public API ----------------------------------------------------------
    def invoke(self, question: str) -> Dict[str, Any]:
        """Run the agentic RAG loop for a single question."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        initial_state: AgentState = {
            "question": question.strip(),
            "rewritten_question": "",
            "retrieved_chunks": [],
            "answer": "",
            "grounded": None,
            "retry_count": 0,
            "retrieval_used": False,
        }
        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state["answer"],
            "rewritten_question": final_state["rewritten_question"],
            "sources": final_state["retrieved_chunks"],
            "grounded": final_state["grounded"],
            "retry_count": final_state["retry_count"],
            "retrieval_used": final_state["retrieval_used"],
        }