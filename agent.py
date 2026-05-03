"""
Salaahkar - Phase 1 RAG Agent
Hybrid search (BM25 + ChromaDB vector) with multi-query expansion.
Returns both the generated answer and the source documents used.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


DB_DIR = "./salaahkar_db"

SYSTEM_PROMPT = (
    "You are **Salaahkar**, an expert academic tutor. "
    "Your role is to help students understand complex academic theory by "
    "answering their questions based **only** on the context retrieved from "
    "their study notes.\n\n"
    "Rules you must follow:\n"
    "- Answer using **bullet points** and **simple explanations**.\n"
    "- If the context does not contain enough information to answer, "
    "say so honestly — do NOT make up information.\n"
    "- Keep your language clear, concise, and student-friendly.\n\n"
    "Context:\n{context}"
)


def setup_salaahkar_agent(temperature: float = 0.2, top_k: int = 5):
    """
    Build and return the Phase 1 retrieval chain.

    Args:
        temperature: LLM sampling temperature (0.0 – 1.0).
        top_k:       Number of chunks each retriever returns.

    Returns:
        A retrieval chain whose .invoke() returns both 'answer' and 'context'.
    """
    # ── Embeddings & Vector Store ────────────────────────────────────────────
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    # ── Retriever 1: Semantic (ChromaDB) ────────────────────────────────────
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # ── Retriever 2: Keyword (BM25) ─────────────────────────────────────────
    # Load all stored documents so BM25 can build its index
    all_docs = vectorstore.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=top_k)

    # ── Hybrid: Ensemble (50/50 weighting) ──────────────────────────────────
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # ── Multi-Query Expansion ───────────────────────────────────────────────
    llm = ChatOllama(model="llama3.2:3b", temperature=temperature)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    # ── Answer-Generation Chain ─────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(multi_query_retriever, question_answer_chain)

    return retrieval_chain
