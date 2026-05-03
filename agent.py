"""
Salaahkar - RAG Agent
Connects to the local ChromaDB, retrieves relevant chunks,
and answers questions using llama3.2:3b via a LangChain retrieval chain.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
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


def build_chain():
    """Build and return the retrieval chain."""
    # Connect to persisted ChromaDB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    # Retrieve top 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM
    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    return retrieval_chain
