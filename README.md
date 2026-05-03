# Salaahkar 2.0 📚🤖

**Salaahkar** (Hindi for *Advisor*) is a high-performance, local Agentic RAG (Retrieval-Augmented Generation) study assistant. Built specifically for college students, it allows you to chat with your theory notes, textbooks, and syllabus PDFs entirely offline.

Designed for **Fedora Linux**, it leverages the power of **Llama 3.2** and **LangChain** to ensure your data stays private and your tutoring stays accurate.

---

## 🚀 Key Features

*   **Privacy First:** Everything runs locally. No data ever leaves your machine.
*   **Context-Aware Tutoring:** Doesn't just give generic AI answers; it cites your specific class notes.
*   **Agentic Reasoning:** Uses query expansion and self-correction to handle complex academic questions.
*   **Fast Ingestion:** Batch process entire directories of PDFs into a persistent **ChromaDB** vector store.
*   **Modern UI:** A clean, ChatGPT-style interface built with **Streamlit**.

---

## 🛠️ Tech Stack

*   **LLM:** Ollama (Llama 3.2:3b)
*   **Embeddings:** `nomic-embed-text`
*   **Orchestration:** LangChain / LangChain-Classic
*   **Vector Database:** ChromaDB
*   **Frontend:** Streamlit
*   **Environment:** Python 3.14 on Fedora

---

