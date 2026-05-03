# Salaahkar 2.0 📚🤖

**Salaahkar** (Hindi for *Advisor*) is a high-performance, local Agentic RAG (Retrieval-Augmented Generation) study assistant designed for university students. It allows you to chat with theory notes, textbooks, and YouTube lectures entirely offline, ensuring 100% privacy and context-aware tutoring.

Built for **Fedora Linux** on a **Dell G15**, this project leverages **Llama 3.2** and **LangChain** to provide a specialized research environment for the AIML curriculum at USAR.

---

## 🚀 Key Features

*   **Hybrid Search Engine:** Combines semantic vector search (ChromaDB) with keyword-based BM25 retrieval for surgical accuracy.
*   **Multi-Modal Learning:** Ingests PDFs, handwritten notes (via Tesseract OCR), and YouTube video transcripts.
*   **Privacy First:** Everything runs locally via **Ollama**. No data leaves your machine.
*   **Agentic Reasoning:** Uses Multi-Query expansion to "understand" lazy prompts and retrieve better context.
*   **Academic UI:** A Streamlit-based dashboard with real-time controls for LLM temperature and retrieval depth.

---

## 🛠️ Tech Stack

*   **LLM:** Ollama (Llama 3.2:3b)
*   **Embeddings:** `nomic-embed-text`
*   **Vector Database:** ChromaDB
*   **OCR Engine:** Tesseract (Fedora)
*   **Orchestration:** LangChain / LangChain-Classic
*   **Frontend:** Streamlit
*   **Environment:** Python 3.14 on Fedora

---

## 📦 Installation & Setup

### 1. System Dependencies (Fedora)
```bash
sudo dnf install tesseract tesseract-langpack-eng
```

### 2. Ollama Setup
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Python Environment
```bash
git clone [https://github.com/KushagrIttan/Salaahkar.git](https://github.com/KushagrIttan/Salaahkar.git)
cd Salaahkar
python3.14 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## 📖 Usage

### Step 1: Ingest Data
Place your PDFs or images in the `/data` folder and run:
```bash
python3.14 ingest.py
```

### Step 2: Launch Salaahkar
```bash
streamlit run app.py
```

### Step 3: Add Lectures on the Fly
Use the sidebar in the UI to paste YouTube URLs. The transcripts will be indexed into your persistent database automatically.

---

## 🗺️ Roadmap & Evolution

### ✅ Phase 1: The Intelligence Upgrade
- [x] Hybrid Search (Vector + BM25)
- [x] Multi-Query Expansion for complex retrieval
- [x] Citation UI with expandable source transparency

### ✅ Phase 2: The Multi-Modal Upgrade
- [x] OCR support for handwritten notes and whiteboard photos
- [x] YouTube transcript ingestion via `pytubefix`
- [x] Dynamic Sidebar for session-based parameter tuning

### ⏳ Phase 3: Memory & Contextual Logic (Current)
- [ ] **SQLite Integration:** Long-term chat history persistence between sessions.
- [ ] **Metadata Filtering:** Subject-specific toggles (e.g., "Only search DBMS notes").
- [ ] **LaTeX Support:** Specialized math parsing for AIML formulas using `Nougat`.

### 🚀 Phase 4: The Professional Agent
- [ ] **Anki Export:** Automated flashcard generation for active recall.
- [ ] **Study Guide Generator:** Multi-document synthesis for exam preparation.
- [ ] **Native Fedora Deployment:** Packaging as a `.rpm` or Flatpak.

---

## ⚖️ License
MIT License. Created by Kushagr Ittan for the student community. ❤️
```