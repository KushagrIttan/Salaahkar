"""
Salaahkar - Phase 2 Multi-Modal Ingestion Script
Loads PDFs and images from ./data, grabs YouTube transcripts on demand,
splits into chunks, embeds with nomic-embed-text, and stores in ChromaDB.
"""

import os
import glob

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredImageLoader,
    YoutubeLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "./data"
DB_DIR = "./salaahkar_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Supported file extensions and their loaders
FILE_LOADERS = {
    ".pdf": PyPDFLoader,
    ".jpg": UnstructuredImageLoader,
    ".jpeg": UnstructuredImageLoader,
    ".png": UnstructuredImageLoader,
}


def _get_splitter():
    """Return the shared text splitter used across all ingestion paths."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _get_embeddings():
    """Return the shared embedding model."""
    return OllamaEmbeddings(model="nomic-embed-text")


def build_knowledge_base():
    """
    Scan ./data for PDFs and images (.pdf, .jpg, .jpeg, .png),
    split into chunks, embed, and store in ChromaDB at ./salaahkar_db.
    """
    all_documents = []

    # Discover and load files by extension
    for ext, loader_cls in FILE_LOADERS.items():
        pattern = os.path.join(DATA_DIR, "**", f"*{ext}")
        files = glob.glob(pattern, recursive=True)

        if files:
            file_type = "PDF" if ext == ".pdf" else "image"
            print(f"📂 Found {len(files)} {file_type} file(s) with extension {ext}")

            for filepath in files:
                print(f"   ⏳ Loading {filepath} ...")
                try:
                    loader = loader_cls(filepath)
                    docs = loader.load()
                    all_documents.extend(docs)
                    print(f"   ✅ Loaded {len(docs)} page(s)/section(s)")
                except Exception as e:
                    print(f"   ❌ Failed to load {filepath}: {e}")

    print(f"\n📊 Total loaded: {len(all_documents)} document section(s)")

    if not all_documents:
        print("⚠️  No supported documents found in ./data — please add some and re-run.")
        return

    # Split into chunks
    splitter = _get_splitter()
    print(f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = splitter.split_documents(all_documents)
    print(f"✅ Created {len(chunks)} chunk(s).")

    # Embed and store in ChromaDB
    print("🧠 Embedding chunks with nomic-embed-text and storing in ChromaDB ...")
    embeddings = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"✅ Vector store created at {DB_DIR} with {vectorstore._collection.count()} vectors.")
    print("🎉 Ingestion complete! You can now run the app.")


def ingest_youtube_video(url: str):
    """
    Download a YouTube video's transcript, split it into chunks,
    embed, and add to the existing ChromaDB at ./salaahkar_db.

    Args:
        url: A YouTube video URL (e.g. https://www.youtube.com/watch?v=...).

    Returns:
        int: The number of chunks added to the vector store.

    Raises:
        ValueError: If no transcript can be retrieved.
    """
    print(f"🎬 Fetching transcript for: {url}")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()

    if not documents:
        raise ValueError(f"Could not retrieve transcript for {url}")

    print(f"✅ Retrieved transcript ({len(documents)} section(s))")

    # Split
    splitter = _get_splitter()
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunk(s)")

    # Embed and add to existing store
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    print(f"✅ Added {len(chunks)} chunks to {DB_DIR}")
    print("🎉 YouTube video ingested! You can now ask questions about it.")

    return len(chunks)


if __name__ == "__main__":
    build_knowledge_base()
