"""
Salaahkar - PDF Ingestion Script
Loads PDFs from ./data, splits into chunks, embeds with nomic-embed-text,
and stores in a local ChromaDB at ./salaahkar_db.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "./data"
DB_DIR = "./salaahkar_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    # Step 1: Load PDFs
    print("📂 Loading PDFs from ./data ...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from PDF files.")

    if not documents:
        print("⚠️  No PDF documents found in ./data — please add some and re-run.")
        return

    # Step 2: Split into chunks
    print(f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunk(s).")

    # Step 3: Embed and store in ChromaDB
    print("🧠 Embedding chunks with nomic-embed-text and storing in ChromaDB ...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"✅ Vector store created at {DB_DIR} with {vectorstore._collection.count()} vectors.")
    print("🎉 Ingestion complete! You can now run the app.")


if __name__ == "__main__":
    main()
