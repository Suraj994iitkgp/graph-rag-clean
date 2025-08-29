# ingest.py
import os
from PyPDF2 import PdfReader
from vectorstore import upsert_documents
import graph_utils
import config

PDF_DIR = "docs"   # put all your PDFs inside a "docs" folder

def load_pdfs(pdf_dir):
    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append({
                "id": fname.replace(".pdf", ""),
                "text": text,
                "metadata": {"filename": fname}
            })
    return docs

if __name__ == "__main__":
    all_docs = load_pdfs(PDF_DIR)
    if not all_docs:
        print("No PDFs found in docs/.")
        exit(1)

    # 1) Upsert vectors (existing function)
    total = upsert_documents(all_docs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    print(f"✅ Uploaded {total} chunks from {len(all_docs)} PDFs to Pinecone")

    # 2) Build graph from docs and save
    print("🔗 Building knowledge graph from ingested docs (this may take a bit)...")
    graph_utils.build_graph_from_docs(all_docs)
    graph_utils.save_graph("graph/kg.graphml")
    print("✅ Knowledge graph saved to graph/kg.graphml (and .pkl).")
