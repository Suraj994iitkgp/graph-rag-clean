# build_evalset.py
import os
import json
import random
from typing import List
from PyPDF2 import PdfReader
from transformers import pipeline

DOCS_DIR = "docs"        # folder with your 6-7 PDFs
OUT_PATH = "eval/evalset.jsonl"
MAX_QAS = 30             # how many total QA pairs to create
CHUNK_SIZE = 1000        # chars per chunk used for prompting
SAMPLE_PER_DOC = 5       # sample chunks per document

def load_pdfs(pdf_dir: str) -> List[dict]:
    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append({"filename": fname, "text": text})
    return docs

def chunk_by_chars(text: str, size: int = CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size
    return chunks

def main():
    # Load generator model (FLAN-T5 small) - device=-1 uses CPU
    print("Loading FLAN-T5 model for Q/A generation (this may take a while)...")
    qgen = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    print("Model loaded.")

    docs = load_pdfs(DOCS_DIR)
    if not docs:
        print("No PDFs found in docs/. Put your PDFs in the docs folder and retry.")
        return

    qa_items = []
    for d in docs:
        chunks = chunk_by_chars(d["text"], CHUNK_SIZE)
        if not chunks:
            continue
        # sample chunks to avoid huge number
        sample_cnt = min(SAMPLE_PER_DOC, len(chunks))
        sampled = random.sample(chunks, k=sample_cnt)
        for chunk in sampled:
            # 1) Generate a clear technical question from the chunk
            q_prompt = ("Generate one clear technical question that can be answered using the following context. "
                        "Return only the question.\n\nContext:\n" + chunk)
            q_out = qgen(q_prompt, max_length=64, do_sample=False)[0]["generated_text"].strip()
            # 2) Generate a concise reference answer from same chunk (ground truth)
            a_prompt = ("Answer this question concisely and precisely using ONLY the provided context. "
                        "Do not add anything else.\n\nQuestion: " + q_out + "\n\nContext:\n" + chunk)
            a_out = qgen(a_prompt, max_length=160, do_sample=False)[0]["generated_text"].strip()

            if q_out and a_out:
                qa_items.append({
                    "question": q_out,
                    "reference_answer": a_out,
                    "source_filename": d["filename"]
                })
            if len(qa_items) >= MAX_QAS:
                break
        if len(qa_items) >= MAX_QAS:
            break

    # save to jsonl
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in qa_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(qa_items)} QA pairs to {OUT_PATH}")

if __name__ == "__main__":
    main()
