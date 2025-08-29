# main.py
import os
import re
from typing import List, Tuple
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vectorstore import query_pinecone       # <-- your existing function
from graph_utils import graph_retrieve      # <-- your existing function
from transformers import pipeline
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Configurable model choice
# -------------------------
# On CPU: "google/flan-t5-small" (fast, lower quality)
# Better (still CPU friendly): "google/flan-t5-base"
GEN_MODEL = os.environ.get("GEN_MODEL", "google/flan-t5-base")

print(f"Loading generation model ({GEN_MODEL})... (this may take a while on CPU)")
gen_pipeline = pipeline("text2text-generation", model=GEN_MODEL, device=-1)
print("Generation model loaded.")

# Optional: use same model for summarization/refinement (cheap)
refinement_pipeline = gen_pipeline

# -------------------------
# Helpers: cleaning + chunking
# -------------------------
def clean_snippet(text: str) -> str:
    """Remove figure/table labels, page numbers, and noisy all-caps headings."""
    if not text:
        return ""
    text = re.sub(r'\b(Figure|Fig|Table)\s*\d+(\.\d+)*\b', '', text, flags=re.I)
    text = re.sub(r'^\s*\d+(\.\d+){0,}\s*[-.:]?\s*', '', text, flags=re.M)
    text = re.sub(r'\b(Page|pp)\.?\s*\d+(-\d+)?\b', '', text, flags=re.I)
    text = re.sub(r'\n{2,}', '\n', text)
    text = '\n'.join([ln for ln in text.splitlines() if not re.match(r'^\s*[A-Z\W]{2,}$', ln)])
    return text.strip()

def split_into_chunks(text: str, max_words: int = 400, overlap: int = 50) -> List[str]:
    """
    Split a long text into chunks of about `max_words` words with `overlap` words overlap,
    attempting to split on sentence boundaries for readability.
    """
    if not text:
        return []
    # naive sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_words = 0

    for s in sentences:
        words = s.split()
        if current_words + len(words) <= max_words or not current:
            current.append(s)
            current_words += len(words)
        else:
            chunks.append(" ".join(current).strip())
            # start overlap
            if overlap > 0:
                # take last `overlap` words from current to seed next
                tail_words = " ".join(current).split()[-overlap:]
                current = [" ".join(tail_words)]
                current_words = len(tail_words)
            else:
                current = []
                current_words = 0
            current.append(s)
            current_words += len(words)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def score_chunk_by_query(query: str, chunk: str) -> int:
    """
    Simple lexical relevance scoring: count occurrences of query words in chunk.
    This is light-weight and avoids extra cross-encoder heavy models.
    """
    q_tokens = re.findall(r'\w+', query.lower())
    chunk_tokens = re.findall(r'\w+', chunk.lower())
    if not q_tokens or not chunk_tokens:
        return 0
    score = 0
    from collections import Counter
    c = Counter(chunk_tokens)
    for t in set(q_tokens):
        score += c.get(t, 0)
    return score

def build_context_from_retrievals(pinecone_results: List[dict],
                                  graph_results: List[Tuple[str, str]],
                                  query: str,
                                  max_chunks: int = 6,
                                  chunk_words: int = 350,
                                  chunk_overlap: int = 50) -> Tuple[str, List[str]]:
    """
    - Split retrieved documents into smaller chunks
    - Score chunks by query term overlap
    - Select top `max_chunks` chunks to build context
    - Also include some compact graph node lines (short)
    Returns (context_text, sources_list)
    """
    all_chunks = []
    sources = []

    # process pinecone results
    for m in pinecone_results:
        raw = m.get("metadata", {}).get("text", "") or m.get("text", "")
        filename = m.get("metadata", {}).get("filename", "unknown")
        cleaned = clean_snippet(raw)
        if not cleaned:
            continue
        chunks = split_into_chunks(cleaned, max_words=chunk_words, overlap=chunk_overlap)
        for c in chunks:
            score = score_chunk_by_query(query, c)
            all_chunks.append({"chunk": c, "score": score, "source": filename})

    # process graph results (short, structured)
    for node, data in graph_results:
        node_text = f"Graph node: {node}. Info: {data}"
        # keep graph nodes as single chunks with higher priority (score boost)
        all_chunks.append({"chunk": node_text, "score": score_chunk_by_query(query, node_text) + 2, "source": f"graph:{node}"})

    # sort by score desc, then by chunk length (prefer concise)
    all_chunks.sort(key=lambda x: (x["score"], -len(x["chunk"])), reverse=True)

    # select top unique sources/chunks up to max_chunks
    selected = []
    seen_texts = set()
    seen_sources = []
    for item in all_chunks:
        txt = item["chunk"]
        if txt in seen_texts:
            continue
        selected.append(item)
        seen_texts.add(txt)
        if item["source"] not in seen_sources:
            seen_sources.append(item["source"])
        if len(selected) >= max_chunks:
            break

    # if not enough selected chunks (low overlap), fallback to taking first N raw results (shortened)
    if not selected and pinecone_results:
        fallback = clean_snippet(pinecone_results[0].get("metadata", {}).get("text", ""))[:2000]
        selected = [{"chunk": fallback, "score": 0, "source": pinecone_results[0].get("metadata", {}).get("filename", "unknown")}]

    # join chunks with separator
    context = "\n\n---\n\n".join([s["chunk"] for s in selected])
    return context, seen_sources

# -------------------------
# Improved prompt templates
# -------------------------
PROMPT_TEMPLATE = (
    "You are an expert automotive braking systems engineer. "
    "Using ONLY the provided context, write a clear, complete, and concise answer to the question. "
    "Do NOT copy long verbatim passages. Instead, synthesize, summarize, and rephrase the information. "
    "When relevant, provide short bullet points for steps or causes. If you cannot answer from the context, say 'Insufficient information in context.' "
    "\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
)

# optional refinement prompt: rephrase the model's draft to reduce verbatim copying and improve clarity
REFINEMENT_PROMPT = (
    "You are a language quality assistant. Rephrase the following answer to be clearer and more concise, "
    "without inventing new facts and without copying long verbatim phrases from the original context. "
    "Keep technical meaning intact.\n\nOriginal Answer:\n{answer}\n\nRefined Answer:"
)

# -------------------------
# FastAPI endpoints
# -------------------------
@app.get("/")
def root():
    return {"message": "Chatbot API running. Visit /chat for UI."}

@app.get("/chat")
def chat_ui():
    # ensure chat.html exists in working dir
    path = os.path.join(os.getcwd(), "chat.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "chat.html not found"}, status_code=404)

@app.get("/ask")
def ask(query: str = Query(..., description="User question"),
        top_k: int = Query(3, description="How many vector results to retrieve (default 3)"),
        max_chunks: int = Query(6, description="Max number of chunks to include in context")):
    """
    Main ask endpoint:
      - retrieves from Pinecone (or your vectorstore)
      - retrieves from Graph
      - builds a small, relevant context by chunking + reranking
      - calls the generator with settings to avoid truncation/copying
      - optionally runs a lightweight refinement pass
    """
    if not query or not query.strip():
        return JSONResponse({"error": "Empty query provided"}, status_code=400)

    # Step 1: vector retrieval (your function)
    try:
        pinecone_results = query_pinecone(query, top_k=top_k) or []
    except Exception as e:
        return JSONResponse({"error": f"Vector retrieval failed: {str(e)}"}, status_code=500)

    # Step 2: graph retrieval (your function)
    try:
        graph_results = graph_retrieve(query, top_k=top_k) or []
    except Exception as e:
        graph_results = []
        # not fatal, continue with vector results

    # Step 3: build compact context from retrievals
    context, sources = build_context_from_retrievals(
        pinecone_results, graph_results, query, max_chunks=max_chunks
    )

    # guard: if context is empty, return a helpful message
    if not context:
        return {"query": query, "answer": "No relevant context found for this query.", "sources": []}

    # Step 4: build prompt & generate (ensure model has fresh tokens to respond)
    prompt = PROMPT_TEMPLATE.format(question=query.strip(), context=context)

    try:
        gen_out = gen_pipeline(
            prompt,
            max_new_tokens=256,   # generate up to 256 new tokens (adjust as required)
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            truncation=True
        )
        raw_answer = gen_out[0].get("generated_text", "").strip()
    except Exception as e:
        return JSONResponse({"error": f"Generation failed: {str(e)}"}, status_code=500)

    # Step 5: optional refinement to reduce copying and improve clarity
    try:
        refine_prompt = REFINEMENT_PROMPT.format(answer=raw_answer)
        ref_out = refinement_pipeline(
            refine_prompt,
            max_new_tokens=160,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            truncation=True
        )
        refined_answer = ref_out[0].get("generated_text", "").strip()
    except Exception:
        # if refinement fails, fall back to raw_answer
        refined_answer = raw_answer

    # Step 6: sanitize answer (strip excessive whitespace)
    answer = re.sub(r'\n{3,}', '\n\n', refined_answer).strip()

    # return unique ordered sources
    unique_sources = []
    for s in sources:
        if s not in unique_sources:
            unique_sources.append(s)

    return {"query": query, "answer": answer, "sources": unique_sources, "context_used": context[:2000]}

# -------------------------
# Run server (development)
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
