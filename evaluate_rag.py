# evaluate_rag.py
import os, csv, json
from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from vectorstore import query_pinecone
import config

EVAL_PATH = "eval/evalset.jsonl"
OUT_CSV = "eval/results.csv"

# Generator model used in your system (should match your chatbot generator)
GEN_MODEL = "google/flan-t5-small"
gen = pipeline("text2text-generation", model=GEN_MODEL, device=-1)

# Embedding model (your retrieval embedder: MiniLM)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

PROMPT_TEMPLATE = (
    "You are an expert automotive braking systems engineer. Using ONLY the provided context, "
    "give a detailed, clear answer to the question. Do NOT invent facts.\n\nQuestion: {q}\n\nContext:\n{ctx}\n\nAnswer:"
)

def load_evalset(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def evaluate():
    data = load_evalset(EVAL_PATH)
    if not data:
        print("No eval items found. Run build_evalset.py first.")
        return

    k = 5
    hits = 0
    mrr_total = 0.0
    sim_scores = []

    rows = []
    for i, ex in enumerate(data, 1):
        q = ex["question"]
        ref = ex["reference_answer"]
        gold_file = ex["source_filename"]

        # Retrieval
        matches = query_pinecone(q, top_k=k)
        retrieved_files = [m["metadata"].get("filename", m["metadata"].get("source_id", "unknown")) for m in matches]

        hit = 1 if gold_file in retrieved_files else 0
        hits += hit

        # MRR
        rank = 0
        for idx, fn in enumerate(retrieved_files, start=1):
            if fn == gold_file:
                rank = idx
                break
        rr = (1.0 / rank) if rank else 0.0
        mrr_total += rr

        # Build context (top-k texts)
        context = " ".join([m["metadata"].get("text", "")[:1500] for m in matches])[:8000]

        # Generation (simulate your chatbot)
        prompt = PROMPT_TEMPLATE.format(q=q, ctx=context)
        gen_out = gen(prompt, max_length=512, do_sample=False, truncation=True)[0]["generated_text"].strip()

        # Semantic similarity between generated and reference using MiniLM
        emb_ref = embedder.encode(ref, convert_to_tensor=True)
        emb_gen = embedder.encode(gen_out, convert_to_tensor=True)
        cos = util.cos_sim(emb_ref, emb_gen).item()

        sim_scores.append(cos)

        rows.append({
            "question": q,
            "gold_source": gold_file,
            "retrieved": "|".join(retrieved_files),
            "rank_of_gold": rank,
            "hit@5": hit,
            "gen_answer": gen_out,
            "ref_answer": ref,
            "semantic_sim": round(cos, 4)
        })

        print(f"[{i}/{len(data)}] hit@5={hit} rr={round(rr,3)} sim={round(cos,3)}")

    # Aggregate metrics
    hit_at_k = hits / len(data)
    mrr = mrr_total / len(data)
    avg_sim = sum(sim_scores) / len(sim_scores)

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"Items evaluated: {len(data)}")
    print(f"Hit@{k}: {hit_at_k:.3f}")
    print(f"MRR:    {mrr:.3f}")
    print(f"Avg semantic similarity (MiniLM cosine): {avg_sim:.3f}")
    print(f"Details saved: {OUT_CSV}")
        # Save summary to JSON and TXT
    summary = {
        "items_evaluated": len(data),
        "hit_at_5": round(hit_at_k, 3),
        "mrr": round(mrr, 3),
        "avg_semantic_similarity": round(avg_sim, 3)
    }

    with open("eval/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open("eval/summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("✅ Summary saved to eval/summary.json and eval/summary.txt")


if __name__ == "__main__":
    evaluate()
