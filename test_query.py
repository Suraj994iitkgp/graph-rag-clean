from vectorstore import query_pinecone

if __name__ == "__main__":
    # Try a query against your uploaded docs
    query = "How does ABS prevent wheel lock during braking?"
    results = query_pinecone(query, top_k=3)

    print("\n🔍 Query:", query)
    print("="*60)

    for i, match in enumerate(results, 1):
        score = match["score"]
        meta = match["metadata"]
        text = meta.get("text", "")[:300]  # show first 300 chars
        filename = meta.get("filename", "unknown")

        print(f"\nResult {i} (score={score:.4f}) from {filename}")
        print(f"Text: {text}...")
        print("-"*60)
