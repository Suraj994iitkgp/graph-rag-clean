# graph_utils.py
import re
import os
import json
import networkx as nx
import spacy
from typing import List, Dict, Tuple

# load spaCy model (make sure en_core_web_sm is installed)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    # helpful error if model is missing
    raise RuntimeError("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm") from e

KG = nx.DiGraph()

# -------------------------
# Entity & relation extraction
# -------------------------
def extract_entities_and_relations(text: str) -> List[Tuple[str, str, str, str]]:
    """
    Very conservative extractor:
    - Use spaCy NER & noun_chunks to find candidate entity spans
    - Create a simple relation for two entities appearing in the same sentence:
        (ent1, relation_text, ent2, sentence)
    - relation_text is the sentence (or short predicate) to preserve provenance
    """
    doc = nlp(text)
    relations = []
    for sent in doc.sents:
        sent_doc = nlp(sent.text.strip())
        ents = [e.text.strip() for e in sent_doc.ents if len(e.text.strip()) > 1]
        # fallback: use noun chunks if NER misses
        if len(ents) < 2:
            ents = [nc.text.strip() for nc in sent_doc.noun_chunks][:3]
        # form pairwise relations
        if len(ents) >= 2:
            for i in range(len(ents)):
                for j in range(i+1, len(ents)):
                    e1 = normalize_entity(ents[i])
                    e2 = normalize_entity(ents[j])
                    rel = sent.text.strip()
                    relations.append((e1, rel, e2, sent.text.strip()))
    return relations

def normalize_entity(text: str) -> str:
    """Clean entity string (lowercase, remove excessive whitespace/punctuation)."""
    t = text.strip()
    t = re.sub(r'[_\n]+', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t)
    return t

# -------------------------
# Add document to graph
# -------------------------
def add_document_to_graph(doc_id: str, text: str, metadata: Dict = None):
    """
    Extract relations from text and add to global KG.
    Each edge stores: relation_text, source_doc, sentence
    """
    rels = extract_entities_and_relations(text)
    for (a, rel_text, b, sentence) in rels:
        if not KG.has_node(a):
            KG.add_node(a, type="component")
        if not KG.has_node(b):
            KG.add_node(b, type="component")
        # if edge exists, append provenance; else create new
        if KG.has_edge(a, b):
            data = KG.get_edge_data(a, b)
            # store multiple provenance entries
            prov = data.get("provenance", [])
            prov.append({"source": doc_id, "sentence": sentence})
            data["provenance"] = prov
        else:
            KG.add_edge(a, b, relation=rel_text, provenance=[{"source": doc_id, "sentence": sentence}])

# -------------------------
# Build graph from docs (list of dicts {'id','text'})
# -------------------------
def build_graph_from_docs(docs: List[Dict]):
    for d in docs:
        add_document_to_graph(d.get("id"), d.get("text", ""), d.get("metadata", {}))

# -------------------------
# Graph expansion utilities
# -------------------------
def expand_nodes(seed_nodes: List[str], hops: int = 1) -> nx.DiGraph:
    """
    BFS expansion starting from seed_nodes for `hops` steps (both directions).
    Returns a subgraph (NetworkX view).
    """
    nodes = set()
    frontier = set(seed_nodes)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            if n not in KG:
                continue
            neigh = set(KG.successors(n)) | set(KG.predecessors(n))
            next_frontier |= neigh
        next_frontier -= nodes
        nodes |= frontier
        frontier = next_frontier
        if not frontier:
            break
    nodes |= frontier
    return KG.subgraph(nodes).copy()

def graph_to_facts(subgraph: nx.DiGraph) -> List[str]:
    """
    Convert subgraph edges into human-friendly facts (strings).
    """
    facts = []
    for u, v, data in subgraph.edges(data=True):
        rel = data.get("relation", "")
        provs = data.get("provenance", [])
        # include a short provenance snippet if available
        prov_sn = provs[0].get("source", "") if provs else ""
        facts.append(f"{u} -> {v} | relation: {shorten(rel,200)} | source: {prov_sn}")
    return facts

def shorten(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ",1)[0] + "..."

# -------------------------
# Save / load graph
# -------------------------
import networkx as nx

def save_graph(path="graph/kg.graphml"):
    global KG

    # Convert list attributes into strings (comma-separated)
    for n, data in KG.nodes(data=True):
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = ", ".join(map(str, v))

    for u, v, data in KG.edges(data=True):
        for k, val in data.items():
            if isinstance(val, list):
                data[k] = ", ".join(map(str, val))

    nx.write_graphml(KG, path)
    print(f"✅ Knowledge graph saved to {path}")


# -------------------------
# Utilities
# -------------------------
def get_top_entities(n=50):
    """
    Return top n nodes by degree (helpful for quick inspection)
    """
    deg = sorted(KG.degree(), key=lambda x: x[1], reverse=True)
    return [d for d,_ in deg[:n]]

def graph_retrieve(query: str, top_k: int = 5):
    """Retrieve related nodes from KG given a query"""
    matches = []
    for node, data in KG.nodes(data=True):
        if query.lower() in node.lower():
            matches.append((node, data))
    return matches[:top_k]

