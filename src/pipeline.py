import os
import json
import time
import numpy as np
import requests
import spacy
import networkx as nx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GROQ_API_KEY             = os.getenv("GROQ_API_KEY")

MODEL_NAME         = "all-MiniLM-L6-v2"
GROQ_MODEL         = "llama-3.3-70b-versatile"
SIMILARITY_THRESHOLD = 0.65

BCI_KEYWORDS = [
    "neural", "electrode", "eeg", "ecog", "meg", "fmri", "bci",
    "brain", "cortex", "neuron", "signal", "implant", "prosthetic",
    "motor", "cognitive", "attention", "seizure", "stroke", "paralysis",
    "deep learning", "machine learning", "classification", "decoding",
    "wireless", "invasive", "non-invasive", "feedback", "rehabilitation",
    "algorithm", "model", "detection", "prediction", "analysis",
    "network", "system", "interface", "sensor", "data", "learning"
]

SATURATED_KEYWORDS = [
    "brain", "signal", "classification", "data", "system"
]
EMERGING_KEYWORDS = [
    "wireless", "implant", "invasive", "rehabilitation",
    "feedback", "prosthetic", "cognitive", "attention",
    "prediction", "detection", "network"
]

WEIGHTS = {
    "research_momentum": 0.30,
    "citation_density" : 0.25,
    "keyword_novelty"  : 0.25,
    "isolation_score"  : 0.20
}

def fetch_papers(query, api_key, limit=50):
    url     = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key}
    params  = {
        "query" : query,
        "limit" : limit,
        "fields": "title,abstract,year,authors,citationCount,externalIds"
    }
    for attempt in range(1, 4):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", [])
        elif response.status_code == 429:
            time.sleep(15 * attempt)
        else:
            break
    return []

def extract_keywords(text):
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in BCI_KEYWORDS if kw in text_lower]

def build_metadata(papers):
    metadata = []
    for i, paper in enumerate(papers):
        title    = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        keywords = extract_keywords(f"{title}. {abstract}")
        metadata.append({
            "id"           : i,
            "title"        : title,
            "year"         : paper.get("year"),
            "citations"    : paper.get("citationCount", 0) or 0,
            "abstract"     : abstract,
            "bci_keywords" : keywords,
            "keyword_count": len(keywords)
        })
    return metadata

def generate_embeddings(metadata, model):
    texts = [f"{p['title']}. {p['abstract']}" for p in metadata]
    return model.encode(texts, show_progress_bar=False, batch_size=32)

def build_graph(embeddings, metadata):
    G          = nx.Graph()
    sim_matrix = cosine_similarity(embeddings)

    for p in metadata:
        G.add_node(p["id"], **p)

    for i in range(len(metadata)):
        for j in range(i + 1, len(metadata)):
            if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=float(sim_matrix[i][j]))

    components = sorted(
        list(nx.connected_components(G)), key=len, reverse=True
    )

    degree     = nx.degree_centrality(G)
    between    = nx.betweenness_centrality(G, normalized=True) if G.number_of_edges() > 0 else {}

    nodes_out = []
    for node_id, data in G.nodes(data=True):
        nodes_out.append({
            "id"                    : node_id,
            "title"                 : data.get("title", ""),
            "year"                  : data.get("year"),
            "citations"             : data.get("citations", 0),
            "keywords"              : data.get("bci_keywords", []),
            "keyword_count"         : data.get("keyword_count", 0),
            "abstract"              : data.get("abstract", "")[:300],
            "degree_centrality"     : round(degree.get(node_id, 0), 4),
            "betweenness_centrality": round(between.get(node_id, 0), 4),
            "combined_score"        : round(
                0.5 * degree.get(node_id, 0) +
                0.5 * between.get(node_id, 0), 4
            ),
            "cluster": next(
                (i for i, comp in enumerate(components) if node_id in comp), 0
            )
        })

    edges_out = [
        {"source": u, "target": v, "weight": round(d.get("weight", 0), 4)}
        for u, v, d in G.edges(data=True)
    ]

    return {
        "nodes"        : nodes_out,
        "edges"        : edges_out,
        "num_nodes"    : G.number_of_nodes(),
        "num_edges"    : G.number_of_edges(),
        "num_clusters" : len(components),
        "cluster_sizes": [len(c) for c in components]
    }, nodes_out

def compute_scores(metadata, nodes_out):
    years     = [p["year"] for p in metadata if p["year"]]
    min_year  = min(years) if years else 2000
    max_year  = max(years) if years else 2025
    year_range = max_year - min_year if max_year != min_year else 1

    citations  = [p["citations"] for p in metadata]
    max_cit    = max(citations) if max(citations) > 0 else 1

    max_degree  = max(n["degree_centrality"] for n in nodes_out) or 1
    max_between = max(n["betweenness_centrality"] for n in nodes_out) or 1

    node_map = {n["id"]: n for n in nodes_out}

    results = []
    for p in metadata:
        pid  = p["id"]
        node = node_map.get(pid, {})

        # 4 signals
        momentum  = round((p["year"] - min_year) / year_range, 4) if p["year"] else 0
        cit_score = round(p["citations"] / max_cit, 4)

        kws        = p["bci_keywords"]
        total      = len(kws) if kws else 1
        eme        = sum(1 for k in kws if k in EMERGING_KEYWORDS)
        sat        = sum(1 for k in kws if k in SATURATED_KEYWORDS)
        novelty    = round(max(0.0, min(1.0, (eme - 0.5 * sat) / total + 0.5)), 4)

        degree  = node.get("degree_centrality", 0) / max_degree
        between = node.get("betweenness_centrality", 0) / max_between
        isolation = round((1 - degree) * 0.6 + between * 0.4, 4)

        final = round(
            WEIGHTS["research_momentum"] * momentum +
            WEIGHTS["citation_density"]  * cit_score +
            WEIGHTS["keyword_novelty"]   * novelty +
            WEIGHTS["isolation_score"]   * isolation, 4
        )

        results.append({
            "id"          : pid,
            "title"       : p["title"],
            "year"        : p["year"],
            "citations"   : p["citations"],
            "bci_keywords": p["bci_keywords"],
            "abstract"    : p["abstract"],
            "scores": {
                "research_momentum"           : momentum,
                "citation_density"            : cit_score,
                "keyword_novelty"             : novelty,
                "isolation_score"             : isolation,
                "innovation_opportunity_score": final
            }
        })

    results = sorted(
        results,
        key=lambda x: x["scores"]["innovation_opportunity_score"],
        reverse=True
    )

    gaps = [
        r for r in results
        if r["scores"]["isolation_score"]  > 0.5
        and r["scores"]["keyword_novelty"] > 0.4
        and r["scores"]["citation_density"] < 0.3
    ]

    return {
        "total_papers"     : len(results),
        "total_gaps"       : len(gaps),
        "weights_used"     : WEIGHTS,
        "top_opportunities": results[:10],
        "research_gaps"    : gaps[:10],
        "all_scores"       : results
    }

def generate_proposal(gap, groq_client, query):
    prompt = f"""You are an expert AI innovation analyst.

A research paper has been identified as UNDEREXPLORED in the domain of "{query}".

Paper:
- Title: {gap['title']}
- Year: {gap['year']}
- Citations: {gap['citations']}
- Keywords: {', '.join(gap['bci_keywords'])}
- Abstract: {gap['abstract'][:400]}

Generate a structured innovation proposal with these exact sections:

1. INNOVATION TITLE
2. PROBLEM IT SOLVES
3. PROPOSED SOLUTION
4. WHY IT IS NOVEL
5. TECHNOLOGY STACK
6. TARGET USERS
7. PATENT OPPORTUNITY
8. RESEARCH CONTRIBUTION
9. IMPACT SCORE

Be specific, technical, and forward-thinking."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert AI innovation analyst. Be specific and technical."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content

def run_pipeline(query, status_callback=None):
    def update(msg):
        if status_callback:
            status_callback(msg)

    results = {}

    # Step 1
    update("📡 Fetching research papers...")
    papers = fetch_papers(query, SEMANTIC_SCHOLAR_API_KEY, limit=50)
    if not papers:
        return None, "Failed to fetch papers. Check API key or try again."
    update(f"✅ Fetched {len(papers)} papers")

    # Step 2
    update("🔍 Extracting entities and keywords...")
    metadata = build_metadata(papers)
    update(f"✅ Extracted keywords from {len(metadata)} papers")

    # Step 3
    update("🧠 Generating embeddings...")
    model      = SentenceTransformer(MODEL_NAME)
    embeddings = generate_embeddings(metadata, model)
    update(f"✅ Generated embeddings: {embeddings.shape}")

    # Step 4
    update("🕸️ Building knowledge graph...")
    graph_data, nodes_out = build_graph(embeddings, metadata)
    update(f"✅ Graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges, {graph_data['num_clusters']} clusters")

    # Step 5
    update("📊 Computing innovation opportunity scores...")
    scores_data = compute_scores(metadata, nodes_out)
    update(f"✅ Scored {scores_data['total_papers']} papers, {scores_data['total_gaps']} gaps detected")

    # Step 6
    update("🤖 Generating AI innovation proposals...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    innovations = []
    gaps        = scores_data["research_gaps"][:3]

    for i, gap in enumerate(gaps):
        update(f"  Generating proposal {i+1}/{len(gaps)}...")
        try:
            raw = generate_proposal(gap, groq_client, query)
            innovations.append({
                "gap_title"    : gap["title"],
                "gap_score"    : gap["scores"]["innovation_opportunity_score"],
                "gap_year"     : gap["year"],
                "gap_citations": gap["citations"],
                "gap_keywords" : gap["bci_keywords"],
                "proposal"     : {"raw": raw}
            })
        except Exception as e:
            update(f"  Warning: proposal {i+1} failed — {e}")

    update(f"✅ Generated {len(innovations)} innovation proposals")
    update("🎉 Pipeline complete!")

    return {
        "query"      : query,
        "papers"     : len(papers),
        "metadata"   : metadata,
        "embeddings" : embeddings.tolist(),
        "graph"      : graph_data,
        "scores"     : scores_data,
        "innovations": innovations
    }, None