import os
import json
import numpy as np
from collections import Counter

# File paths
GRAPH_FILE    = "data/processed/bci_graph.json"
METADATA_FILE = "data/processed/bci_metadata.json"
SCORES_FILE   = "data/processed/bci_scores.json"

# BCI domain keywords
BCI_KEYWORDS = [
    "neural", "electrode", "eeg", "ecog", "meg", "fmri", "bci",
    "brain", "cortex", "neuron", "signal", "implant", "prosthetic",
    "motor", "cognitive", "attention", "seizure", "stroke", "paralysis",
    "deep learning", "machine learning", "classification", "decoding",
    "wireless", "invasive", "non-invasive", "feedback", "rehabilitation"
]

# Weights for each signal (must add to 1.0)
WEIGHTS = {
    "research_momentum" : 0.25,
    "citation_density"  : 0.20,
    "keyword_novelty"   : 0.20,
    "isolation_score"   : 0.20,
    "patent_gap_score"  : 0.15
}

# Keywords that are saturated (too many papers = low novelty)
SATURATED_KEYWORDS = [
    "brain", "bci", "eeg", "signal", "classification"
]

# Keywords that are emerging (fewer papers = high novelty)
EMERGING_KEYWORDS = [
    "wireless", "implant", "invasive", "rehabilitation",
    "feedback", "prosthetic", "cognitive", "attention"
]

def load_data():
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded graph: {graph['num_nodes']} nodes, {graph['num_edges']} edges")
    return graph, metadata

def compute_research_momentum(metadata):
    """Score based on how recently papers are being published."""
    years = [p["year"] for p in metadata if p["year"]]
    if not years:
        return {p["id"]: 0.5 for p in metadata}

    min_year = min(years)
    max_year = max(years)
    year_range = max_year - min_year if max_year != min_year else 1

    scores = {}
    for p in metadata:
        year = p["year"] or min_year
        # Normalize: newer = higher score
        scores[p["id"]] = round((year - min_year) / year_range, 4)

    return scores

def compute_citation_density(metadata):
    """Score based on citation count — more cited = more important field."""
    citations = [p["citations"] or 0 for p in metadata]
    max_cit = max(citations) if max(citations) > 0 else 1

    scores = {}
    for p in metadata:
        cit = p["citations"] or 0
        # Normalize to 0-1
        scores[p["id"]] = round(cit / max_cit, 4)

    return scores

def compute_keyword_novelty(metadata):
    """Score based on presence of emerging vs saturated keywords."""
    scores = {}
    for p in metadata:
        keywords = p["bci_keywords"]

        emerging_count  = sum(1 for kw in keywords if kw in EMERGING_KEYWORDS)
        saturated_count = sum(1 for kw in keywords if kw in SATURATED_KEYWORDS)
        total           = len(keywords) if keywords else 1

        # High emerging ratio = high novelty
        novelty = (emerging_count - 0.5 * saturated_count) / total
        # Clip to 0-1
        novelty = max(0.0, min(1.0, novelty + 0.5))
        scores[p["id"]] = round(novelty, 4)

    return scores

def compute_isolation_score(graph):
    """
    Papers with few connections are underexplored = high opportunity.
    Papers too central are already well-studied = lower opportunity.
    Sweet spot: moderate connections with high betweenness.
    """
    scores = {}
    nodes = graph["nodes"]

    max_degree = max(n["degree_centrality"] for n in nodes) or 1
    max_between = max(n["betweenness_centrality"] for n in nodes) or 1

    for node in nodes:
        degree   = node["degree_centrality"] / max_degree
        between  = node["betweenness_centrality"] / max_between

        # Low degree but some betweenness = bridge paper = gap opportunity
        isolation = (1 - degree) * 0.6 + between * 0.4
        scores[node["id"]] = round(isolation, 4)

    return scores

def compute_patent_gap_score(metadata, patent_file="data/raw/patents/bci_patents.json"):
    """
    Score each paper based on how much white space exists in the patent
    landscape around its keywords.
    Low patent coverage around a topic = high opportunity.
    """
    import os

    # Load patents if available
    patent_keywords = []
    if os.path.exists(patent_file):
        with open(patent_file, "r", encoding="utf-8") as f:
            patents = json.load(f)
        for pat in patents:
            title    = (pat.get("title", "") or "").lower()
            abstract = (pat.get("abstract", "") or "").lower()
            text     = f"{title} {abstract}"
            for kw in BCI_KEYWORDS:
                if kw in text:
                    patent_keywords.append(kw)

    from collections import Counter
    patent_kw_counts = Counter(patent_keywords)
    max_count        = max(patent_kw_counts.values()) if patent_kw_counts else 1

    scores = {}
    for p in metadata:
        paper_kws = p.get("bci_keywords", [])
        if not paper_kws:
            scores[p["id"]] = 0.5
            continue

        # How saturated is the patent space for this paper's keywords?
        patent_coverage = sum(
            patent_kw_counts.get(kw, 0) for kw in paper_kws
        ) / (len(paper_kws) * max_count)

        # Low coverage = high gap score (opportunity)
        scores[p["id"]] = round(1.0 - patent_coverage, 4)

    return scores


def compute_final_scores(metadata, graph, momentum, citations,
                          novelty, isolation, patent_gap):
    """Combine all signals into one Innovation Opportunity Score."""
    results = []

    for p in metadata:
        pid = p["id"]

        m  = momentum.get(pid, 0)
        c  = citations.get(pid, 0)
        n  = novelty.get(pid, 0)
        i  = isolation.get(pid, 0)
        pg = patent_gap.get(pid, 0)

        final_score = (
            WEIGHTS["research_momentum"] * m  +
            WEIGHTS["citation_density"]  * c  +
            WEIGHTS["keyword_novelty"]   * n  +
            WEIGHTS["isolation_score"]   * i  +
            WEIGHTS["patent_gap_score"]  * pg
        )

        results.append({
            "id"           : pid,
            "title"        : p["title"],
            "year"         : p["year"],
            "citations"    : p["citations"],
            "bci_keywords" : p["bci_keywords"],
            "abstract"     : p["abstract"],
            "scores": {
                "research_momentum"           : m,
                "citation_density"            : c,
                "keyword_novelty"             : n,
                "isolation_score"             : i,
                "patent_gap_score"            : pg,
                "innovation_opportunity_score": round(final_score, 4)
            }
        })

    return sorted(
        results,
        key=lambda x: x["scores"]["innovation_opportunity_score"],
        reverse=True
    )

def identify_gaps(results):
    """
    Find research gaps: papers scoring high on isolation + novelty
    but lower on citations = underexplored areas.
    """
    gaps = []
    for r in results:
        s = r["scores"]
        if (
            s["isolation_score"]   > 0.5 and
            s["keyword_novelty"]   > 0.4 and
            s["citation_density"]  < 0.3
        ):
            gaps.append(r)

    return gaps

def save_scores(results, gaps, output_file):
    output = {
        "total_papers"  : len(results),
        "total_gaps"    : len(gaps),
        "weights_used"  : WEIGHTS,
        "top_opportunities": results[:10],
        "research_gaps" : gaps[:10],
        "all_scores"    : results
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved scores to {output_file}")

def print_summary(results, gaps):
    print("\n--- Innovation Opportunity Scores ---")
    print(f"{'Rank':<5} {'Score':<8} {'Year':<6} {'Title'[:55]}")
    print("-" * 75)

    for rank, r in enumerate(results[:10], 1):
        score = r["scores"]["innovation_opportunity_score"]
        year  = r["year"] or "N/A"
        title = r["title"][:55]
        print(f"  {rank:<4} {score:<8} {year:<6} {title}...")

    print(f"\n--- Research Gaps Detected: {len(gaps)} ---")
    if gaps:
        for g in gaps[:5]:
            score = g["scores"]["innovation_opportunity_score"]
            print(f"  [{score}] {g['title'][:65]}...")
            print(f"          Keywords: {', '.join(g['bci_keywords'][:4])}")
    else:
        print("  No gaps detected with current thresholds.")

def main():
    print("Running Innovation Opportunity Scoring Engine...\n")

    graph, metadata = load_data()

    print("\nComputing signal scores...")
    momentum  = compute_research_momentum(metadata)
    citations = compute_citation_density(metadata)
    novelty   = compute_keyword_novelty(metadata)
    isolation = compute_isolation_score(graph)
    print("  research_momentum : done")
    print("  citation_density  : done")
    print("  keyword_novelty   : done")
    print("  isolation_score   : done")

    print("\nComputing patent gap scores...")
    patent_gap = compute_patent_gap_score(metadata)
    print("  patent_gap_score  : done")

    print("\nComputing final Innovation Opportunity Scores...")
    results = compute_final_scores(
        metadata, graph, momentum, citations, novelty, isolation, patent_gap
    )

    print("\nIdentifying research gaps...")
    gaps = identify_gaps(results)

    save_scores(results, gaps, SCORES_FILE)
    print_summary(results, gaps)

    print("\nScoring complete.")

if __name__ == "__main__":
    main()