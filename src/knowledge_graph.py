import os
import json
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# File paths
EMBEDDINGS_FILE = "data/processed/bci_embeddings.npy"
METADATA_FILE   = "data/processed/bci_metadata.json"
GRAPH_FILE      = "data/processed/bci_graph.json"

# How similar two papers must be to draw an edge between them
# 0.7 = only very similar papers connect
# 0.5 = more connections, denser graph
SIMILARITY_THRESHOLD = 0.65

def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} papers and embeddings of shape {embeddings.shape}")
    return embeddings, metadata

def build_graph(embeddings, metadata):
    print("\nBuilding knowledge graph...")

    G = nx.Graph()

    # Add every paper as a node
    for paper in metadata:
        G.add_node(
            paper["id"],
            title         = paper["title"],
            year          = paper["year"],
            citations     = paper["citations"],
            keywords      = paper["bci_keywords"],
            keyword_count = paper["keyword_count"],
            abstract      = paper["abstract"][:300] if paper["abstract"] else ""
        )

    print(f"Added {G.number_of_nodes()} nodes")

    # Compute similarity between every pair of papers
    sim_matrix = cosine_similarity(embeddings)

    # Add edges where similarity exceeds threshold
    edge_count = 0
    for i in range(len(metadata)):
        for j in range(i + 1, len(metadata)):
            sim = sim_matrix[i][j]
            if sim >= SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=float(sim))
                edge_count += 1

    print(f"Added {edge_count} edges (similarity >= {SIMILARITY_THRESHOLD})")
    return G, sim_matrix

def detect_clusters(G):
    print("\nDetecting research clusters...")

    # Find connected components (natural clusters)
    components = list(nx.connected_components(G))
    components = sorted(components, key=len, reverse=True)

    print(f"Found {len(components)} research clusters")
    for i, comp in enumerate(components[:5]):
        print(f"  Cluster {i+1}: {len(comp)} papers")

    return components

def compute_importance(G, metadata):
    """Rank papers by their centrality in the graph."""
    if G.number_of_edges() == 0:
        return {}

    # Degree centrality = how many connections a paper has
    degree_centrality = nx.degree_centrality(G)

    # Betweenness = how often a paper bridges different clusters
    betweenness = nx.betweenness_centrality(G, normalized=True)

    importance = {}
    for node in G.nodes():
        importance[node] = {
            "degree_centrality"     : round(degree_centrality.get(node, 0), 4),
            "betweenness_centrality": round(betweenness.get(node, 0), 4),
            "combined_score"        : round(
                0.5 * degree_centrality.get(node, 0) +
                0.5 * betweenness.get(node, 0), 4
            )
        }

    return importance

def save_graph(G, components, importance, output_file):
    # Convert graph to serializable format
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id"          : node_id,
            "title"       : data.get("title", ""),
            "year"        : data.get("year"),
            "citations"   : data.get("citations", 0),
            "keywords"    : data.get("keywords", []),
            "keyword_count": data.get("keyword_count", 0),
            "abstract"    : data.get("abstract", ""),
            "degree_centrality"     : importance.get(node_id, {}).get("degree_centrality", 0),
            "betweenness_centrality": importance.get(node_id, {}).get("betweenness_centrality", 0),
            "combined_score"        : importance.get(node_id, {}).get("combined_score", 0),
            "cluster"     : next(
                (i for i, comp in enumerate(components) if node_id in comp), -1
            )
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": round(data.get("weight", 0), 4)
        })

    graph_data = {
        "nodes"        : nodes,
        "edges"        : edges,
        "num_nodes"    : G.number_of_nodes(),
        "num_edges"    : G.number_of_edges(),
        "num_clusters" : len(components),
        "cluster_sizes": [len(c) for c in components]
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved graph to {output_file}")

def print_summary(G, components, importance, metadata):
    print("\n--- Knowledge Graph Summary ---")
    print(f"Total nodes (papers) : {G.number_of_nodes()}")
    print(f"Total edges          : {G.number_of_edges()}")
    print(f"Research clusters    : {len(components)}")

    if importance:
        # Top 5 most important papers
        sorted_nodes = sorted(
            importance.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )

        print("\nTop 5 most central papers in BCI research:")
        for node_id, scores in sorted_nodes[:5]:
            title = metadata[node_id]["title"][:65]
            score = scores["combined_score"]
            print(f"  [{score:.4f}] {title}...")

def main():
    print("Building BCI Knowledge Graph...\n")

    # Load data
    embeddings, metadata = load_data()

    # Build graph
    G, sim_matrix = build_graph(embeddings, metadata)

    # Detect clusters
    components = detect_clusters(G)

    # Compute importance scores
    importance = compute_importance(G, metadata)

    # Save graph
    save_graph(G, components, importance, GRAPH_FILE)

    # Print summary
    print_summary(G, components, importance, metadata)

    print("\nKnowledge graph complete.")

if __name__ == "__main__":
    main()