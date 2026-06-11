import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "NIIE — National Innovation Intelligence Engine",
    page_icon  = "🧠",
    layout     = "wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_scores():
    with open("data/processed/bci_scores.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_graph():
    with open("data/processed/bci_graph.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_embeddings():
    return np.load("data/processed/bci_embeddings.npy")

scores_data = load_scores()
graph_data  = load_graph()
embeddings  = load_embeddings()

all_scores = scores_data["all_scores"]
gaps       = scores_data["research_gaps"]
nodes      = graph_data["nodes"]
edges      = graph_data["edges"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#00D4FF; font-size:2.4rem; margin-bottom:0'>
    🧠 NIIE — National Innovation Intelligence Engine
</h1>
<p style='text-align:center; color:#888; font-size:1rem; margin-top:4px'>
    AI-Powered Research Gap & Innovation Opportunity Discovery System
</p>
<hr style='border-color:#222; margin: 12px 0 24px 0'>
""", unsafe_allow_html=True)

# ── Domain badge ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-bottom:24px'>
    <span style='background:#1a1a2e; border:1px solid #00D4FF; color:#00D4FF;
                 padding:6px 20px; border-radius:20px; font-size:0.9rem'>
        🔬 Domain: Brain-Computer Interfaces (BCI)
    </span>
</div>
""", unsafe_allow_html=True)

# ── Top KPI metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 Papers Analysed",   graph_data["num_nodes"])
col2.metric("🔗 Graph Connections", graph_data["num_edges"])
col3.metric("🗂️ Research Clusters", graph_data["num_clusters"])
col4.metric("💡 Gaps Detected",     scores_data["total_gaps"])

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Knowledge Graph",
    "📊 Innovation Scores",
    "💡 Research Gaps",
    "🏆 Top Opportunities"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("BCI Research Knowledge Graph")
    st.caption("Each node is a paper. Edges connect similar papers. Clusters emerge automatically.")

    # Build node positions using embeddings (2D projection via PCA)
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    # Node colours by cluster
    cluster_colors = [
        "#00D4FF","#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF",
        "#FF8B94","#B4F8C8","#FBE7C6","#A0C4FF","#BDB2FF",
        "#FFC6FF","#FFADAD"
    ]

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

    for node in nodes:
        idx = node["id"]
        node_x.append(float(coords[idx, 0]))
        node_y.append(float(coords[idx, 1]))
        cluster = node.get("cluster", 0)
        color   = cluster_colors[cluster % len(cluster_colors)]
        node_color.append(color)
        # Size by combined centrality score
        size = 8 + node.get("combined_score", 0) * 60
        node_size.append(size)
        node_text.append(
            f"<b>{node['title'][:50]}...</b><br>"
            f"Year: {node.get('year','N/A')} | "
            f"Citations: {node.get('citations',0)}<br>"
            f"Cluster: {cluster} | "
            f"Keywords: {', '.join(node.get('keywords',[])[:3])}"
        )

    # Build edge traces (only draw a sample to keep it fast)
    edge_x, edge_y = [], []
    sampled_edges  = edges[:300]          # cap at 300 for performance
    node_pos       = {n["id"]: (float(coords[n["id"],0]),
                                float(coords[n["id"],1])) for n in nodes}

    for edge in sampled_edges:
        x0, y0 = node_pos[edge["source"]]
        x1, y1 = node_pos[edge["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#333"),
        hoverinfo="none",
        name="Connections"
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=0.5, color="#fff")
        ),
        name="Papers"
    )

    fig_graph = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend   = False,
            hovermode    = "closest",
            height       = 580,
            paper_bgcolor= "#0e1117",
            plot_bgcolor = "#0e1117",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
    )

    st.plotly_chart(fig_graph, use_container_width=True)
    st.caption("💡 Hover over any node to see paper details. Colour = research cluster.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — INNOVATION SCORES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Innovation Opportunity Scores — All Papers")

    df = pd.DataFrame([{
        "Title"              : p["title"][:60] + "...",
        "Year"               : p["year"],
        "Citations"          : p["citations"],
        "Momentum"           : p["scores"]["research_momentum"],
        "Citation Density"   : p["scores"]["citation_density"],
        "Keyword Novelty"    : p["scores"]["keyword_novelty"],
        "Isolation Score"    : p["scores"]["isolation_score"],
        "Innovation Score"   : p["scores"]["innovation_opportunity_score"],
        "Keywords"           : ", ".join(p["bci_keywords"][:4])
    } for p in all_scores])

    # Score distribution chart
    fig_dist = px.histogram(
        df, x="Innovation Score", nbins=20,
        title="Distribution of Innovation Opportunity Scores",
        color_discrete_sequence=["#00D4FF"],
        template="plotly_dark"
    )
    fig_dist.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor ="#0e1117",
        height=300
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Full table
    st.dataframe(
        df.style.background_gradient(subset=["Innovation Score"], cmap="Blues"),
        use_container_width=True,
        height=400
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESEARCH GAPS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"💡 {len(gaps)} Research Gaps Detected")
    st.caption("These are underexplored areas: high novelty, low citations, isolated in the graph.")

    if gaps:
        for i, gap in enumerate(gaps[:10], 1):
            score = gap["scores"]["innovation_opportunity_score"]
            with st.expander(f"Gap #{i} — Score: {score:.4f} — {gap['title'][:60]}..."):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Title:** {gap['title']}")
                    st.markdown(f"**Year:** {gap['year']} | **Citations:** {gap['citations']}")
                    abstract = gap.get("abstract", "") or "No abstract available."
                    st.markdown(f"**Abstract:** {abstract[:400]}...")
                with col_b:
                    st.markdown("**Signal Scores:**")
                    signal_df = pd.DataFrame({
                        "Signal": [
                            "Research Momentum",
                            "Citation Density",
                            "Keyword Novelty",
                            "Isolation Score"
                        ],
                        "Score": [
                            gap["scores"]["research_momentum"],
                            gap["scores"]["citation_density"],
                            gap["scores"]["keyword_novelty"],
                            gap["scores"]["isolation_score"]
                        ]
                    })
                    fig_bar = px.bar(
                        signal_df, x="Score", y="Signal",
                        orientation="h",
                        color="Score",
                        color_continuous_scale="Blues",
                        template="plotly_dark",
                        height=200
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="#0e1117",
                        plot_bgcolor ="#0e1117",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown(f"**BCI Keywords:** {', '.join(gap['bci_keywords'])}")
    else:
        st.info("No gaps detected with current thresholds.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — TOP OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🏆 Top 10 Innovation Opportunities")
    st.caption("Highest scoring papers across all four signals combined.")

    top10 = all_scores[:10]

    for rank, paper in enumerate(top10, 1):
        score = paper["scores"]["innovation_opportunity_score"]
        color = "#00D4FF" if rank <= 3 else "#888"
        st.markdown(f"""
        <div style='border:1px solid #222; border-radius:8px;
                    padding:12px 16px; margin-bottom:10px;
                    background:#0e1117'>
            <span style='color:{color}; font-size:1.1rem; font-weight:bold'>
                #{rank} — {score:.4f}
            </span>
            <span style='color:#fff; margin-left:10px'>{paper['title']}</span>
            <br>
            <span style='color:#888; font-size:0.8rem'>
                📅 {paper['year']} &nbsp;|&nbsp;
                📚 {paper['citations']} citations &nbsp;|&nbsp;
                🏷️ {', '.join(paper['bci_keywords'][:4])}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart for top 3
    st.markdown("### Signal Breakdown — Top 3 Papers")
    categories = ["Momentum", "Citations", "Novelty", "Isolation"]

    fig_radar = go.Figure()
    colors    = ["#00D4FF", "#FF6B6B", "#4ECDC4"]

    for i, paper in enumerate(top10[:3]):
        s = paper["scores"]
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                s["research_momentum"],
                s["citation_density"],
                s["keyword_novelty"],
                s["isolation_score"]
            ],
            theta=categories,
            fill="toself",
            name=paper["title"][:35] + "...",
            line_color=colors[i]
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        font_color="#fff",
        height=420,
        legend=dict(font=dict(size=9))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#444; font-size:0.8rem'>
    NIIE — National Innovation Intelligence Engine &nbsp;|&nbsp;
    Built with Python · Streamlit · NetworkX · Sentence-Transformers · Plotly
</p>
""", unsafe_allow_html=True)