import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import re
import html
import sys
import unicodedata
# No iframe-based JS — keep UI fixes CSS-only to work inside Streamlit's page
from datetime import datetime
import pickle
import hashlib
from pathlib import Path

sys.path.append("src")
HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ── HISTORY HELPERS ───────────────────────────────────────────────────────────
def save_to_history(query, result):
    """Save a pipeline result to disk."""
    key      = hashlib.md5(query.lower().strip().encode()).hexdigest()[:10]
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = HISTORY_DIR / f"{key}.pkl"
    payload  = {
        "query"    : query,
        "timestamp": ts,
        "papers"   : result["papers"],
        "graph"    : {
            "num_nodes"   : result["graph"]["num_nodes"],
            "num_edges"   : result["graph"]["num_edges"],
            "num_clusters": result["graph"]["num_clusters"],
        },
        "total_gaps"       : result["scores"]["total_gaps"],
        "top_opportunities": result["scores"]["top_opportunities"][:5],
        "research_gaps"    : result["scores"]["research_gaps"][:5],
        "all_scores"       : result["scores"]["all_scores"],
        "innovations"      : result["innovations"],
    }
    with open(filename, "wb") as f:
        pickle.dump(payload, f)
    return ts

def load_history():
    """Load all saved history entries sorted by newest first."""
    entries = []
    for file in HISTORY_DIR.glob("*.pkl"):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
            data["_file"] = file
            entries.append(data)
        except Exception:
            continue
    return sorted(entries, key=lambda x: x["timestamp"], reverse=True)

def delete_history_entry(file_path):
    try:
        Path(file_path).unlink()
    except Exception:
        pass

st.set_page_config(
    page_title="NIIE — National Innovation Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME DEFINITIONS ─────────────────────────────────────────────────────────
THEMES = {
    "Midnight Pro": {
        "bg"         : "#0a0a0f",
        "surface"    : "#13131a",
        "surface2"   : "#1a1a25",
        "border"     : "#2a2a3a",
        "accent"     : "#6366f1",
        "accent2"    : "#8b5cf6",
        "text"       : "#e2e8f0",
        "muted"      : "#64748b",
        "success"    : "#10b981",
        "chart_seq"  : ["#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#ddd6fe"],
        "gradient"   : "linear-gradient(135deg, #6366f1, #8b5cf6)",
    },
    "Arctic Light": {
        "bg"         : "#f8fafc",
        "surface"    : "#ffffff",
        "surface2"   : "#f1f5f9",
        "border"     : "#e2e8f0",
        "accent"     : "#0ea5e9",
        "accent2"    : "#38bdf8",
        "text"       : "#0f172a",
        "muted"      : "#94a3b8",
        "success"    : "#059669",
        "chart_seq"  : ["#0ea5e9","#38bdf8","#7dd3fc","#bae6fd","#e0f2fe"],
        "gradient"   : "linear-gradient(135deg, #0ea5e9, #38bdf8)",
    },
    "Forest Deep": {
        "bg"         : "#0d1117",
        "surface"    : "#161b22",
        "surface2"   : "#21262d",
        "border"     : "#30363d",
        "accent"     : "#2ea043",
        "accent2"    : "#3fb950",
        "text"       : "#c9d1d9",
        "muted"      : "#6e7681",
        "success"    : "#2ea043",
        "chart_seq"  : ["#2ea043","#3fb950","#56d364","#7ee787","#b5efb5"],
        "gradient"   : "linear-gradient(135deg, #2ea043, #3fb950)",
    },
    "Amber Dusk": {
        "bg"         : "#0c0a00",
        "surface"    : "#1a1600",
        "surface2"   : "#252000",
        "border"     : "#3d3500",
        "accent"     : "#f59e0b",
        "accent2"    : "#fbbf24",
        "text"       : "#fef3c7",
        "muted"      : "#92400e",
        "success"    : "#10b981",
        "chart_seq"  : ["#f59e0b","#fbbf24","#fcd34d","#fde68a","#fef3c7"],
        "gradient"   : "linear-gradient(135deg, #f59e0b, #fbbf24)",
    },
    "Rose Gold": {
        "bg"         : "#0f0608",
        "surface"    : "#1a0d10",
        "surface2"   : "#251318",
        "border"     : "#3d1e24",
        "accent"     : "#f43f5e",
        "accent2"    : "#fb7185",
        "text"       : "#fce7f0",
        "muted"      : "#9f1239",
        "success"    : "#10b981",
        "chart_seq"  : ["#f43f5e","#fb7185","#fda4af","#fecdd3","#fff1f2"],
        "gradient"   : "linear-gradient(135deg, #f43f5e, #fb7185)",
    },
}

# ── SIDEBAR THEME PICKER ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎨 Theme")
    selected_theme = st.radio(
        "", list(THEMES.keys()), index=0, label_visibility="collapsed"
    )
    T = THEMES[selected_theme]

    st.markdown("---")
    st.markdown("### 📥 Export Report")

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Building report..."):
            try:
                import sys
                sys.path.append("src")
                from export_report import generate_report

                with open("data/processed/bci_scores.json","r",encoding="utf-8") as f:
                    _scores = json.load(f)
                with open("data/processed/bci_graph.json","r",encoding="utf-8") as f:
                    _graph = json.load(f)

                _innovations = []
                if os.path.exists("data/processed/bci_innovations.json"):
                    with open("data/processed/bci_innovations.json","r",encoding="utf-8") as f:
                        _innovations = json.load(f)

                pdf_path = generate_report(
                    _scores, _graph, _innovations,
                    domain="Brain-Computer Interfaces",
                    output_path="reports/NIIE_Innovation_Report.pdf"
                )

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="NIIE_Innovation_Report.pdf",
                    mime="application/pdf"
                )
                st.success("Ready!")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        f"<span style='color:{T['muted']}; font-size:0.8rem;'>"
        "NIIE v1.0 · Built with Python, Streamlit, NetworkX, "
        "Sentence-Transformers, Plotly, Groq LLaMA 3.3"
        "</span>",
        unsafe_allow_html=True
    )

# ── INJECT CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

*, *::before, *::after {{
    transition: background-color 0.4s ease, color 0.3s ease, border-color 0.3s ease !important;
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {{
    font-family: 'Inter', sans-serif !important;
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
}}

[data-testid="stIconMaterial"] {{
    font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
    font-style: normal !important;
    font-weight: normal !important;
    font-variant: normal !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    word-wrap: normal !important;
    white-space: nowrap !important;
    direction: ltr !important;
    font-feature-settings: 'liga' !important;
}}

/* Force material icon font onto common button/icon spans so ligature names render as glyphs */
button [data-testid="stIconMaterial"],
button span[translate="no"],
.stButton button [data-testid="stIconMaterial"],
.stButton button span[translate="no"],
[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] summary span[translate="no"] {{
    font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
    font-size: 1.1em !important;
    color: {T['text']} !important;
    display: inline-block !important;
    line-height: 1 !important;
    width: 1.1em !important;
    min-width: 1.1em !important;
    text-indent: 0 !important;
}}

[data-testid="stHeader"] {{
    background-color: {T['bg']} !important;
    border-bottom: 1px solid {T['border']} !important;
}}

[data-testid="stSidebar"] {{
    background-color: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
}}

[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
}}

.block-container {{
    padding-top: 2rem !important;
    max-width: 1300px !important;
}}

/* Tabs */
[data-testid="stTabs"] button {{
    color: {T['muted']} !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    border-radius: 0 !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stTabs"] button:hover {{
    color: {T['text']} !important;
    background: {T['surface2']} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {T['accent']} !important;
    border-bottom: 2px solid {T['accent']} !important;
    background: transparent !important;
    font-weight: 600 !important;
}}
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 1px solid {T['border']} !important;
    gap: 0 !important;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    transition: all 0.3s ease !important;
}}
[data-testid="stMetric"]:hover {{
    border-color: {T['accent']} !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stMetricLabel"] p {{
    color: {T['muted']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}
[data-testid="stMetricValue"] {{
    color: {T['accent']} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: {T['accent']} !important;
}}
[data-testid="stExpander"] summary p {{
    color: {T['text']} !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}}

/* Ensure the expander summary aligns icon and text and avoids overlap */
[data-testid="stExpander"] summary {{
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 12px !important;
}}
[data-testid="stExpander"] summary p {{
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
}}

[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] summary span[translate="no"] {{
    position: relative !important;
    display: inline-block !important;
    width: 1.1em !important;
    min-width: 1.1em !important;
    height: 1.1em !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}}

[data-testid="stExpander"] details[open] summary [data-testid="stIconMaterial"]::after,
[data-testid="stExpander"] details[open] summary span[translate="no"]::after {{
    content: '▾' !important;
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    color: {T['text']} !important;
    font-size: 1.1em !important;
    line-height: 1 !important;
}}

[data-testid="stExpander"] details:not([open]) summary [data-testid="stIconMaterial"]::after,
[data-testid="stExpander"] details:not([open]) summary span[translate="no"]::after {{
    content: '▸' !important;
    position: absolute !important;
    left: 0 !important;
    right: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    color: {T['text']} !important;
    font-size: 1.1em !important;
    line-height: 1 !important;
}}

/* Text input */
[data-testid="stTextInput"] input {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px {T['accent']}22 !important;
}}

/* Primary button */
[data-testid="stButton"] button[kind="primary"] {{
    background: {T['gradient']} !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stButton"] button[kind="primary"]:hover {{
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px {T['accent']}44 !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}

/* Radio */
[data-testid="stRadio"] label {{
    color: {T['text']} !important;
}}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {{
    background: {T['gradient']} !important;
}}

/* Alerts */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border: 1px solid {T['border']} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {T['bg']}; }}
::-webkit-scrollbar-thumb {{
    background: {T['border']};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {T['accent']}; }}

hr {{ border-color: {T['border']} !important; margin: 20px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# Note: iframe JS removed - use CSS to enforce Material Icons font and hide
# raw ligature text where possible. This avoids cross-frame security/sandboxing issues.

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_scores():
    with open("data/processed/bci_scores.json","r",encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_graph():
    with open("data/processed/bci_graph.json","r",encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_embeddings():
    return np.load("data/processed/bci_embeddings.npy")

scores_data = load_scores()
graph_data  = load_graph()
embeddings  = load_embeddings()
all_scores  = scores_data["all_scores"]
gaps        = scores_data["research_gaps"]
nodes       = graph_data["nodes"]
edges       = graph_data["edges"]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding: 1.5rem 0 1rem 0; border-bottom: 1px solid {T['border']}; margin-bottom: 1.5rem;'>
    <div style='display:flex; align-items:center; gap:14px;'>
        <div style='width:42px; height:42px; background:{T['gradient']};
                    border-radius:10px; display:flex; align-items:center;
                    justify-content:center; font-size:1.4rem; flex-shrink:0;'>🧠</div>
        <div>
            <h1 style='margin:0; font-size:1.5rem; font-weight:800;
                       color:{T['text']}; letter-spacing:-0.5px;'>
                NIIE
                <span style='font-weight:300; color:{T['muted']}; font-size:1rem;
                             margin-left:8px;'>
                    National Innovation Intelligence Engine
                </span>
            </h1>
            <p style='margin:2px 0 0 0; color:{T['muted']}; font-size:0.78rem;'>
                AI-Powered Research Gap & Innovation Opportunity Discovery System
                &nbsp;·&nbsp;
                <span style='color:{T['accent']}; font-weight:600;'>
                    Brain-Computer Interfaces
                </span>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Papers Analysed",   graph_data["num_nodes"])
k2.metric("Graph Connections", graph_data["num_edges"])
k3.metric("Research Clusters", graph_data["num_clusters"])
k4.metric("Gaps Detected",     scores_data["total_gaps"])

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🔍  Live Query",
    "📚  Literature Search",
    "⏱️  Timing Advisor",
    "🗺️  Roadmap",
    "⚔️  Compare Domains",
    "📂  History",
    "🌐  Knowledge Graph",
    "📊  Scores",
    "💡  Research Gaps",
    "🏆  Opportunities",
    "🤖  AI Proposals",
])

def card(content, padding="16px 20px"):
    return f"""<div style='background:{T['surface']}; border:1px solid {T['border']};
    border-radius:10px; padding:{padding}; margin-bottom:10px;'>{content}</div>"""

def accent_tag(text):
    return (f"<span style='background:{T['accent']}18; color:{T['accent']}; "
            f"border:1px solid {T['accent']}33; padding:2px 9px; "
            f"border-radius:20px; font-size:0.72rem; font-weight:600; "
            f"margin-right:4px;'>{text}</span>")


def hex_to_rgba(hex_color, alpha=0.14):
    """Convert hex color to rgba string. Safe for all Plotly fillcolor uses."""
    try:
        hex_color = hex_color.lstrip('#')
        # Handle shorthand hex like #fff
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        if len(hex_color) != 6:
            return f"rgba(99,102,241,{alpha})"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(99,102,241,{alpha})"

def clean_text(value, max_len=None):
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize('NFKC', value)
    value = ''.join(
        c for c in value
        if not unicodedata.category(c).startswith(('C', 'M'))
    )
    # Strip leading punctuation/symbols that may be artifacts from generation
    value = re.sub(r'^[^\w\d\s]+', '', value)
    # Collapse repeated whitespace
    value = re.sub(r'\s+', ' ', value).strip()
    value = html.escape(value)
    return value[:max_len] if max_len is not None else value


def plot_cfg(fig, height=320):
    fig.update_layout(
        paper_bgcolor=T['surface'],
        plot_bgcolor=T['surface2'],
        font_color=T['muted'],
        height=height,
        margin=dict(l=16,r=16,t=36,b=16),
    )
    fig.update_xaxes(gridcolor=T['border'], zerolinecolor=T['border'])
    fig.update_yaxes(gridcolor=T['border'], zerolinecolor=T['border'])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — LIVE QUERY
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    st.markdown(f"""
    {card(f'''
    <h3 style="margin:0 0 4px 0; color:{T['text']}; font-size:1.05rem; font-weight:700;">
        Live Innovation Discovery
    </h3>
    <p style="margin:0; color:{T['muted']}; font-size:0.85rem; line-height:1.6;">
        Enter any research domain. NIIE fetches real papers, builds a knowledge graph,
        scores innovation opportunities, and generates AI-powered proposals — in real time.
    </p>
    ''', "20px 24px")}
    """, unsafe_allow_html=True)

    query_input = st.text_input(
        "", placeholder="e.g.  quantum computing  ·  gene editing  ·  neuromorphic chips",
        key="live_query", label_visibility="collapsed"
    )
    run_btn = st.button("Run Analysis →", type="primary")

    if run_btn and query_input.strip():
        try:
            from pipeline import run_pipeline
        except Exception as e:
            st.error(f"Could not import pipeline or required dependencies: {e}")
            st.stop()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        log   = st.empty()
        prog  = st.progress(0)
        steps = []
        cnt   = [0]

        def upd(msg):
            steps.append(msg)
            cnt[0] += 1
            html = "".join([
                f"<div style='padding:3px 0; color:"
                f"{'#10b981' if '✅' in s else T['muted']}; font-size:0.82rem;'>{s}</div>"
                for s in steps[-7:]
            ])
            log.markdown(
                f"<div style='background:{T['surface']}; border:1px solid {T['border']}; "
                f"border-radius:8px; padding:14px 18px;'>{html}</div>",
                unsafe_allow_html=True
            )
            prog.progress(min(cnt[0]/12, 1.0))

        with st.spinner(""):
            result, err = run_pipeline(query_input.strip(), upd)
        prog.progress(1.0)

        if not err and result:
            save_to_history(query_input.strip(), result)

        if err:
            st.error(err)
        else:
            st.markdown(f"""
            <div style='background:{T['success']}18; border:1px solid {T['success']}44;
                        border-radius:8px; padding:12px 18px; margin:12px 0;'>
                <span style='color:{T['success']}; font-weight:700;'>✓ Complete</span>
                <span style='color:{T['muted']}; margin-left:8px; font-size:0.88rem;'>
                    Analysis for <strong style='color:{T['text']}'>{query_input}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)

            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Papers",   result["papers"])
            r2.metric("Edges",    result["graph"]["num_edges"])
            r3.metric("Clusters", result["graph"]["num_clusters"])
            r4.metric("Gaps",     result["scores"]["total_gaps"])

            st.markdown(f"<p style='color:{T['muted']}; font-size:0.8rem; "
                        f"text-transform:uppercase; letter-spacing:1px; "
                        f"margin:16px 0 8px 0; font-weight:600;'>Top Opportunities</p>",
                        unsafe_allow_html=True)

            for i, p in enumerate(result["scores"]["top_opportunities"][:5], 1):
                sc = p["scores"]["innovation_opportunity_score"]
                bc = T['accent'] if i <= 3 else T['border']
                st.markdown(f"""
                <div style='background:{T['surface']}; border:1px solid {T['border']};
                            border-left:3px solid {bc}; border-radius:8px;
                            padding:12px 18px; margin-bottom:6px;'>
                    <span style='color:{T['accent']}; font-weight:700; font-size:0.9rem;'>
                        #{i} &nbsp; {sc:.4f}
                    </span>
                    <span style='color:{T['text']}; margin-left:10px; font-size:0.88rem;'>
                        {p['title']}
                    </span>
                    <br>
                    <span style='color:{T['muted']}; font-size:0.76rem;'>
                        {p['year']} · {p['citations']} citations
                    </span>
                </div>
                """, unsafe_allow_html=True)

            if result["innovations"]:
                st.markdown(f"<p style='color:{T['muted']}; font-size:0.8rem; "
                            f"text-transform:uppercase; letter-spacing:1px; "
                            f"margin:16px 0 8px 0; font-weight:600;'>AI Proposals</p>",
                            unsafe_allow_html=True)
                for i, inv in enumerate(result["innovations"], 1):
                    with st.expander(f"Proposal #{i} — {inv['gap_title'][:55]}..."):
                        st.markdown(inv["proposal"]["raw"])

    elif run_btn:
        st.warning("Please enter a topic.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LITERATURE SEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            📚 Literature Search & Summarization
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Search any research question. NIIE finds relevant papers,
            generates AI summaries, and maps the research landscape —
            replacing Elicit, ResearchRabbit, and Semantic Scholar in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    lit_query = st.text_input(
        "",
        placeholder="e.g.  What are the latest non-invasive BCI methods for motor rehabilitation?",
        key="lit_query",
        label_visibility="collapsed"
    )
    lit_btn = st.button("🔍  Search Literature", type="primary")

    if lit_btn and lit_query.strip():
        from literature_search import run_literature_search

        log  = st.empty()
        steps = []

        def upd_lit(msg):
            steps.append(msg)
            log.markdown(
                f"<div style='background:{T['surface']}; "
                f"border:1px solid {T['border']}; border-radius:8px; "
                f"padding:12px 16px; font-size:0.82rem;'>"
                + "".join([
                    f"<div style='color:"
                    f"{'#10b981' if '✅' in s or '🎉' in s else T['muted']}'>"
                    f"{s}</div>"
                    for s in steps[-6:]
                ]) + "</div>",
                unsafe_allow_html=True
            )

        with st.spinner(""):
            lit_result, lit_err = run_literature_search(
                lit_query.strip(),
                summarize_top=5,
                status_callback=upd_lit
            )

        if lit_err:
            st.error(lit_err)
        elif lit_result:
            st.markdown("---")

            # Stats row
            s1, s2, s3 = st.columns(3)
            s1.metric("Papers Found",    lit_result["total"])
            s2.metric("Papers Summarized", 5)
            s3.metric("Query", lit_query.strip()[:20] + "...")

            # Research landscape
            st.markdown(f"""
            <div style='background:{T['surface']};
                        border:1px solid {T['border']};
                        border-left:3px solid {T['accent']};
                        border-radius:8px; padding:20px 24px;
                        margin:16px 0;'>
                <p style='color:{T['accent']}; font-weight:700;
                          font-size:0.75rem; text-transform:uppercase;
                          letter-spacing:1px; margin:0 0 12px 0;'>
                    🌐 Research Landscape Overview
                </p>
                <p style='color:{T['text']}; font-size:0.88rem;
                          line-height:1.7; margin:0;'>
                    {lit_result['landscape'].replace(chr(10), '<br>')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Papers
            st.markdown(f"""
            <p style='color:{T['muted']}; font-size:0.75rem;
                      text-transform:uppercase; letter-spacing:1px;
                      margin:16px 0 10px 0; font-weight:600;'>
                Papers — Ranked by Citations
            </p>
            """, unsafe_allow_html=True)

            papers = lit_result["papers"]

            for i, paper in enumerate(papers, 1):
                title     = paper.get("title","") or "Untitled"
                year      = paper.get("year","N/A")
                citations = paper.get("citationCount", 0) or 0
                abstract  = paper.get("abstract","") or ""
                summary   = paper.get("ai_summary","")
                authors   = paper.get("authors",[]) or []
                author_str = ", ".join([
                    a.get("name","") for a in authors[:3]
                ])
                if len(authors) > 3:
                    author_str += f" +{len(authors)-3} more"

                is_summarized = i <= 5
                border = T['accent'] if i <= 3 else T['border']

                with st.expander(
                    f"{'⭐ ' if i<=3 else ''}#{i}  ·  "
                    f"{title[:65]}{'...' if len(title)>65 else ''}",
                    expanded=(i == 1)
                ):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Year",      year)
                    m2.metric("Citations", citations)
                    m3.metric("Rank",      f"#{i}")

                    if author_str:
                        st.markdown(
                            f"<p style='color:{T['muted']}; "
                            f"font-size:0.78rem; margin:8px 0;'>"
                            f"👤 {author_str}</p>",
                            unsafe_allow_html=True
                        )

                    if is_summarized and summary:
                        st.markdown(f"""
                        <div style='background:{T['accent']}08;
                                    border:1px solid {T['accent']}22;
                                    border-radius:8px;
                                    padding:14px 16px; margin:10px 0;'>
                            <p style='color:{T['accent']}; font-weight:700;
                                      font-size:0.72rem; text-transform:uppercase;
                                      letter-spacing:1px; margin:0 0 8px 0;'>
                                🤖 AI Summary
                            </p>
                            <p style='color:{T['text']}; font-size:0.86rem;
                                      line-height:1.65; margin:0;'>
                                {summary}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    if abstract:
                        st.markdown(f"""
                        <div style='margin-top:10px;'>
                            <p style='color:{T['muted']}; font-weight:600;
                                      font-size:0.72rem; text-transform:uppercase;
                                      letter-spacing:1px; margin:0 0 6px 0;'>
                                Abstract
                            </p>
                            <p style='color:{T['muted']}; font-size:0.83rem;
                                      line-height:1.6; margin:0;'>
                                {abstract[:500]}{'...' if len(abstract)>500 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # External IDs
                    ext_ids = paper.get("externalIds", {}) or {}
                    doi = ext_ids.get("DOI","")
                    if doi:
                        st.markdown(
                            f"<a href='https://doi.org/{doi}' "
                            f"target='_blank' "
                            f"style='color:{T['accent']}; "
                            f"font-size:0.78rem;'>"
                            f"🔗 View Paper (DOI)</a>",
                            unsafe_allow_html=True
                        )

    elif lit_btn:
        st.warning("Please enter a search query.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — INNOVATION TIMING ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            ⏱️ Innovation Timing Advisor
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Is now the right time to publish, patent, or pivot?
            NIIE analyzes publication momentum curves and patent density
            to tell you exactly where a field stands — and what to do next.
            No other tool does this.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Use existing BCI data for timing analysis
    if all_scores:
        years = [p["year"] for p in all_scores if p.get("year")]
        from collections import Counter
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        counts       = [year_counts[y] for y in sorted_years]

        if len(sorted_years) >= 3:
            # Momentum calculation
            recent_3  = sum(counts[-3:])
            previous_3 = sum(counts[-6:-3]) if len(counts) >= 6 else sum(counts[:-3])
            momentum_ratio = recent_3 / previous_3 if previous_3 > 0 else 2.0

            # Determine field status
            if momentum_ratio >= 1.8:
                status      = "🚀 PEAK MOMENTUM"
                status_color = "#10b981"
                advice      = "This field is exploding right now. Best time to publish and file patents before saturation hits."
                action      = "PUBLISH NOW · FILE PATENT NOW"
            elif momentum_ratio >= 1.2:
                status      = "📈 GROWING"
                status_color = "#6366f1"
                advice      = "Research activity is accelerating. Strong window to establish presence before the field peaks."
                action      = "PUBLISH SOON · START PATENT PROCESS"
            elif momentum_ratio >= 0.8:
                status      = "⚖️ STABLE"
                status_color = "#f59e0b"
                advice      = "Field is mature and consistent. Focus on differentiation — find the gaps that others are missing."
                action      = "FIND GAPS · TARGET NICHE"
            else:
                status      = "📉 DECLINING"
                status_color = "#f43f5e"
                advice      = "Research activity is slowing. Pivot to adjacent emerging areas or find the contrarian angle."
                action      = "PIVOT · FIND ADJACENT FIELD"

            # Status banner
            st.markdown(f"""
            <div style='background:{status_color}12;
                        border:2px solid {status_color}44;
                        border-radius:12px; padding:20px 24px;
                        margin-bottom:20px; text-align:center;'>
                <p style='color:{status_color}; font-size:1.6rem;
                          font-weight:800; margin:0 0 6px 0;'>
                    {status}
                </p>
                <p style='color:{T['text']}; font-size:0.95rem;
                          margin:0 0 10px 0; line-height:1.6;'>
                    {advice}
                </p>
                <div style='background:{status_color}22;
                            border-radius:6px; padding:8px 16px;
                            display:inline-block;'>
                    <span style='color:{status_color}; font-weight:800;
                                 font-size:0.85rem; letter-spacing:1px;'>
                        RECOMMENDED ACTION: {action}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Momentum metrics
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Recent 3yr Papers",   recent_3)
            t2.metric("Previous 3yr Papers", previous_3)
            t3.metric("Momentum Ratio",      f"{momentum_ratio:.2f}x")
            t4.metric("Peak Year",
                      sorted_years[counts.index(max(counts))])

            st.markdown("<div style='height:12px'></div>",
                        unsafe_allow_html=True)

            # Publication trend chart
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_time = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Publication Volume by Year",
                                "Cumulative Research Growth"]
            )

            # Bar chart
            fig_time.add_trace(
                go.Bar(
                    x=sorted_years, y=counts,
                    marker_color=[
                        status_color if y >= sorted_years[-3]
                        else T['border']
                        for y in sorted_years
                    ],
                    name="Papers/Year"
                ),
                row=1, col=1
            )

            # Cumulative line
            cumulative = []
            total = 0
            for c in counts:
                total += c
                cumulative.append(total)

            fig_time.add_trace(
                go.Scatter(
                    x=sorted_years, y=cumulative,
                    mode="lines+markers",
                    line=dict(color=T['accent'], width=2.5),
                    marker=dict(size=6, color=T['accent']),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(T['accent'], 0.1),
                    name="Cumulative"
                ),
                row=1, col=2
            )

            fig_time.update_layout(
                paper_bgcolor=T['surface'],
                plot_bgcolor=T['surface2'],
                font_color=T['muted'],
                height=320,
                showlegend=False,
                margin=dict(l=10,r=10,t=40,b=10)
            )
            fig_time.update_xaxes(gridcolor=T['border'],
                                   color=T['muted'])
            fig_time.update_yaxes(gridcolor=T['border'],
                                   color=T['muted'])

            st.plotly_chart(fig_time, use_container_width=True)

            # Score evolution by year
            st.markdown(f"""
            <p style='color:{T['muted']}; font-size:0.75rem;
                      text-transform:uppercase; letter-spacing:1px;
                      margin:16px 0 8px 0; font-weight:600;'>
                Innovation Score Evolution by Year
            </p>
            """, unsafe_allow_html=True)

            year_scores = {}
            for p in all_scores:
                y = p.get("year")
                s = p["scores"]["innovation_opportunity_score"]
                if y:
                    year_scores.setdefault(y, []).append(s)

            avg_by_year = {
                y: round(sum(v)/len(v), 4)
                for y, v in year_scores.items()
            }
            ys = sorted(avg_by_year.keys())
            vs = [avg_by_year[y] for y in ys]

            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=ys, y=vs,
                mode="lines+markers",
                line=dict(color=T['accent2'], width=2.5),
                marker=dict(size=8, color=T['accent2'],
                            line=dict(width=1.5,
                                      color=T['surface'])),
                fill="tozeroy",
                fillcolor=hex_to_rgba(T['accent2'], 0.1),
                name="Avg Innovation Score"
            ))
            fig_evo.update_layout(
                paper_bgcolor=T['surface'],
                plot_bgcolor=T['surface2'],
                font_color=T['muted'],
                height=260,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(gridcolor=T['border'],
                           color=T['muted']),
                yaxis=dict(gridcolor=T['border'],
                           color=T['muted'],
                           range=[0, 1])
            )
            st.plotly_chart(fig_evo, use_container_width=True)

            # 2-year prediction
            st.markdown(f"""
            <div style='background:{T['surface']};
                        border:1px solid {T['border']};
                        border-left:3px solid {T['accent2']};
                        border-radius:8px; padding:16px 20px;
                        margin-top:8px;'>
                <p style='color:{T['accent2']}; font-weight:700;
                          font-size:0.75rem; text-transform:uppercase;
                          letter-spacing:1px; margin:0 0 8px 0;'>
                    🔮 2-Year Trajectory Prediction
                </p>
                <p style='color:{T['text']}; font-size:0.88rem;
                          line-height:1.65; margin:0;'>
                    Based on a momentum ratio of <strong>{momentum_ratio:.2f}x</strong>,
                    this field is projected to
                    {'accelerate significantly — expect 2-3x more papers by 2027 and increasing patent activity.' if momentum_ratio >= 1.5
                     else 'continue growing steadily — expect moderate increase in publications and patent filings.' if momentum_ratio >= 1.0
                     else 'plateau or contract — consider pivoting to adjacent emerging areas for maximum impact.'}
                    The current innovation score distribution suggests
                    {'high opportunity in underexplored sub-domains.' if scores_data['total_gaps'] >= 15
                     else 'moderate opportunity with targeted gap areas available.' if scores_data['total_gaps'] >= 8
                     else 'a maturing field with limited white space remaining.'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("Not enough year data for timing analysis. "
                    "Run a Live Query first.")
    else:
        st.info("No data loaded. Run a Live Query first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            📂 Query History
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Every analysis you run is saved here automatically.
            Revisit results without re-running the pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = load_history()

    if not history:
        st.markdown(f"""
        <div style='background:{T['surface']}; border:1px solid {T['border']};
                    border-radius:10px; padding:32px; text-align:center;'>
            <p style='color:{T['muted']}; font-size:0.9rem; margin:0;'>
                No history yet. Run a Live Query to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:{T['success']}12; border:1px solid {T['success']}33;
                    border-radius:8px; padding:10px 16px; margin-bottom:16px;
                    display:inline-block;'>
            <span style='color:{T['success']}; font-weight:700; font-size:0.88rem;'>
                {len(history)} saved {"analysis" if len(history)==1 else "analyses"}
            </span>
        </div>
        """, unsafe_allow_html=True)

        for entry in history:
            inn_scores = [
                p["scores"]["innovation_opportunity_score"]
                for p in entry["all_scores"]
            ]
            avg_score = round(
                sum(inn_scores)/len(inn_scores), 4
            ) if inn_scores else 0
            top_score = round(max(inn_scores), 4) if inn_scores else 0

            with st.expander(
                f"🔍  {entry['query'].title()}  ·  {entry['timestamp']}",
                expanded=False
            ):
                # KPI row
                h1,h2,h3,h4 = st.columns(4)
                h1.metric("Papers",    entry["papers"])
                h2.metric("Edges",     entry["graph"]["num_edges"])
                h3.metric("Gaps",      entry["total_gaps"])
                h4.metric("Top Score", top_score)

                st.markdown("<div style='height:8px'></div>",
                            unsafe_allow_html=True)

                left, right = st.columns([3,2])

                with left:
                    st.markdown(f"""
                    <p style='color:{T['muted']}; font-size:0.72rem;
                              text-transform:uppercase; letter-spacing:1px;
                              margin:0 0 8px 0; font-weight:600;'>
                        Top Opportunities
                    </p>
                    """, unsafe_allow_html=True)

                    for i, p in enumerate(
                        entry["top_opportunities"][:5], 1
                    ):
                        sc = p["scores"]["innovation_opportunity_score"]
                        st.markdown(f"""
                        <div style='background:{T['surface2']};
                                    border-left:2px solid {T['accent']};
                                    border-radius:0 6px 6px 0;
                                    padding:8px 12px; margin-bottom:5px;'>
                            <span style='color:{T['accent']};
                                         font-weight:700; font-size:0.8rem;'>
                                #{i} {sc:.4f}
                            </span><br>
                            <span style='color:{T['text']}; font-size:0.78rem;'>
                                {p['title'][:60]}...
                            </span><br>
                            <span style='color:{T['muted']}; font-size:0.72rem;'>
                                {p['year']} · {p['citations']} citations
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                with right:
                    # Mini score distribution
                    if inn_scores:
                        fig_h = go.Figure(go.Histogram(
                            x=inn_scores, nbinsx=15,
                            marker_color=T['accent'],
                            name="Score dist"
                        ))
                        fig_h.update_layout(
                            paper_bgcolor=T['surface2'],
                            plot_bgcolor=T['surface2'],
                            height=180,
                            margin=dict(l=10,r=10,t=10,b=10),
                            showlegend=False,
                            xaxis=dict(
                                color=T['muted'],
                                gridcolor=T['border']
                            ),
                            yaxis=dict(
                                color=T['muted'],
                                gridcolor=T['border']
                            ),
                            font_color=T['muted']
                        )
                        fig_h.update_traces(marker_line_width=0)
                        st.plotly_chart(fig_h, use_container_width=True)

                    st.markdown(f"""
                    <div style='background:{T['surface2']};
                                border:1px solid {T['border']};
                                border-radius:8px; padding:10px 14px;'>
                        <div style='display:flex; justify-content:space-between;
                                    margin-bottom:6px;'>
                            <span style='color:{T['muted']}; font-size:0.75rem;'>
                                Avg Score
                            </span>
                            <span style='color:{T['accent']};
                                         font-weight:700;'>
                                {avg_score:.4f}
                            </span>
                        </div>
                        <div style='display:flex; justify-content:space-between;
                                    margin-bottom:6px;'>
                            <span style='color:{T['muted']}; font-size:0.75rem;'>
                                Top Score
                            </span>
                            <span style='color:{T['accent']};
                                         font-weight:700;'>
                                {top_score:.4f}
                            </span>
                        </div>
                        <div style='display:flex; justify-content:space-between;'>
                            <span style='color:{T['muted']}; font-size:0.75rem;'>
                                Clusters
                            </span>
                            <span style='color:{T['accent']};
                                         font-weight:700;'>
                                {entry['graph']['num_clusters']}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Research gaps
                if entry.get("research_gaps"):
                    st.markdown(f"""
                    <p style='color:{T['muted']}; font-size:0.72rem;
                              text-transform:uppercase; letter-spacing:1px;
                              margin:12px 0 6px 0; font-weight:600;'>
                        Research Gaps
                    </p>
                    """, unsafe_allow_html=True)
                    for gap in entry["research_gaps"][:3]:
                        sc = gap["scores"]["innovation_opportunity_score"]
                        kws = " ".join([
                            accent_tag(kw)
                            for kw in gap["bci_keywords"][:3]
                        ])
                        st.markdown(f"""
                        <div style='background:{T['surface']};
                                    border:1px solid {T['border']};
                                    border-radius:6px; padding:8px 12px;
                                    margin-bottom:5px;'>
                            <span style='color:{T['accent']};
                                         font-weight:700; font-size:0.8rem;'>
                                {sc:.4f}
                            </span>
                            <span style='color:{T['text']}; font-size:0.8rem;
                                         margin-left:8px;'>
                                {gap['title'][:55]}...
                            </span><br>
                            <div style='margin-top:4px;'>{kws}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # AI innovations
                if entry.get("innovations"):
                    st.markdown(f"""
                    <p style='color:{T['muted']}; font-size:0.72rem;
                              text-transform:uppercase; letter-spacing:1px;
                              margin:12px 0 6px 0; font-weight:600;'>
                        AI Proposals
                    </p>
                    """, unsafe_allow_html=True)
                    for inv in entry["innovations"][:2]:
                        title = (
                            inv["proposal"].get(
                                "innovation_title","Untitled"
                            ) or "Untitled"
                        )
                        st.markdown(f"""
                        <div style='background:{T['accent']}08;
                                    border:1px solid {T['accent']}22;
                                    border-radius:6px; padding:8px 12px;
                                    margin-bottom:5px;'>
                            <span style='color:{T['accent']};
                                         font-weight:700; font-size:0.8rem;'>
                                💡 {title[:60]}
                            </span><br>
                            <span style='color:{T['muted']};
                                         font-size:0.75rem;'>
                                Gap score: {inv['gap_score']:.4f}
                                · {inv.get('gap_year','N/A')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                # Delete button
                st.markdown("<div style='height:8px'></div>",
                            unsafe_allow_html=True)
                if st.button(
                    "🗑️ Delete this entry",
                    key=f"del_{entry['_file'].stem}"
                ):
                    delete_history_entry(entry["_file"])
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESEARCH ROADMAP GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            🗺️ AI Research Roadmap Generator
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Select any research gap and get a complete 6-step execution plan —
            hypothesis, methodology, dataset, expected results, publication
            strategy, and patent claim. No other tool does this.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not gaps:
        st.info("No gaps loaded. Run a Live Query first.")
    else:
        # Gap selector
        gap_titles = [
            f"Gap #{i+1} [{g['scores']['innovation_opportunity_score']:.4f}] "
            f"— {g['title'][:55]}..."
            for i, g in enumerate(gaps[:10])
        ]

        selected_gap_idx = st.selectbox(
            "Select a research gap to generate a roadmap for:",
            range(len(gap_titles)),
            format_func=lambda i: gap_titles[i],
            key="roadmap_gap_selector"
        )

        selected_gap = gaps[selected_gap_idx]

        # Preview selected gap
        st.markdown(f"""
        <div style='background:{T['surface']};
                    border:1px solid {T['border']};
                    border-left:3px solid {T['accent']};
                    border-radius:8px; padding:14px 18px;
                    margin:12px 0;'>
            <p style='color:{T['accent']}; font-weight:700;
                      font-size:0.72rem; text-transform:uppercase;
                      letter-spacing:1px; margin:0 0 6px 0;'>
                Selected Gap
            </p>
            <p style='color:{T['text']}; font-weight:600;
                      font-size:0.9rem; margin:0 0 4px 0;'>
                {selected_gap['title']}
            </p>
            <span style='color:{T['muted']}; font-size:0.78rem;'>
                {selected_gap.get('year','N/A')}
                · {selected_gap.get('citations',0)} citations
                · Score: {selected_gap['scores']['innovation_opportunity_score']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        roadmap_btn = st.button(
            "🗺️  Generate Research Roadmap", type="primary"
        )

        if roadmap_btn:
            from roadmap_generator import generate_roadmap

            with st.spinner(
                "Generating your complete research roadmap..."
            ):
                roadmap = generate_roadmap(
                    selected_gap,
                    domain="Brain-Computer Interfaces"
                )

            st.markdown("---")
            st.markdown(f"""
            <div style='background:{T['success']}12;
                        border:1px solid {T['success']}33;
                        border-radius:8px; padding:10px 18px;
                        margin-bottom:20px;'>
                <span style='color:{T['success']}; font-weight:700;'>
                    ✓ Roadmap generated
                </span>
                <span style='color:{T['muted']}; font-size:0.85rem;
                             margin-left:8px;'>
                    6-step execution plan for:
                    {selected_gap['title'][:50]}...
                </span>
            </div>
            """, unsafe_allow_html=True)

            # 6 steps
            steps = [
                ("hypothesis",  "1",  "Hypothesis",
                 "The specific research question to investigate.",
                 "🔬"),
                ("methodology", "2",  "Methodology",
                 "Exact research design and statistical approach.",
                 "⚗️"),
                ("dataset",     "3",  "Dataset",
                 "Specific data sources and collection strategy.",
                 "🗄️"),
                ("expected",    "4",  "Expected Results",
                 "What success looks like — specific metrics.",
                 "📈"),
                ("publication", "5",  "Publication Strategy",
                 "Target journals/conferences and timeline.",
                 "📄"),
                ("patent",      "6",  "Patent Strategy",
                 "The specific independent claim to file.",
                 "⚖️"),
            ]

            for key, num, label, desc, icon in steps:
                content = roadmap.get(key, "")
                if not content:
                    content = roadmap.get("raw", "")[:200]

                # Color varies by step
                step_colors = [
                    T['accent'], T['accent2'],
                    "#10b981", "#f59e0b",
                    "#06b6d4", "#f43f5e"
                ]
                sc = step_colors[int(num)-1]

                st.markdown(f"""
                <div style='background:{T['surface']};
                            border:1px solid {T['border']};
                            border-radius:10px; padding:18px 22px;
                            margin-bottom:12px;
                            border-left:4px solid {sc};'>
                    <div style='display:flex; align-items:center;
                                gap:10px; margin-bottom:10px;'>
                        <div style='width:32px; height:32px;
                                    background:{sc}22;
                                    border:1px solid {sc}44;
                                    border-radius:8px;
                                    display:flex; align-items:center;
                                    justify-content:center;
                                    font-size:1rem; flex-shrink:0;'>
                            {icon}
                        </div>
                        <div>
                            <span style='color:{sc}; font-weight:800;
                                         font-size:0.72rem;
                                         text-transform:uppercase;
                                         letter-spacing:1px;'>
                                Step {num}
                            </span>
                            <span style='color:{T['text']};
                                         font-weight:700;
                                         font-size:0.95rem;
                                         margin-left:8px;'>
                                {label}
                            </span>
                        </div>
                    </div>
                    <p style='color:{T['muted']}; font-size:0.75rem;
                              margin:0 0 8px 0; font-style:italic;'>
                        {desc}
                    </p>
                    <p style='color:{T['text']}; font-size:0.87rem;
                              line-height:1.7; margin:0;'>
                        {content}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Risk assessment
            risk = roadmap.get("risk", "")
            if risk:
                st.markdown(f"""
                <div style='background:{T['surface2']};
                            border:1px solid {T['border']};
                            border-radius:10px; padding:18px 22px;
                            margin-bottom:12px;
                            border-left:4px solid #f59e0b;'>
                    <div style='display:flex; align-items:center;
                                gap:10px; margin-bottom:10px;'>
                        <div style='width:32px; height:32px;
                                    background:#f59e0b22;
                                    border:1px solid #f59e0b44;
                                    border-radius:8px;
                                    display:flex; align-items:center;
                                    justify-content:center;
                                    font-size:1rem;'>
                            ⚠️
                        </div>
                        <span style='color:#f59e0b; font-weight:800;
                                     font-size:0.9rem;'>
                            Risk Assessment
                        </span>
                    </div>
                    <p style='color:{T['text']}; font-size:0.87rem;
                              line-height:1.7; margin:0;'>
                        {risk}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Export this roadmap
            st.markdown("---")
            roadmap_text = f"""NIIE — Research Roadmap
Generated for: {selected_gap['title']}
Innovation Score: {selected_gap['scores']['innovation_opportunity_score']:.4f}

STEP 1 - HYPOTHESIS
{roadmap.get('hypothesis','')}

STEP 2 - METHODOLOGY
{roadmap.get('methodology','')}

STEP 3 - DATASET
{roadmap.get('dataset','')}

STEP 4 - EXPECTED RESULTS
{roadmap.get('expected','')}

STEP 5 - PUBLICATION STRATEGY
{roadmap.get('publication','')}

STEP 6 - PATENT STRATEGY
{roadmap.get('patent','')}

RISK ASSESSMENT
{roadmap.get('risk','')}
"""
            st.download_button(
                label="⬇️ Download Roadmap as .txt",
                data=roadmap_text,
                file_name="NIIE_Research_Roadmap.txt",
                mime="text/plain"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DOMAIN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            ⚔️ Domain Comparison
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Run two research domains simultaneously and compare innovation
            opportunities head-to-head.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_d1, col_vs, col_d2 = st.columns([5, 1, 5])

    with col_d1:
        domain_a = st.text_input(
            "Domain A",
            placeholder="e.g. brain-computer interfaces",
            key="domain_a"
        )
    with col_vs:
        st.markdown(f"""
        <div style='text-align:center; padding-top:28px;
                    color:{T['accent']}; font-weight:800; font-size:1.2rem;'>
            VS
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        domain_b = st.text_input(
            "Domain B",
            placeholder="e.g. neuromorphic chips",
            key="domain_b"
        )

    compare_btn = st.button("⚔️  Run Comparison", type="primary")

    if compare_btn and domain_a.strip() and domain_b.strip():
        from pipeline import run_pipeline

        st.markdown("---")
        col_a, col_b = st.columns(2)

        results = {}
        errors  = {}

        for domain, col in [(domain_a.strip(), col_a),
                             (domain_b.strip(), col_b)]:
            with col:
                st.markdown(f"""
                <div style='background:{T['surface']}; border:1px solid {T['border']};
                            border-top:3px solid {T['accent']};
                            border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
                    <span style='color:{T['accent']}; font-weight:700;'>
                        {domain.upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                log   = st.empty()
                steps = []
                cnt   = [0]

                def upd_cmp(msg, steps=steps, cnt=cnt, log=log):
                    steps.append(msg)
                    cnt[0] += 1
                    log.markdown(
                        f"<div style='background:{T['surface2']}; "
                        f"border:1px solid {T['border']}; border-radius:6px; "
                        f"padding:10px 14px; font-size:0.78rem; color:{T['muted']};'>"
                        + "<br>".join(steps[-5:]) + "</div>",
                        unsafe_allow_html=True
                    )

                with st.spinner(f"Analysing {domain}..."):
                    result, err = run_pipeline(domain, upd_cmp)

                if err:
                    errors[domain] = err
                    st.error(err)
                else:
                    results[domain] = result
                    s = result["scores"]

                    # Mini KPIs
                    m1, m2 = st.columns(2)
                    m1.metric("Papers",  result["papers"])
                    m2.metric("Gaps",    s["total_gaps"])
                    m1.metric("Edges",   result["graph"]["num_edges"])
                    m2.metric("Clusters",result["graph"]["num_clusters"])

                    # Top 3 opportunities
                    st.markdown(f"""
                    <p style='color:{T['muted']}; font-size:0.72rem;
                              text-transform:uppercase; letter-spacing:1px;
                              margin:12px 0 6px 0; font-weight:600;'>
                        Top Opportunities
                    </p>
                    """, unsafe_allow_html=True)

                    for i, p in enumerate(s["top_opportunities"][:3], 1):
                        sc = p["scores"]["innovation_opportunity_score"]
                        st.markdown(f"""
                        <div style='background:{T['surface2']};
                                    border-left:2px solid {T['accent']};
                                    border-radius:0 6px 6px 0;
                                    padding:8px 12px; margin-bottom:5px;'>
                            <span style='color:{T['accent']}; font-weight:700;
                                         font-size:0.82rem;'>
                                #{i} {sc:.4f}
                            </span><br>
                            <span style='color:{T['text']}; font-size:0.78rem;'>
                                {p['title'][:50]}...
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

        # ── COMPARISON CHARTS ─────────────────────────────────────────────
        if len(results) == 2:
            st.markdown("---")
            st.markdown(f"""
            <h4 style='color:{T['text']}; font-weight:700; margin-bottom:12px;'>
                Head-to-Head Analysis
            </h4>
            """, unsafe_allow_html=True)

            domains   = list(results.keys())
            signals   = ["research_momentum","citation_density",
                         "keyword_novelty","isolation_score",
                         "patent_gap_score"]
            sig_labels = ["Momentum","Cit. Density",
                          "Novelty","Isolation","Patent Gap"]

            # Average scores per signal
            def avg_signal(result, signal):
                scores = result["scores"]["all_scores"]
                vals   = [p["scores"].get(signal, 0) for p in scores]
                return sum(vals) / len(vals) if vals else 0

            avg_a = [avg_signal(results[domains[0]], s) for s in signals]
            avg_b = [avg_signal(results[domains[1]], s) for s in signals]

            # Overall innovation score average
            def avg_inn(result):
                scores = result["scores"]["all_scores"]
                vals   = [p["scores"]["innovation_opportunity_score"]
                          for p in scores]
                return round(sum(vals)/len(vals), 4) if vals else 0

            inn_a = avg_inn(results[domains[0]])
            inn_b = avg_inn(results[domains[1]])

            # Winner banner
            winner = domains[0] if inn_a >= inn_b else domains[1]
            winner_score = max(inn_a, inn_b)
            st.markdown(f"""
            <div style='background:{T['success']}12;
                        border:1px solid {T['success']}44;
                        border-radius:10px; padding:14px 20px;
                        margin-bottom:16px; text-align:center;'>
                <span style='color:{T['success']}; font-size:1.1rem;
                             font-weight:800;'>
                    🏆 Higher Opportunity Domain:
                </span>
                <span style='color:{T['text']}; font-size:1.1rem;
                             font-weight:700; margin-left:8px;'>
                    {winner.upper()}
                </span>
                <span style='color:{T['muted']}; font-size:0.85rem;
                             margin-left:8px;'>
                    (avg score: {winner_score:.4f})
                </span>
            </div>
            """, unsafe_allow_html=True)

            ch1, ch2 = st.columns(2)

            # Signal comparison bar chart
            with ch1:
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name=domains[0][:20],
                    x=sig_labels, y=avg_a,
                    marker_color=T['accent'],
                    text=[f"{v:.3f}" for v in avg_a],
                    textposition="outside",
                    textfont=dict(size=9, color=T['muted'])
                ))
                fig_cmp.add_trace(go.Bar(
                    name=domains[1][:20],
                    x=sig_labels, y=avg_b,
                    marker_color=T['accent2'],
                    text=[f"{v:.3f}" for v in avg_b],
                    textposition="outside",
                    textfont=dict(size=9, color=T['muted'])
                ))
                fig_cmp.update_layout(
                    title="Signal Comparison",
                    barmode="group",
                    paper_bgcolor=T['surface'],
                    plot_bgcolor=T['surface2'],
                    font_color=T['muted'],
                    height=320,
                    margin=dict(l=10,r=10,t=40,b=10),
                    legend=dict(font_size=9,
                                bgcolor=T['surface'],
                                bordercolor=T['border']),
                    yaxis=dict(range=[0,1],
                               gridcolor=T['border']),
                    xaxis=dict(gridcolor=T['border']),
                    title_font_color=T['text']
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            # Radar comparison
            with ch2:
                fig_rad = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(
                    r=avg_a + [avg_a[0]],
                    theta=sig_labels + [sig_labels[0]],
                    fill="toself",
                    name=domains[0][:20],
                    line_color=T['accent'],
                    fillcolor=hex_to_rgba(T['accent'], 0.13)
                ))
                fig_rad.add_trace(go.Scatterpolar(
                    r=avg_b + [avg_b[0]],
                    theta=sig_labels + [sig_labels[0]],
                    fill="toself",
                    name=domains[1][:20],
                    line_color=T['accent2'],
                    fillcolor=hex_to_rgba(T['accent2'], 0.13)
                ))
                fig_rad.update_layout(
                    title="Radar Comparison",
                    polar=dict(
                        bgcolor=T['surface2'],
                        radialaxis=dict(
                            visible=True, range=[0,1],
                            gridcolor=T['border'],
                            color=T['muted']
                        ),
                        angularaxis=dict(
                            gridcolor=T['border'],
                            color=T['muted']
                        )
                    ),
                    paper_bgcolor=T['surface'],
                    font_color=T['muted'],
                    height=320,
                    margin=dict(l=10,r=10,t=40,b=10),
                    legend=dict(font_size=9,
                                bgcolor=T['surface'],
                                bordercolor=T['border']),
                    title_font_color=T['text']
                )
                st.plotly_chart(fig_rad, use_container_width=True)

            # Score summary table
            st.markdown(f"""
            <table style='width:100%; border-collapse:collapse;
                          font-size:0.85rem; margin-top:8px;'>
                <thead>
                    <tr style='background:{T['surface2']};'>
                        <th style='padding:10px 14px; text-align:left;
                                   color:{T['muted']}; font-weight:600;
                                   border-bottom:1px solid {T['border']};'>
                            Signal
                        </th>
                        <th style='padding:10px 14px; text-align:center;
                                   color:{T['accent']}; font-weight:700;
                                   border-bottom:1px solid {T['border']};'>
                            {domains[0][:25]}
                        </th>
                        <th style='padding:10px 14px; text-align:center;
                                   color:{T['accent2']}; font-weight:700;
                                   border-bottom:1px solid {T['border']};'>
                            {domains[1][:25]}
                        </th>
                        <th style='padding:10px 14px; text-align:center;
                                   color:{T['muted']}; font-weight:600;
                                   border-bottom:1px solid {T['border']};'>
                            Winner
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f"<tr style='background:{T['surface'] if i%2==0 else T['surface2']};'>"
                        f"<td style='padding:9px 14px; color:{T['text']}; "
                        f"border-bottom:1px solid {T['border']};'>{sig_labels[i]}</td>"
                        f"<td style='padding:9px 14px; text-align:center; "
                        f"color:{T['accent']}; font-weight:600; "
                        f"border-bottom:1px solid {T['border']};'>{avg_a[i]:.3f}</td>"
                        f"<td style='padding:9px 14px; text-align:center; "
                        f"color:{T['accent2']}; font-weight:600; "
                        f"border-bottom:1px solid {T['border']};'>{avg_b[i]:.3f}</td>"
                        f"<td style='padding:9px 14px; text-align:center; "
                        f"border-bottom:1px solid {T['border']};'>"
                        f"<span style='color:{T['success']}; font-weight:700;'>"
                        f"{'→ ' + domains[0][:15] if avg_a[i] >= avg_b[i] else '→ ' + domains[1][:15]}"
                        f"</span></td></tr>"
                        for i in range(len(sig_labels))
                    ])}
                    <tr style='background:{T['surface2']};'>
                        <td style='padding:10px 14px; color:{T['text']};
                                   font-weight:700;'>Overall Innovation</td>
                        <td style='padding:10px 14px; text-align:center;
                                   color:{T['accent']}; font-weight:800;
                                   font-size:1rem;'>{inn_a:.4f}</td>
                        <td style='padding:10px 14px; text-align:center;
                                   color:{T['accent2']}; font-weight:800;
                                   font-size:1rem;'>{inn_b:.4f}</td>
                        <td style='padding:10px 14px; text-align:center;'>
                            <span style='color:{T['success']}; font-weight:800;'>
                                🏆 {winner[:15]}
                            </span>
                        </td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

    elif compare_btn:
        st.warning("Please enter both domains before running comparison.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    from sklearn.decomposition import PCA

    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            Knowledge Graph
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Each node is a paper · Edges connect semantically similar papers ·
            Node size = centrality · Colour = research cluster
        </p>
    </div>
    """, unsafe_allow_html=True)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    COLORS = [
        T['accent'], T['accent2'],
        "#f43f5e","#f59e0b","#10b981",
        "#06b6d4","#8b5cf6","#ec4899",
        "#14b8a6","#f97316","#6366f1","#84cc16"
    ]

    nx_arr, ny_arr, nt_arr, nc_arr, ns_arr = [], [], [], [], []
    for node in nodes:
        idx = node["id"]
        nx_arr.append(float(coords[idx,0]))
        ny_arr.append(float(coords[idx,1]))
        cl = node.get("cluster",0)
        nc_arr.append(COLORS[cl % len(COLORS)])
        ns_arr.append(9 + node.get("combined_score",0) * 70)
        nt_arr.append(
            f"<b>{node['title'][:52]}...</b><br>"
            f"Year: {node.get('year','N/A')} · Citations: {node.get('citations',0)}<br>"
            f"Cluster {cl} · {', '.join(node.get('keywords',[])[:3])}"
        )

    ex, ey = [], []
    npos = {n["id"]:(float(coords[n["id"],0]),float(coords[n["id"],1])) for n in nodes}
    for e in edges[:380]:
        x0,y0 = npos[e["source"]]; x1,y1 = npos[e["target"]]
        ex += [x0,x1,None]; ey += [y0,y1,None]

    fig_g = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines",
                   line=dict(width=0.6, color=T['border']),
                   hoverinfo="none"),
        go.Scatter(x=nx_arr, y=ny_arr, mode="markers",
                   hoverinfo="text", text=nt_arr,
                   marker=dict(size=ns_arr, color=nc_arr,
                               line=dict(width=0.5,
                                         color=T['surface']),
                               opacity=0.88))
    ])
    fig_g.update_layout(
        showlegend=False, hovermode="closest",
        height=560,
        paper_bgcolor=T['surface'],
        plot_bgcolor=T['surface'],
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showline=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showline=False),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # Cluster legend
    sizes = graph_data["cluster_sizes"]
    cols  = st.columns(min(len(sizes), 6))
    for i, sz in enumerate(sizes[:6]):
        c = COLORS[i % len(COLORS)]
        cols[i].markdown(
            f"<div style='background:{c}12; border:1px solid {c}33; "
            f"border-radius:6px; padding:7px 10px; text-align:center;'>"
            f"<span style='color:{c}; font-weight:700; font-size:0.82rem;'>"
            f"Cluster {i+1}</span><br>"
            f"<span style='color:{T['muted']}; font-size:0.75rem;'>{sz} papers</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCORES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            Innovation Opportunity Scores
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            5-signal model: Research Momentum · Citation Density ·
            Keyword Novelty · Isolation · Patent Gap
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Title"          : p["title"][:55]+"...",
        "Year"           : p["year"],
        "Citations"      : p["citations"],
        "Momentum"       : p["scores"]["research_momentum"],
        "Cit. Density"   : p["scores"]["citation_density"],
        "Novelty"        : p["scores"]["keyword_novelty"],
        "Isolation"      : p["scores"]["isolation_score"],
        "Patent Gap"     : p["scores"].get("patent_gap_score",0),
        "Score"          : p["scores"]["innovation_opportunity_score"],
    } for p in all_scores])

    ca, cb = st.columns(2)
    with ca:
        fig_h = px.histogram(df, x="Score", nbins=20, title="Score Distribution",
                             color_discrete_sequence=[T['accent']])
        plot_cfg(fig_h, 280)
        fig_h.update_traces(marker_line_width=0)
        fig_h.update_layout(title_font_color=T['text'])
        st.plotly_chart(fig_h, use_container_width=True)
    with cb:
        fig_p = go.Figure(go.Pie(
            labels=["Momentum","Cit. Density","Novelty","Isolation","Patent Gap"],
            values=[0.25,0.20,0.20,0.20,0.15],
            hole=0.6,
            marker_colors=T['chart_seq'],
            textinfo="label+percent",
            textfont_size=10,
        ))
        fig_p.update_layout(
            title="Signal Weights",
            title_font_color=T['text'],
            paper_bgcolor=T['surface'],
            height=280,
            margin=dict(l=10,r=10,t=36,b=10),
            font_color=T['muted'],
            showlegend=False
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.dataframe(
        df.style.background_gradient(subset=["Score"], cmap="Blues")
          .format({"Momentum":"{:.3f}","Cit. Density":"{:.3f}",
                   "Novelty":"{:.3f}","Isolation":"{:.3f}",
                   "Patent Gap":"{:.3f}","Score":"{:.4f}"}),
        use_container_width=True, height=370
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RESEARCH GAPS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            {len(gaps)} Research Gaps Detected
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            High novelty · Low citations · Graph-isolated · Low patent coverage
        </p>
    </div>
    """, unsafe_allow_html=True)

    for i, gap in enumerate(gaps[:10], 1):
        sc  = gap["scores"]["innovation_opportunity_score"]
        bc  = T['accent'] if i <= 3 else T['border']
        with st.expander(
            f"Gap #{i}  ·  {sc:.4f}  ·  {gap['title'][:58]}...",
            expanded=(i==1)
        ):
            left, right = st.columns([3,2])
            with left:
                abstract = gap.get("abstract","") or "No abstract."
                st.markdown(f"""
                <p style='color:{T['text']}; font-weight:600; margin:0 0 6px 0;'>
                    {gap['title']}
                </p>
                <div style='display:flex; gap:20px; margin-bottom:10px;'>
                    <div>
                        <span style='color:{T['muted']}; font-size:0.72rem;
                                     text-transform:uppercase; letter-spacing:1px;'>
                            Year
                        </span><br>
                        <span style='color:{T['accent']}; font-weight:700;'>
                            {gap['year']}
                        </span>
                    </div>
                    <div>
                        <span style='color:{T['muted']}; font-size:0.72rem;
                                     text-transform:uppercase; letter-spacing:1px;'>
                            Citations
                        </span><br>
                        <span style='color:{T['accent']}; font-weight:700;'>
                            {gap['citations']}
                        </span>
                    </div>
                    <div>
                        <span style='color:{T['muted']}; font-size:0.72rem;
                                     text-transform:uppercase; letter-spacing:1px;'>
                            Score
                        </span><br>
                        <span style='color:{T['accent']}; font-weight:700;'>
                            {sc:.4f}
                        </span>
                    </div>
                </div>
                <p style='color:{T['muted']}; font-size:0.84rem;
                           line-height:1.6; margin:0 0 10px 0;'>
                    {abstract[:320]}...
                </p>
                <div>
                    {''.join([accent_tag(kw) for kw in gap['bci_keywords']])}
                </div>
                """, unsafe_allow_html=True)
            with right:
                s = gap["scores"]
                snames = ["Momentum","Cit. Density","Novelty","Isolation","Patent Gap"]
                svals  = [s["research_momentum"],s["citation_density"],
                          s["keyword_novelty"],s["isolation_score"],
                          s.get("patent_gap_score",0)]
                fig_b = go.Figure(go.Bar(
                    x=svals, y=snames, orientation="h",
                    marker_color=T['chart_seq'],
                    text=[f"{v:.3f}" for v in svals],
                    textposition="outside",
                    textfont=dict(color=T['muted'], size=11)
                ))
                fig_b.update_layout(
                    paper_bgcolor=T['surface2'],
                    plot_bgcolor=T['surface2'],
                    height=200,
                    margin=dict(l=0,r=40,t=8,b=8),
                    xaxis=dict(range=[0,1.1],showgrid=False,
                               zeroline=False,color=T['muted']),
                    yaxis=dict(color=T['muted'],tickfont_size=11),
                    font_color=T['muted']
                )
                st.plotly_chart(fig_b, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — TOP OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            Top 10 Innovation Opportunities
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Highest combined score across all 5 signals
        </p>
    </div>
    """, unsafe_allow_html=True)

    for rank, paper in enumerate(all_scores[:10], 1):
        sc     = paper["scores"]["innovation_opportunity_score"]
        medal  = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"#{rank}"
        border = T['accent'] if rank<=3 else T['border']
        bg     = T['surface2'] if rank<=3 else T['surface']

        st.markdown(f"""
        <div style='background:{bg}; border:1px solid {border};
                    border-left:3px solid {border};
                    border-radius:8px; padding:13px 18px; margin-bottom:7px;
                    transition: all 0.2s ease;'>
            <div style='display:flex; justify-content:space-between;
                        align-items:baseline;'>
                <div>
                    <span style='font-size:1rem;'>{medal}</span>
                    <span style='color:{T['accent']}; font-weight:700;
                                 margin-left:8px;'>{sc:.4f}</span>
                    <span style='color:{T['text']}; margin-left:10px;
                                 font-size:0.9rem;'>{paper['title']}</span>
                </div>
            </div>
            <div style='margin-top:5px;'>
                <span style='color:{T['muted']}; font-size:0.76rem;'>
                    {paper['year']} &nbsp;·&nbsp; {paper['citations']} citations
                    &nbsp;·&nbsp; {', '.join(paper['bci_keywords'][:4])}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style='color:{T['muted']}; font-size:0.8rem; text-transform:uppercase;
              letter-spacing:1px; margin:20px 0 10px 0; font-weight:600;'>
        Signal Breakdown — Top 3
    </p>
    """, unsafe_allow_html=True)

    cats   = ["Momentum","Cit. Density","Novelty","Isolation","Patent Gap"]
    fig_r  = go.Figure()
    rcols  = T['chart_seq'][:3]

    for i, paper in enumerate(all_scores[:3]):
        s = paper["scores"]
        fig_r.add_trace(go.Scatterpolar(
            r=[s["research_momentum"],s["citation_density"],
               s["keyword_novelty"],s["isolation_score"],
               s.get("patent_gap_score",0)],
            theta=cats, fill="toself",
            name=paper["title"][:35]+"...",
            line_color=rcols[i],
            fillcolor=hex_to_rgba(rcols[i], 0.14)
        ))

    fig_r.update_layout(
        polar=dict(
            bgcolor=T['surface2'],
            radialaxis=dict(visible=True,range=[0,1],
                            gridcolor=T['border'],color=T['muted']),
            angularaxis=dict(gridcolor=T['border'],color=T['muted'])
        ),
        paper_bgcolor=T['surface'],
        font_color=T['muted'],
        height=420,
        legend=dict(font_size=10, bgcolor=T['surface'],
                    bordercolor=T['border'])
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI PROPOSALS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"""
    <div style='margin-bottom:16px;'>
        <h3 style='margin:0 0 3px 0; color:{T['text']}; font-weight:700;'>
            AI-Generated Innovation Proposals
        </h3>
        <p style='margin:0; color:{T['muted']}; font-size:0.83rem;'>
            Generated by Llama 3.3-70B · Analyzing underexplored BCI research gaps
        </p>
    </div>
    """, unsafe_allow_html=True)

    innovations_file = "data/processed/bci_innovations.json"
    if not os.path.exists(innovations_file):
        st.warning("No proposals found. Run `python src/llm_generator.py` first.")
    else:
        with open(innovations_file,"r",encoding="utf-8") as f:
            innovations = json.load(f)

        st.markdown(f"""
        <div style='background:{T['success']}12; border:1px solid {T['success']}33;
                    border-radius:8px; padding:10px 16px; margin-bottom:16px;
                    display:inline-block;'>
            <span style='color:{T['success']}; font-weight:700; font-size:0.88rem;'>
                ✓ {len(innovations)} proposals generated
            </span>
            <span style='color:{T['muted']}; font-size:0.82rem; margin-left:6px;'>
                by Llama 3.3-70B via Groq
            </span>
        </div>
        """, unsafe_allow_html=True)

        for i, inv in enumerate(innovations, 1):
            p     = inv["proposal"]
            sc    = inv["gap_score"]
            title = clean_text(p.get("innovation_title","Untitled") or "Untitled").strip()
            if len(title) > 70:
                title_summary = title[:67].rstrip() + "..."
            else:
                title_summary = title
            summary_prefix = f"★ " if i <= 3 else ""
            with st.expander(
                f"{summary_prefix}#{i} · {title_summary}",
                expanded=(i==1)
            ):
                m1,m2,m3 = st.columns(3)
                m1.metric("Gap Score", f"{sc:.4f}")
                m2.metric("Year",      inv.get("gap_year","N/A"))
                m3.metric("Citations", inv.get("gap_citations",0))

                st.markdown(f"<div style='height:8px'></div>",
                            unsafe_allow_html=True)
                left, right = st.columns([3,2])

                with left:
                    for key, label in [
                        ("problem_it_solves",    "Problem It Solves"),
                        ("proposed_solution",    "Proposed Solution"),
                        ("why_novel",            "Why It Is Novel"),
                        ("research_contribution","Research Contribution"),
                    ]:
                        val = clean_text(p.get(key,""))
                        if val:
                            st.markdown(f"""
                            <div style='margin-bottom:16px;'>
                                <p style='color:{T['accent']}; font-weight:700;
                                          font-size:0.75rem; text-transform:uppercase;
                                          letter-spacing:1px; margin:0 0 5px 0;'>
                                    {label}
                                </p>
                                <p style='color:{T['text']}; font-size:0.87rem;
                                          line-height:1.65; margin:0;'>{val}</p>
                            </div>
                            """, unsafe_allow_html=True)

                with right:
                    for key, label in [
                        ("technology_stack","Technology Stack"),
                        ("target_users",    "Target Users"),
                    ]:
                        val = clean_text(p.get(key,""))
                        if val:
                            st.markdown(f"""
                            <div style='background:{T['surface2']}; border:1px solid {T['border']};
                                        border-radius:8px; padding:12px 14px; margin-bottom:10px;'>
                                <p style='color:{T['accent']}; font-weight:700;
                                          font-size:0.72rem; text-transform:uppercase;
                                          letter-spacing:1px; margin:0 0 6px 0;'>{label}</p>
                                <p style='color:{T['muted']}; font-size:0.83rem;
                                          line-height:1.55; margin:0;'>{val}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    patent = p.get("patent_opportunity","")
                    patent = clean_text(patent)
                    if patent:
                        st.markdown(f"""
                        <div style='background:{T['accent']}08; border:1px solid {T['accent']}22;
                                    border-radius:8px; padding:12px 14px; margin-bottom:10px;'>
                            <p style='color:{T['accent']}; font-weight:700;
                                      font-size:0.72rem; text-transform:uppercase;
                                      letter-spacing:1px; margin:0 0 6px 0;'>
                                Patent Opportunity
                            </p>
                            <p style='color:{T['text']}; font-size:0.83rem;
                                      line-height:1.55; margin:0; font-style:italic;'>
                                {patent}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    impact = p.get("impact_score","")
                    if impact:
                        num = re.search(r'\d+', impact)
                        if num:
                            n = int(num.group())
                            st.markdown(f"""
                            <div style='background:{T['surface2']}; border:1px solid {T['border']};
                                        border-radius:8px; padding:12px 14px;'>
                                <p style='color:{T['muted']}; font-weight:700;
                                          font-size:0.72rem; text-transform:uppercase;
                                          letter-spacing:1px; margin:0 0 8px 0;'>
                                    Impact Score
                                </p>
                                <p style='color:{T['accent']}; font-weight:800;
                                          font-size:1.4rem; margin:0 0 6px 0;'>
                                    {n}<span style='font-size:0.8rem; color:{T['muted']};'>/10</span>
                                </p>
                            """, unsafe_allow_html=True)
                            st.progress(n/10)
                            st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style='margin-top:14px; padding-top:12px;
                            border-top:1px solid {T['border']};'>
                    <p style='color:{T['muted']}; font-size:0.72rem;
                              text-transform:uppercase; letter-spacing:1px;
                              margin:0 0 4px 0;'>Source Paper</p>
                    <p style='color:{T['text']}; font-size:0.83rem;
                              font-style:italic; margin:0 0 8px 0;'>
                        {clean_text(inv.get('gap_title',''))}
                    </p>
                    {''.join([accent_tag(clean_text(kw)) for kw in inv.get('gap_keywords',[])])}
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding:20px 0 6px 0;
            border-top:1px solid {T['border']}; margin-top:20px;'>
    <span style='color:{T['muted']}; font-size:0.75rem;'>
        NIIE · National Innovation Intelligence Engine ·
        Python · Streamlit · NetworkX · Sentence-Transformers · Plotly · Groq
    </span>
</div>
""", unsafe_allow_html=True)