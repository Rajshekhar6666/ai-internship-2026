import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import re
import sys
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots
from collections import Counter

try:
    from streamlit_option_menu import option_menu
    _has_option_menu = True
except ImportError:
    option_menu = None
    _has_option_menu = False

sys.path.append("src")

st.set_page_config(
    page_title="NIIE — National Innovation Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── HISTORY HELPERS ───────────────────────────────────────────────────────────
HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def save_to_history(query, result):
    key     = hashlib.md5(query.lower().strip().encode()).hexdigest()[:10]
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    fname   = HISTORY_DIR / f"{key}.pkl"
    payload = {
        "query"            : query,
        "timestamp"        : ts,
        "papers"           : result["papers"],
        "graph"            : {
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
    with open(fname, "wb") as f:
        pickle.dump(payload, f)
    return ts

def load_history():
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

def delete_history_entry(fp):
    try:
        Path(fp).unlink()
    except Exception:
        pass

# ── THEMES ────────────────────────────────────────────────────────────────────
THEMES = {
    "Midnight Pro": {
        "bg":"#0a0a0f","surface":"#13131a","surface2":"#1a1a25",
        "border":"#2a2a3a","accent":"#6366f1","accent2":"#8b5cf6",
        "text":"#e2e8f0","muted":"#64748b","success":"#10b981",
        "chart_seq":["#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#ddd6fe"],
        "gradient":"linear-gradient(135deg,#6366f1,#8b5cf6)",
        "nav_bg":"#0d0d14","nav_selected":"#6366f1",
    },
    "Arctic Light": {
        "bg":"#f8fafc","surface":"#ffffff","surface2":"#f1f5f9",
        "border":"#e2e8f0","accent":"#0ea5e9","accent2":"#38bdf8",
        "text":"#0f172a","muted":"#94a3b8","success":"#059669",
        "chart_seq":["#0ea5e9","#38bdf8","#7dd3fc","#bae6fd","#e0f2fe"],
        "gradient":"linear-gradient(135deg,#0ea5e9,#38bdf8)",
        "nav_bg":"#f1f5f9","nav_selected":"#0ea5e9",
    },
    "Forest Deep": {
        "bg":"#0d1117","surface":"#161b22","surface2":"#21262d",
        "border":"#30363d","accent":"#2ea043","accent2":"#3fb950",
        "text":"#c9d1d9","muted":"#6e7681","success":"#2ea043",
        "chart_seq":["#2ea043","#3fb950","#56d364","#7ee787","#b5efb5"],
        "gradient":"linear-gradient(135deg,#2ea043,#3fb950)",
        "nav_bg":"#010409","nav_selected":"#2ea043",
    },
    "Amber Dusk": {
        "bg":"#ffffef","surface":"#fffff8","surface2":"#fffde7",
        "border":"#f0e68c","accent":"#d4a017","accent2":"#f59e0b",
        "text":"#1a1400","muted":"#6b5a00","success":"#4caf50",
        "chart_seq":["#d4a017","#f59e0b","#fcd34d","#fde68a","#fef9c3"],
        "gradient":"linear-gradient(135deg,#d4a017,#f59e0b)",
        "nav_bg":"#fffde7","nav_selected":"#d4a017",
    },
    "Rose Gold": {
        "bg":"#fff5f7","surface":"#ffffff","surface2":"#fff0f3",
        "border":"#fecdd3","accent":"#e11d48","accent2":"#fb7185",
        "text":"#1f0a0f","muted":"#9f1239","success":"#059669",
        "chart_seq":["#e11d48","#fb7185","#fda4af","#fecdd3","#fff1f2"],
        "gradient":"linear-gradient(135deg,#e11d48,#fb7185)",
        "nav_bg":"#fff0f3","nav_selected":"#e11d48",
    },
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px 0;'>
        <div style='font-size:2rem;'>🧠</div>
        <div style='font-weight:800; font-size:1.1rem; letter-spacing:-0.3px;'>
            NIIE
        </div>
        <div style='font-size:0.72rem; opacity:0.5; margin-top:2px;'>
            National Innovation Intelligence Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    menu_items = [
        "Live Query",
        "Literature Search",
        "Timing Advisor",
        "Research Roadmap",
        "Paper Maker",
        "Domain Comparison",
        "Query History",
        "Knowledge Graph",
        "Innovation Scores",
        "Research Gaps",
        "Opportunities",
        "AI Proposals",
    ]
    selected = None
    if _has_option_menu and option_menu is not None:
        selected = option_menu(
        menu_title=None,
        options=[
            "Live Query",
            "Literature Search",
            "Timing Advisor",
            "Research Roadmap",
            "Paper Maker",
            "Domain Comparison",
            "Query History",
            "Knowledge Graph",
            "Innovation Scores",
            "Research Gaps",
            "Opportunities",
            "AI Proposals",
        ],
        icons=[
            "search",
            "book",
            "clock-history",
            "map",
            "file-earmark-text",
            "intersect",
            "archive",
            "diagram-3",
            "bar-chart",
            "lightbulb",
            "trophy",
            "robot",
        ],
        default_index=0,
        styles={
            "container": {
                "padding"         : "0",
                "background-color": "transparent",
            },
            "icon": {
                "font-size": "0.85rem",
            },
            "nav-link": {
                "font-size"    : "0.82rem",
                "font-weight"  : "500",
                "padding"      : "8px 12px",
                "border-radius": "6px",
                "margin"       : "1px 0",
            },
            "nav-link-selected": {
                "font-weight": "700",
            },
        }
    )
    else:
        selected = st.sidebar.radio(
            "Navigation",
            menu_items,
            index=0,
            label_visibility="collapsed"
        )

    st.markdown("---")

    # Theme picker
    st.markdown(
        "<p style='font-size:0.72rem; font-weight:700; "
        "text-transform:uppercase; letter-spacing:1px; "
        "opacity:0.5; margin:0 0 6px 0;'>Theme</p>",
        unsafe_allow_html=True
    )
    selected_theme = st.radio(
        "", list(THEMES.keys()), index=0,
        label_visibility="collapsed"
    )
    T = THEMES[selected_theme]

    st.markdown("---")

    # Export
    st.markdown(
        "<p style='font-size:0.72rem; font-weight:700; "
        "text-transform:uppercase; letter-spacing:1px; "
        "opacity:0.5; margin:0 0 6px 0;'>Export</p>",
        unsafe_allow_html=True
    )

    if st.button("Generate PDF Report", type="primary",
                 use_container_width=True):
        with st.spinner("Building report..."):
            try:
                from export_report import generate_report
                with open("data/processed/bci_scores.json",
                          "r", encoding="utf-8") as f:
                    _sc = json.load(f)
                with open("data/processed/bci_graph.json",
                          "r", encoding="utf-8") as f:
                    _gr = json.load(f)
                _inv = []
                if os.path.exists(
                    "data/processed/bci_innovations.json"
                ):
                    with open(
                        "data/processed/bci_innovations.json",
                        "r", encoding="utf-8"
                    ) as f:
                        _inv = json.load(f)
                pdf_path = generate_report(
                    _sc, _gr, _inv,
                    domain="Brain-Computer Interfaces",
                    output_path="reports/NIIE_Innovation_Report.pdf"
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "⬇️ Download PDF", pdf_bytes,
                    "NIIE_Innovation_Report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
                st.success("Ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem; opacity:0.35; "
        "text-align:center;'>NIIE v1.0 · Python · Streamlit<br>"
        "NetworkX · Groq · Plotly</p>",
        unsafe_allow_html=True
    )

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html {{
    overflow-y: auto !important;
    scroll-behavior: smooth !important;
}}

body {{
    overflow-y: auto !important;
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {{
    font-family: 'Inter', sans-serif !important;
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
}}

[data-testid="stMain"] {{
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh !important;
}}

section[data-testid="stMain"] > div:first-child {{
    overflow-y: auto !important;
    height: 100% !important;
    padding-top: 4rem !important;
    padding-bottom: 60px !important;
}}

[data-testid="stAppViewContainer"] {{
    overflow: visible !important;
}}

[data-testid="stHeader"] {{
    background-color: {T['bg']} !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 14000 !important;
    display: flex !important;
    align-items: center !important;
    padding: 8px 20px !important;
    height: 60px !important;
    backdrop-filter: blur(4px) !important;
}}

[data-testid="stSidebar"] {{
    background-color: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
    overflow-y: auto !important;
    padding-top: 0 !important;
}}

[data-testid="stSidebarHeader"] {{
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}}

[data-testid="stSidebarUserContent"] {{
    margin-top: 0 !important;
}}

[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
}}

[data-testid="stSidebar"] .stVerticalBlock > div:first-child {{
    padding-top: 0 !important;
}}

.block-container {{
    padding-top: 0.5rem !important;
    max-width: 1300px !important;
    overflow: visible !important;
}}

[data-testid="stVerticalBlock"] {{
    overflow: visible !important;
    height: auto !important;
}}

[data-testid="stMetric"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s ease !important;
}}
[data-testid="stMetric"]:hover {{
    border-color: {T['accent']} !important;
}}
[data-testid="stMetricLabel"] p {{
    color: {T['muted']} !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}
[data-testid="stMetricValue"] {{
    color: {T['accent']} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}
[data-testid="stExpander"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: {T['accent']} !important;
}}
[data-testid="stExpander"] summary p {{
    color: {T['text']} !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
}}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px {T['accent']}22 !important;
}}
[data-testid="stButton"] button[kind="primary"] {{
    background: {T['gradient']} !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}}
[data-testid="stButton"] button[kind="primary"]:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stSelectbox"] > div {{
    background: {T['surface']} !important;
    border-color: {T['border']} !important;
    color: {T['text']} !important;
}}
hr {{ border-color: {T['border']} !important; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{T['bg']}; }}
::-webkit-scrollbar-thumb {{
    background:{T['border']}; border-radius:3px;
}}
::-webkit-scrollbar-thumb:hover {{ background:{T['accent']}; }}

</style>
""", unsafe_allow_html=True)

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

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.14):
    try:
        h = hex_color.lstrip('#')
        if len(h)==3: h=''.join([c*2 for c in h])
        if len(h)!=6: return f"rgba(99,102,241,{alpha})"
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"
    except:
        return f"rgba(99,102,241,{alpha})"

def accent_tag(text):
    return (f"<span style='background:{T['accent']}18;"
            f"color:{T['accent']};border:1px solid {T['accent']}33;"
            f"padding:2px 8px;border-radius:20px;"
            f"font-size:0.72rem;font-weight:600;"
            f"margin-right:4px;'>{text}</span>")

def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style='margin-bottom:20px;
                padding-bottom:12px;
                border-bottom:1px solid {T['border']};'>
        <h2 style='margin:0 0 3px 0; color:{T['text']};
                   font-weight:800; font-size:1.4rem;'>
            {title}
        </h2>
        {"<p style='margin:0;color:"+T['muted']+";font-size:0.83rem;'>"+subtitle+"</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def plot_cfg(fig, height=320):
    fig.update_layout(
        paper_bgcolor=T['surface'],
        plot_bgcolor=T['surface2'],
        font_color=T['muted'],
        height=height,
        margin=dict(l=16,r=16,t=36,b=16),
    )
    fig.update_xaxes(gridcolor=T['border'],
                     zerolinecolor=T['border'])
    fig.update_yaxes(gridcolor=T['border'],
                     zerolinecolor=T['border'])
    return fig

def kpi_row(data):
    cols = st.columns(len(data))
    for col, (label, value) in zip(cols, data):
        col.metric(label, value)

# ── KPI HEADER (always visible) ───────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px;
            padding-bottom:16px;
            border-bottom:1px solid {T['border']};
            margin-bottom:20px;'>
    <div style='width:36px; height:36px;
                background:{T['gradient']};
                border-radius:8px; display:flex;
                align-items:center; justify-content:center;
                font-size:1.2rem; flex-shrink:0;'>🧠</div>
    <div>
        <span style='font-weight:800; font-size:1.1rem;
                     color:{T['text']};'>NIIE</span>
        <span style='color:{T['muted']}; font-size:0.85rem;
                     margin-left:8px;'>
            National Innovation Intelligence Engine
        </span>
        <span style='color:{T['accent']}; font-size:0.78rem;
                     margin-left:8px; font-weight:600;'>
            · Brain-Computer Interfaces
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

kpi_row([
    ("📄 Papers",   graph_data["num_nodes"]),
    ("🔗 Edges",    graph_data["num_edges"]),
    ("🗂️ Clusters", graph_data["num_clusters"]),
    ("💡 Gaps",     scores_data["total_gaps"]),
])

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE QUERY
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Live Query":
    section_header(
        "🔍 Live Innovation Discovery",
        "Enter any research domain. NIIE fetches real papers, "
        "builds a knowledge graph, scores opportunities, "
        "and generates AI proposals — in real time."
    )

    query_input = st.text_input(
        "",
        placeholder="e.g.  quantum computing  ·  "
                    "gene editing  ·  neuromorphic chips",
        key="live_query",
        label_visibility="collapsed"
    )
    run_btn = st.button("🚀  Run Full Analysis", type="primary")

    if run_btn and query_input.strip():
        from pipeline import run_pipeline
        log  = st.empty()
        prog = st.progress(0)
        steps = []
        cnt   = [0]

        def upd(msg):
            steps.append(msg)
            cnt[0] += 1
            log.markdown(
                f"<div style='background:{T['surface']};"
                f"border:1px solid {T['border']};"
                f"border-radius:8px;padding:12px 16px;"
                f"font-size:0.82rem;'>"
                + "".join([
                    f"<div style='color:"
                    f"{'#10b981' if '✅' in s or '🎉' in s else T['muted']}'>"
                    f"{s}</div>"
                    for s in steps[-7:]
                ]) + "</div>",
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
            <div style='background:{T['success']}18;
                        border:1px solid {T['success']}44;
                        border-radius:8px;padding:12px 18px;
                        margin:12px 0;'>
                <span style='color:{T['success']};font-weight:700;'>
                    ✓ Complete
                </span>
                <span style='color:{T['muted']};margin-left:8px;
                             font-size:0.88rem;'>
                    {query_input}
                </span>
            </div>
            """, unsafe_allow_html=True)

            kpi_row([
                ("Papers",   result["papers"]),
                ("Edges",    result["graph"]["num_edges"]),
                ("Clusters", result["graph"]["num_clusters"]),
                ("Gaps",     result["scores"]["total_gaps"]),
            ])

            st.markdown(f"""
            <p style='color:{T['muted']};font-size:0.75rem;
                      text-transform:uppercase;letter-spacing:1px;
                      margin:16px 0 8px 0;font-weight:600;'>
                Top Opportunities
            </p>
            """, unsafe_allow_html=True)

            for i, p in enumerate(
                result["scores"]["top_opportunities"][:5], 1
            ):
                sc = p["scores"]["innovation_opportunity_score"]
                bc = T['accent'] if i<=3 else T['border']
                st.markdown(f"""
                <div style='background:{T['surface']};
                            border:1px solid {T['border']};
                            border-left:3px solid {bc};
                            border-radius:8px;
                            padding:10px 16px;margin-bottom:6px;'>
                    <span style='color:{T['accent']};
                                 font-weight:700;'>
                        #{i} {sc:.4f}
                    </span>
                    <span style='color:{T['text']};
                                 margin-left:10px;
                                 font-size:0.87rem;'>
                        {p['title']}
                    </span><br>
                    <span style='color:{T['muted']};
                                 font-size:0.75rem;'>
                        {p['year']} · {p['citations']} citations
                    </span>
                </div>
                """, unsafe_allow_html=True)

            if result["innovations"]:
                st.markdown(f"""
                <p style='color:{T['muted']};font-size:0.75rem;
                          text-transform:uppercase;letter-spacing:1px;
                          margin:16px 0 8px 0;font-weight:600;'>
                    AI Proposals
                </p>
                """, unsafe_allow_html=True)
                for i, inv in enumerate(result["innovations"], 1):
                    with st.expander(
                        f"Proposal #{i} — "
                        f"{inv['gap_title'][:55]}..."
                    ):
                        st.markdown(inv["proposal"]["raw"])
    elif run_btn:
        st.warning("Please enter a topic.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LITERATURE SEARCH
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Literature Search":
    section_header(
        "📚 Literature Search & Summarization",
        "Search any research question. NIIE finds papers, "
        "generates AI summaries, and maps the research landscape — "
        "replacing Elicit and Semantic Scholar in one place."
    )

    lit_query = st.text_input(
        "",
        placeholder="e.g.  What are the latest non-invasive BCI "
                    "methods for motor rehabilitation?",
        key="lit_query",
        label_visibility="collapsed"
    )
    lit_btn = st.button("🔍  Search Literature", type="primary")

    if lit_btn and lit_query.strip():
        from literature_search import run_literature_search
        log   = st.empty()
        steps = []

        def upd_lit(msg):
            steps.append(msg)
            log.markdown(
                f"<div style='background:{T['surface']};"
                f"border:1px solid {T['border']};"
                f"border-radius:8px;padding:12px 16px;"
                f"font-size:0.82rem;'>"
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
            kpi_row([
                ("Papers Found",     lit_result["total"]),
                ("AI Summarized",    5),
                ("Landscape",        "Generated"),
            ])

            st.markdown(f"""
            <div style='background:{T['surface']};
                        border:1px solid {T['border']};
                        border-left:3px solid {T['accent']};
                        border-radius:8px;
                        padding:20px 24px;margin:16px 0;'>
                <p style='color:{T['accent']};font-weight:700;
                          font-size:0.72rem;text-transform:uppercase;
                          letter-spacing:1px;margin:0 0 10px 0;'>
                    🌐 Research Landscape Overview
                </p>
                <p style='color:{T['text']};font-size:0.87rem;
                          line-height:1.72;margin:0;'>
                    {lit_result['landscape'].replace(chr(10),'<br>')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            for i, paper in enumerate(lit_result["papers"], 1):
                title     = paper.get("title","") or "Untitled"
                year      = paper.get("year","N/A")
                citations = paper.get("citationCount",0) or 0
                abstract  = paper.get("abstract","") or ""
                summary   = paper.get("ai_summary","")
                authors   = paper.get("authors",[]) or []
                author_str = ", ".join([
                    a.get("name","") for a in authors[:3]
                ])
                if len(authors)>3:
                    author_str += f" +{len(authors)-3} more"

                with st.expander(
                    f"{'⭐ ' if i<=3 else ''}#{i}  ·  "
                    f"{title[:65]}{'...' if len(title)>65 else ''}",
                    expanded=(i==1)
                ):
                    kpi_row([
                        ("Year",      year),
                        ("Citations", citations),
                        ("Rank",      f"#{i}"),
                    ])
                    if author_str:
                        st.markdown(
                            f"<p style='color:{T['muted']};"
                            f"font-size:0.78rem;margin:8px 0;'>"
                            f"👤 {author_str}</p>",
                            unsafe_allow_html=True
                        )
                    if i<=5 and summary:
                        st.markdown(f"""
                        <div style='background:{T['accent']}08;
                                    border:1px solid {T['accent']}22;
                                    border-radius:8px;
                                    padding:14px 16px;margin:10px 0;'>
                            <p style='color:{T['accent']};
                                      font-weight:700;font-size:0.72rem;
                                      text-transform:uppercase;
                                      letter-spacing:1px;
                                      margin:0 0 6px 0;'>
                                🤖 AI Summary
                            </p>
                            <p style='color:{T['text']};
                                      font-size:0.86rem;
                                      line-height:1.65;margin:0;'>
                                {summary}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    if abstract:
                        st.markdown(f"""
                        <p style='color:{T['muted']};
                                  font-size:0.83rem;
                                  line-height:1.6;margin:8px 0 0 0;'>
                            {abstract[:500]}
                            {'...' if len(abstract)>500 else ''}
                        </p>
                        """, unsafe_allow_html=True)
                    ext_ids = paper.get("externalIds",{}) or {}
                    doi = ext_ids.get("DOI","")
                    if doi:
                        st.markdown(
                            f"<a href='https://doi.org/{doi}' "
                            f"target='_blank' "
                            f"style='color:{T['accent']};"
                            f"font-size:0.78rem;'>"
                            f"🔗 View Paper</a>",
                            unsafe_allow_html=True
                        )
    elif lit_btn:
        st.warning("Please enter a search query.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TIMING ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Timing Advisor":
    section_header(
        "⏱️ Innovation Timing Advisor",
        "Is now the right time to publish, patent, or pivot? "
        "No other tool does this."
    )

    if all_scores:
        years      = [p["year"] for p in all_scores if p.get("year")]
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        counts       = [year_counts[y] for y in sorted_years]

        if len(sorted_years) >= 3:
            recent_3   = sum(counts[-3:])
            prev_3     = sum(counts[-6:-3]) if len(counts)>=6 \
                         else sum(counts[:-3]) or 1
            mom_ratio  = recent_3 / prev_3

            if mom_ratio >= 1.8:
                status="#10b981"; label="🚀 PEAK MOMENTUM"
                advice="This field is exploding. Best time to publish and file patents."
                action="PUBLISH NOW · FILE PATENT NOW"
            elif mom_ratio >= 1.2:
                status=T['accent']; label="📈 GROWING"
                advice="Strong window to establish presence before the field peaks."
                action="PUBLISH SOON · START PATENT PROCESS"
            elif mom_ratio >= 0.8:
                status="#f59e0b"; label="⚖️ STABLE"
                advice="Mature field. Focus on differentiation and gap hunting."
                action="FIND GAPS · TARGET NICHE"
            else:
                status="#f43f5e"; label="📉 DECLINING"
                advice="Activity slowing. Pivot to adjacent emerging areas."
                action="PIVOT · FIND ADJACENT FIELD"

            st.markdown(f"""
            <div style='background:{status}12;
                        border:2px solid {status}44;
                        border-radius:12px;
                        padding:24px;margin-bottom:20px;
                        text-align:center;'>
                <p style='color:{status};font-size:1.8rem;
                          font-weight:800;margin:0 0 8px 0;'>
                    {label}
                </p>
                <p style='color:{T['text']};font-size:0.95rem;
                          margin:0 0 12px 0;line-height:1.6;'>
                    {advice}
                </p>
                <div style='background:{status}22;border-radius:6px;
                            padding:8px 20px;display:inline-block;'>
                    <span style='color:{status};font-weight:800;
                                 font-size:0.82rem;letter-spacing:1px;'>
                        {action}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            kpi_row([
                ("Recent 3yr",    recent_3),
                ("Previous 3yr",  prev_3),
                ("Momentum",      f"{mom_ratio:.2f}x"),
                ("Peak Year",
                 sorted_years[counts.index(max(counts))]),
            ])

            fig_t = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "Publication Volume by Year",
                    "Cumulative Growth"
                ]
            )
            fig_t.add_trace(go.Bar(
                x=sorted_years, y=counts,
                marker_color=[
                    status if y>=sorted_years[-3]
                    else T['border']
                    for y in sorted_years
                ],
                name="Papers/Year"
            ), row=1, col=1)

            cum = []
            tot = 0
            for c in counts:
                tot += c; cum.append(tot)
            fig_t.add_trace(go.Scatter(
                x=sorted_years, y=cum,
                mode="lines+markers",
                line=dict(color=T['accent'], width=2.5),
                marker=dict(size=6, color=T['accent']),
                fill="tozeroy",
                fillcolor=hex_to_rgba(T['accent'], 0.1),
                name="Cumulative"
            ), row=1, col=2)

            fig_t.update_layout(
                paper_bgcolor=T['surface'],
                plot_bgcolor=T['surface2'],
                font_color=T['muted'],
                height=300, showlegend=False,
                margin=dict(l=10,r=10,t=40,b=10)
            )
            fig_t.update_xaxes(gridcolor=T['border'],
                                color=T['muted'])
            fig_t.update_yaxes(gridcolor=T['border'],
                                color=T['muted'])
            st.plotly_chart(fig_t, use_container_width=True)

            # Score evolution
            year_scores = {}
            for p in all_scores:
                y = p.get("year")
                s = p["scores"]["innovation_opportunity_score"]
                if y: year_scores.setdefault(y,[]).append(s)
            avg_by_year = {
                y: round(sum(v)/len(v),4)
                for y,v in year_scores.items()
            }
            ys = sorted(avg_by_year.keys())
            vs = [avg_by_year[y] for y in ys]

            fig_evo = go.Figure(go.Scatter(
                x=ys, y=vs,
                mode="lines+markers",
                line=dict(color=T['accent2'],width=2.5),
                marker=dict(size=8,color=T['accent2'],
                            line=dict(width=1.5,
                                      color=T['surface'])),
                fill="tozeroy",
                fillcolor=hex_to_rgba(T['accent2'],0.1)
            ))
            fig_evo.update_layout(
                title="Avg Innovation Score by Year",
                paper_bgcolor=T['surface'],
                plot_bgcolor=T['surface2'],
                font_color=T['muted'],
                height=260,
                margin=dict(l=10,r=10,t=40,b=10),
                xaxis=dict(gridcolor=T['border'],color=T['muted']),
                yaxis=dict(gridcolor=T['border'],color=T['muted'],
                           range=[0,1]),
                title_font_color=T['text']
            )
            st.plotly_chart(fig_evo, use_container_width=True)

            st.markdown(f"""
            <div style='background:{T['surface']};
                        border:1px solid {T['border']};
                        border-left:3px solid {T['accent2']};
                        border-radius:8px;
                        padding:16px 20px;margin-top:8px;'>
                <p style='color:{T['accent2']};font-weight:700;
                          font-size:0.75rem;text-transform:uppercase;
                          letter-spacing:1px;margin:0 0 8px 0;'>
                    🔮 2-Year Trajectory
                </p>
                <p style='color:{T['text']};font-size:0.87rem;
                          line-height:1.65;margin:0;'>
                    Momentum ratio <strong>{mom_ratio:.2f}x</strong> →
                    {'Significant acceleration expected. 2-3x more papers by 2027.' if mom_ratio>=1.5
                     else 'Steady growth expected. Moderate increase in publications.' if mom_ratio>=1.0
                     else 'Plateau or contraction likely. Consider adjacent areas.'}
                    {scores_data['total_gaps']} gaps detected suggest
                    {'high opportunity in underexplored sub-domains.' if scores_data['total_gaps']>=15
                     else 'moderate opportunity in targeted areas.'}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run a Live Query first to enable timing analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESEARCH ROADMAP
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Research Roadmap":
    section_header(
        "🗺️ AI Research Roadmap Generator",
        "Select any gap → get a complete 6-step execution plan. "
        "No other tool does this."
    )

    if not gaps:
        st.info("No gaps loaded. Run a Live Query first.")
    else:
        gap_titles = [
            f"Gap #{i+1} [{g['scores']['innovation_opportunity_score']:.4f}]"
            f" — {g['title'][:55]}..."
            for i,g in enumerate(gaps[:10])
        ]
        idx = st.selectbox(
            "Select a research gap:",
            range(len(gap_titles)),
            format_func=lambda i: gap_titles[i],
            key="roadmap_gap"
        )
        gap = gaps[idx]

        st.markdown(f"""
        <div style='background:{T['surface']};
                    border:1px solid {T['border']};
                    border-left:3px solid {T['accent']};
                    border-radius:8px;padding:14px 18px;
                    margin:12px 0;'>
            <p style='color:{T['text']};font-weight:600;
                      font-size:0.9rem;margin:0 0 4px 0;'>
                {gap['title']}
            </p>
            <span style='color:{T['muted']};font-size:0.78rem;'>
                {gap.get('year','N/A')} ·
                {gap.get('citations',0)} citations ·
                Score: {gap['scores']['innovation_opportunity_score']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗺️  Generate Roadmap", type="primary"):
            from roadmap_generator import generate_roadmap
            with st.spinner("Generating 6-step plan..."):
                roadmap = generate_roadmap(
                    gap, "Brain-Computer Interfaces"
                )

            steps_cfg = [
                ("hypothesis","1","Hypothesis","🔬",T['accent']),
                ("methodology","2","Methodology","⚗️",T['accent2']),
                ("dataset","3","Dataset","🗄️","#10b981"),
                ("expected","4","Expected Results","📈","#f59e0b"),
                ("publication","5","Publication Strategy","📄","#06b6d4"),
                ("patent","6","Patent Strategy","⚖️","#f43f5e"),
            ]

            for key,num,label,icon,color in steps_cfg:
                content = roadmap.get(key,"") or ""
                st.markdown(f"""
                <div style='background:{T['surface']};
                            border:1px solid {T['border']};
                            border-left:4px solid {color};
                            border-radius:10px;
                            padding:18px 22px;
                            margin-bottom:10px;'>
                    <div style='display:flex;align-items:center;
                                gap:10px;margin-bottom:8px;'>
                        <span style='font-size:1.2rem;'>{icon}</span>
                        <span style='color:{color};font-weight:800;
                                     font-size:0.72rem;
                                     text-transform:uppercase;
                                     letter-spacing:1px;'>
                            Step {num}
                        </span>
                        <span style='color:{T['text']};
                                     font-weight:700;
                                     font-size:0.95rem;'>
                            {label}
                        </span>
                    </div>
                    <p style='color:{T['text']};font-size:0.87rem;
                              line-height:1.7;margin:0;'>
                        {content}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            risk = roadmap.get("risk","")
            if risk:
                st.markdown(f"""
                <div style='background:{T['surface2']};
                            border:1px solid {T['border']};
                            border-left:4px solid #f59e0b;
                            border-radius:10px;
                            padding:18px 22px;
                            margin-bottom:10px;'>
                    <div style='margin-bottom:8px;'>
                        <span style='font-size:1.2rem;'>⚠️</span>
                        <span style='color:#f59e0b;font-weight:800;
                                     font-size:0.95rem;
                                     margin-left:8px;'>
                            Risk Assessment
                        </span>
                    </div>
                    <p style='color:{T['text']};font-size:0.87rem;
                              line-height:1.7;margin:0;'>
                        {risk}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            roadmap_txt = f"""NIIE Research Roadmap
Gap: {gap['title']}
Score: {gap['scores']['innovation_opportunity_score']:.4f}

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
                "⬇️ Download Roadmap",
                roadmap_txt,
                "NIIE_Roadmap.txt",
                "text/plain"
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PAPER MAKER
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Paper Maker":
    section_header(
        "📝 Research Paper Maker",
        "Fill in your details. NIIE generates a complete "
        "IEEE-format paper — all 8 sections. Export as LaTeX "
        "for Overleaf or Markdown."
    )

    auto_gaps = "; ".join([g["title"][:60] for g in gaps[:3]]) \
                if gaps else ""
    auto_top  = str(
        all_scores[0]["scores"]["innovation_opportunity_score"]
    ) if all_scores else "0.6"

    c1,c2 = st.columns(2)
    with c1:
        pm_title = st.text_input(
            "Paper Title",
            value="NIIE: An AI-Powered Multi-Signal Innovation "
                  "Opportunity Discovery System Using Knowledge "
                  "Graphs and Predictive Scoring",
            key="pm_title"
        )
        pm_authors = st.text_input(
            "Authors",
            value="Student Name, Supervisor Name",
            key="pm_authors"
        )
        pm_domain = st.text_input(
            "Research Domain",
            value="Brain-Computer Interfaces",
            key="pm_domain"
        )
        pm_keywords = st.text_input(
            "Keywords",
            value="innovation discovery, knowledge graphs, "
                  "research gap detection, AI, patent analysis",
            key="pm_keywords"
        )
    with c2:
        pm_problem = st.text_area(
            "Problem Statement",
            value="Research knowledge, patents, and innovation "
                  "opportunities are fragmented across disconnected "
                  "systems.",
            height=100, key="pm_problem"
        )
        pm_solution = st.text_area(
            "Your Solution",
            value="NIIE is an AI-powered system that ingests papers, "
                  "constructs a semantic knowledge graph, scores "
                  "opportunities using 5 signals, and generates AI "
                  "proposals.",
            height=100, key="pm_solution"
        )

    pm_methodology = st.text_area(
        "Methodology",
        value="Semantic Scholar API → entity extraction → "
              "sentence embeddings → cosine similarity graph → "
              "5-signal Innovation Opportunity Score.",
        height=70, key="pm_meth"
    )
    pm_results = st.text_area(
        "Key Results",
        value=f"{graph_data['num_nodes']} papers, "
              f"{graph_data['num_edges']} edges, "
              f"{graph_data['num_clusters']} clusters, "
              f"{scores_data['total_gaps']} gaps, "
              f"top IOS {auto_top}.",
        height=70, key="pm_res"
    )

    if st.button("📝  Generate Full Paper", type="primary"):
        from paper_maker import (
            generate_paper, format_paper_latex,
            format_paper_markdown
        )

        paper_data = {
            "title":pm_title,"authors":pm_authors,
            "domain":pm_domain,"problem":pm_problem,
            "solution":pm_solution,"methodology":pm_methodology,
            "results":pm_results,"keywords":pm_keywords,
            "gaps_found":auto_gaps,"top_score":auto_top,
            "num_papers":graph_data["num_nodes"],
            "num_clusters":graph_data["num_clusters"],
            "num_gaps":scores_data["total_gaps"],
        }

        section_labels = {
            "abstract"    : "Abstract",
            "introduction": "1. Introduction",
            "related_work": "2. Related Work",
            "methodology" : "3. Methodology",
            "results"     : "4. Results",
            "discussion"  : "5. Discussion",
            "conclusion"  : "6. Conclusion",
            "references"  : "References",
        }

        prog_bar   = st.progress(0)
        status_txt = st.empty()
        sections   = {}
        completed  = [0]

        st.markdown("---")
        st.markdown(f"""
        
            Generated Sections
        

        """, unsafe_allow_html=True)

        live_container = st.container()

        def on_section_done(key, content):
            completed[0] += 1
            prog_bar.progress(completed[0] / 8)
            status_txt.markdown(
                f"<p style='color:{T['muted']};font-size:0.83rem;'>"
                f"✅ {section_labels.get(key, key)} written "
                f"({completed[0]}/8)</p>",
                unsafe_allow_html=True
            )
            sections[key] = content
            with live_container:
                with st.expander(
                    section_labels.get(key, key),
                    expanded=False
                ):
                    st.markdown(
                        f"<div style='color:{T['text']};"
                        f"font-size:0.87rem;line-height:1.75;'>"
                        f"{content.replace(chr(10),'<br>')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        from paper_maker import (
            generate_paper, format_paper_latex,
            format_paper_markdown
        )
        generate_paper(paper_data, progress_callback=on_section_done)
        prog_bar.progress(1.0)
        status_txt.empty()
        st.success("✓ Paper complete — 8 sections · IEEE format · Ready for Overleaf")

        st.markdown("---")
        latex_out = format_paper_latex(paper_data, sections)
        md_out    = format_paper_markdown(paper_data, sections)

        d1,d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ LaTeX for Overleaf (.tex)",
                latex_out, "NIIE_Paper.tex", "text/plain"
            )
        with d2:
            st.download_button(
                "⬇️ Markdown (.md)",
                md_out, "NIIE_Paper.md", "text/plain"
            )

        st.markdown(f"""
        <div style='background:{T['surface2']};
                    border:1px solid {T['border']};
                    border-radius:8px;padding:12px 16px;
                    margin-top:12px;'>
            <p style='color:{T['muted']};font-size:0.82rem;margin:0;'>
                📋 Next: Download .tex →
                <a href='https://overleaf.com' target='_blank'
                   style='color:{T['accent']};'>overleaf.com</a>
                → New Project → Upload → Compile.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DOMAIN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Domain Comparison":
    section_header(
        "⚔️ Domain Comparison",
        "Run two domains simultaneously. "
        "Head-to-head innovation analysis."
    )

    c1,vs,c2 = st.columns([5,1,5])
    with c1:
        domain_a = st.text_input(
            "Domain A",
            placeholder="e.g. brain-computer interfaces",
            key="domain_a"
        )
    with vs:
        st.markdown(
            f"<div style='text-align:center;padding-top:28px;"
            f"color:{T['accent']};font-weight:800;"
            f"font-size:1.3rem;'>VS</div>",
            unsafe_allow_html=True
        )
    with c2:
        domain_b = st.text_input(
            "Domain B",
            placeholder="e.g. neuromorphic chips",
            key="domain_b"
        )

    cmp_btn = st.button("⚔️  Run Comparison", type="primary")

    if cmp_btn and domain_a.strip() and domain_b.strip():
        from pipeline import run_pipeline
        st.markdown("---")
        col_a, col_b = st.columns(2)
        results = {}

        for domain, col in [
            (domain_a.strip(), col_a),
            (domain_b.strip(), col_b)
        ]:
            with col:
                st.markdown(f"""
                <div style='background:{T['surface']};
                            border:1px solid {T['border']};
                            border-top:3px solid {T['accent']};
                            border-radius:8px;
                            padding:10px 14px;margin-bottom:10px;'>
                    <span style='color:{T['accent']};
                                 font-weight:700;
                                 font-size:0.85rem;'>
                        {domain.upper()}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                log   = st.empty()
                steps = []
                cnt   = [0]

                def upd_c(msg,steps=steps,cnt=cnt,log=log):
                    steps.append(msg)
                    cnt[0] += 1
                    log.markdown(
                        f"<div style='background:{T['surface2']};"
                        f"border:1px solid {T['border']};"
                        f"border-radius:6px;padding:8px 12px;"
                        f"font-size:0.76rem;color:{T['muted']};'>"
                        + "<br>".join(steps[-4:])
                        + "</div>",
                        unsafe_allow_html=True
                    )

                with st.spinner(f"Analysing {domain}..."):
                    res, err = run_pipeline(domain, upd_c)

                if err:
                    st.error(err)
                else:
                    results[domain] = res
                    s = res["scores"]
                    m1,m2 = st.columns(2)
                    m1.metric("Papers", res["papers"])
                    m2.metric("Gaps",   s["total_gaps"])
                    m1.metric("Edges",  res["graph"]["num_edges"])
                    m2.metric("Clusters",res["graph"]["num_clusters"])

                    st.markdown(f"""
                    <p style='color:{T['muted']};font-size:0.72rem;
                              text-transform:uppercase;letter-spacing:1px;
                              margin:10px 0 6px 0;font-weight:600;'>
                        Top Opportunities
                    </p>
                    """, unsafe_allow_html=True)
                    for i,p in enumerate(
                        s["top_opportunities"][:3],1
                    ):
                        sc = p["scores"][
                            "innovation_opportunity_score"
                        ]
                        st.markdown(f"""
                        <div style='background:{T['surface2']};
                                    border-left:2px solid {T['accent']};
                                    border-radius:0 6px 6px 0;
                                    padding:7px 11px;
                                    margin-bottom:4px;'>
                            <span style='color:{T['accent']};
                                         font-weight:700;
                                         font-size:0.8rem;'>
                                #{i} {sc:.4f}
                            </span><br>
                            <span style='color:{T['text']};
                                         font-size:0.76rem;'>
                                {p['title'][:48]}...
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

        if len(results)==2:
            st.markdown("---")
            st.markdown(f"""
            <h4 style='color:{T['text']};font-weight:700;
                        margin-bottom:12px;'>
                Head-to-Head Analysis
            </h4>
            """, unsafe_allow_html=True)

            domains = list(results.keys())
            signals = ["research_momentum","citation_density",
                       "keyword_novelty","isolation_score",
                       "patent_gap_score"]
            slabels = ["Momentum","Cit. Density","Novelty",
                       "Isolation","Patent Gap"]

            def avg_sig(res, sig):
                sc = res["scores"]["all_scores"]
                v  = [p["scores"].get(sig,0) for p in sc]
                return sum(v)/len(v) if v else 0

            avg_a = [avg_sig(results[domains[0]],s) for s in signals]
            avg_b = [avg_sig(results[domains[1]],s) for s in signals]

            def avg_inn(res):
                sc = res["scores"]["all_scores"]
                v  = [p["scores"]["innovation_opportunity_score"]
                      for p in sc]
                return round(sum(v)/len(v),4) if v else 0

            inn_a  = avg_inn(results[domains[0]])
            inn_b  = avg_inn(results[domains[1]])
            winner = domains[0] if inn_a>=inn_b else domains[1]

            st.markdown(f"""
            <div style='background:{T['success']}12;
                        border:1px solid {T['success']}44;
                        border-radius:10px;padding:14px 20px;
                        margin-bottom:16px;text-align:center;'>
                <span style='color:{T['success']};
                             font-size:1.05rem;font-weight:800;'>
                    🏆 Higher Opportunity:
                </span>
                <span style='color:{T['text']};font-weight:700;
                             margin-left:8px;'>
                    {winner.upper()}
                </span>
                <span style='color:{T['muted']};font-size:0.85rem;
                             margin-left:6px;'>
                    ({max(inn_a,inn_b):.4f})
                </span>
            </div>
            """, unsafe_allow_html=True)

            ch1,ch2 = st.columns(2)
            with ch1:
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(
                    name=domains[0][:18], x=slabels, y=avg_a,
                    marker_color=T['accent'],
                    text=[f"{v:.3f}" for v in avg_a],
                    textposition="outside",
                    textfont=dict(size=9,color=T['muted'])
                ))
                fig_c.add_trace(go.Bar(
                    name=domains[1][:18], x=slabels, y=avg_b,
                    marker_color=T['accent2'],
                    text=[f"{v:.3f}" for v in avg_b],
                    textposition="outside",
                    textfont=dict(size=9,color=T['muted'])
                ))
                fig_c.update_layout(
                    title="Signal Comparison", barmode="group",
                    paper_bgcolor=T['surface'],
                    plot_bgcolor=T['surface2'],
                    font_color=T['muted'], height=300,
                    margin=dict(l=10,r=10,t=40,b=10),
                    legend=dict(font_size=9,bgcolor=T['surface']),
                    yaxis=dict(range=[0,1],gridcolor=T['border']),
                    xaxis=dict(gridcolor=T['border']),
                    title_font_color=T['text']
                )
                st.plotly_chart(fig_c, use_container_width=True)

            with ch2:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=avg_a+[avg_a[0]], theta=slabels+[slabels[0]],
                    fill="toself", name=domains[0][:18],
                    line_color=T['accent'],
                    fillcolor=hex_to_rgba(T['accent'],0.13)
                ))
                fig_r.add_trace(go.Scatterpolar(
                    r=avg_b+[avg_b[0]], theta=slabels+[slabels[0]],
                    fill="toself", name=domains[1][:18],
                    line_color=T['accent2'],
                    fillcolor=hex_to_rgba(T['accent2'],0.13)
                ))
                fig_r.update_layout(
                    title="Radar Comparison",
                    polar=dict(
                        bgcolor=T['surface2'],
                        radialaxis=dict(visible=True,range=[0,1],
                                        gridcolor=T['border'],
                                        color=T['muted']),
                        angularaxis=dict(gridcolor=T['border'],
                                         color=T['muted'])
                    ),
                    paper_bgcolor=T['surface'],
                    font_color=T['muted'], height=300,
                    margin=dict(l=10,r=10,t=40,b=10),
                    legend=dict(font_size=9,bgcolor=T['surface']),
                    title_font_color=T['text']
                )
                st.plotly_chart(fig_r, use_container_width=True)

    elif cmp_btn:
        st.warning("Enter both domains.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: QUERY HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Query History":
    section_header(
        "📂 Query History",
        "Every analysis saved automatically. "
        "Revisit without re-running."
    )

    history = load_history()

    if not history:
        st.info("No history yet. Run a Live Query to get started.")
    else:
        st.markdown(f"""
        <div style='background:{T['success']}12;
                    border:1px solid {T['success']}33;
                    border-radius:8px;padding:10px 16px;
                    margin-bottom:16px;display:inline-block;'>
            <span style='color:{T['success']};font-weight:700;'>
                {len(history)} saved
                {'analysis' if len(history)==1 else 'analyses'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        for entry in history:
            inn = [
                p["scores"]["innovation_opportunity_score"]
                for p in entry["all_scores"]
            ]
            avg = round(sum(inn)/len(inn),4) if inn else 0
            top = round(max(inn),4) if inn else 0

            with st.expander(
                f"🔍 {entry['query'].title()} · "
                f"{entry['timestamp']}",
                expanded=False
            ):
                kpi_row([
                    ("Papers", entry["papers"]),
                    ("Edges",  entry["graph"]["num_edges"]),
                    ("Gaps",   entry["total_gaps"]),
                    ("Top Score", top),
                ])

                l,r = st.columns([3,2])
                with l:
                    for i,p in enumerate(
                        entry["top_opportunities"][:3],1
                    ):
                        sc = p["scores"][
                            "innovation_opportunity_score"
                        ]
                        st.markdown(f"""
                        <div style='background:{T['surface2']};
                                    border-left:2px solid {T['accent']};
                                    border-radius:0 6px 6px 0;
                                    padding:7px 11px;
                                    margin-bottom:4px;'>
                            <span style='color:{T['accent']};
                                         font-weight:700;
                                         font-size:0.8rem;'>
                                #{i} {sc:.4f}
                            </span><br>
                            <span style='color:{T['text']};
                                         font-size:0.76rem;'>
                                {p['title'][:52]}...
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                with r:
                    if inn:
                        fig_h = go.Figure(go.Histogram(
                            x=inn, nbinsx=12,
                            marker_color=T['accent']
                        ))
                        fig_h.update_layout(
                            paper_bgcolor=T['surface2'],
                            plot_bgcolor=T['surface2'],
                            height=160,showlegend=False,
                            margin=dict(l=8,r=8,t=8,b=8),
                            xaxis=dict(color=T['muted'],
                                       gridcolor=T['border']),
                            yaxis=dict(color=T['muted'],
                                       gridcolor=T['border']),
                            font_color=T['muted']
                        )
                        fig_h.update_traces(marker_line_width=0)
                        st.plotly_chart(
                            fig_h, use_container_width=True
                        )

                if st.button(
                    "🗑️ Delete",
                    key=f"del_{entry['_file'].stem}"
                ):
                    delete_history_entry(entry["_file"])
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Knowledge Graph":
    section_header(
        "🌐 Knowledge Graph",
        "Each node is a paper · Edges = semantic similarity · "
        "Node size = centrality · Colour = cluster"
    )

    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    COLORS = [
        T['accent'],T['accent2'],"#f43f5e","#f59e0b","#10b981",
        "#06b6d4","#8b5cf6","#ec4899","#14b8a6","#f97316",
        "#6366f1","#84cc16"
    ]

    nx_arr,ny_arr,nt_arr,nc_arr,ns_arr = [],[],[],[],[]
    for node in nodes:
        idx = node["id"]
        nx_arr.append(float(coords[idx,0]))
        ny_arr.append(float(coords[idx,1]))
        cl = node.get("cluster",0)
        nc_arr.append(COLORS[cl%len(COLORS)])
        ns_arr.append(9+node.get("combined_score",0)*70)
        nt_arr.append(
            f"<b>{node['title'][:52]}...</b><br>"
            f"Year: {node.get('year','N/A')} · "
            f"Citations: {node.get('citations',0)}<br>"
            f"Cluster {cl} · "
            f"{', '.join(node.get('keywords',[])[:3])}"
        )

    ex,ey = [],[]
    npos = {
        n["id"]:(float(coords[n["id"],0]),float(coords[n["id"],1]))
        for n in nodes
    }
    for e in edges[:380]:
        x0,y0=npos[e["source"]]; x1,y1=npos[e["target"]]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]

    fig_g = go.Figure([
        go.Scatter(x=ex,y=ey,mode="lines",
                   line=dict(width=0.6,color=T['border']),
                   hoverinfo="none"),
        go.Scatter(x=nx_arr,y=ny_arr,mode="markers",
                   hoverinfo="text",text=nt_arr,
                   marker=dict(size=ns_arr,color=nc_arr,
                               line=dict(width=0.5,
                                         color=T['surface']),
                               opacity=0.88))
    ])
    fig_g.update_layout(
        showlegend=False,hovermode="closest",height=580,
        paper_bgcolor=T['surface'],plot_bgcolor=T['surface'],
        xaxis=dict(showgrid=False,zeroline=False,
                   showticklabels=False,showline=False),
        yaxis=dict(showgrid=False,zeroline=False,
                   showticklabels=False,showline=False),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig_g, use_container_width=True)

    sizes = graph_data["cluster_sizes"]
    cols  = st.columns(min(len(sizes),6))
    for i,sz in enumerate(sizes[:6]):
        c = COLORS[i%len(COLORS)]
        cols[i].markdown(
            f"<div style='background:{c}12;"
            f"border:1px solid {c}33;border-radius:6px;"
            f"padding:7px 10px;text-align:center;'>"
            f"<span style='color:{c};font-weight:700;"
            f"font-size:0.8rem;'>Cluster {i+1}</span><br>"
            f"<span style='color:{T['muted']};"
            f"font-size:0.74rem;'>{sz} papers</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INNOVATION SCORES
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Innovation Scores":
    section_header(
        "📊 Innovation Opportunity Scores",
        "5-signal model: Momentum · Citation Density · "
        "Novelty · Isolation · Patent Gap"
    )

    df = pd.DataFrame([{
        "Title"      : p["title"][:55]+"...",
        "Year"       : p["year"],
        "Citations"  : p["citations"],
        "Momentum"   : p["scores"]["research_momentum"],
        "Cit. Density": p["scores"]["citation_density"],
        "Novelty"    : p["scores"]["keyword_novelty"],
        "Isolation"  : p["scores"]["isolation_score"],
        "Patent Gap" : p["scores"].get("patent_gap_score",0),
        "Score"      : p["scores"]["innovation_opportunity_score"],
    } for p in all_scores])

    ca,cb = st.columns(2)
    with ca:
        fig_h = px.histogram(
            df,x="Score",nbins=20,
            title="Score Distribution",
            color_discrete_sequence=[T['accent']]
        )
        plot_cfg(fig_h,280)
        fig_h.update_traces(marker_line_width=0)
        fig_h.update_layout(title_font_color=T['text'])
        st.plotly_chart(fig_h, use_container_width=True)
    with cb:
        fig_p = go.Figure(go.Pie(
            labels=["Momentum","Cit. Density","Novelty",
                    "Isolation","Patent Gap"],
            values=[0.25,0.20,0.20,0.20,0.15],
            hole=0.6,
            marker_colors=T['chart_seq'],
            textinfo="label+percent",textfont_size=10,
        ))
        fig_p.update_layout(
            title="Signal Weights",
            title_font_color=T['text'],
            paper_bgcolor=T['surface'],height=280,
            margin=dict(l=10,r=10,t=36,b=10),
            font_color=T['muted'],showlegend=False
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.dataframe(
        df.style.background_gradient(
            subset=["Score"],cmap="Blues"
        ).format({
            "Momentum":"{:.3f}","Cit. Density":"{:.3f}",
            "Novelty":"{:.3f}","Isolation":"{:.3f}",
            "Patent Gap":"{:.3f}","Score":"{:.4f}"
        }),
        use_container_width=True,height=370
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESEARCH GAPS
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Research Gaps":
    section_header(
        f"💡 {len(gaps)} Research Gaps Detected",
        "High novelty · Low citations · "
        "Graph-isolated · Low patent coverage"
    )

    for i,gap in enumerate(gaps[:10],1):
        sc = gap["scores"]["innovation_opportunity_score"]
        with st.expander(
            f"Gap #{i} · {sc:.4f} · {gap['title'][:58]}...",
            expanded=(i==1)
        ):
            l,r = st.columns([3,2])
            with l:
                abstract = gap.get("abstract","") or "No abstract."
                st.markdown(f"""
                <p style='color:{T['text']};font-weight:600;
                          margin:0 0 6px 0;'>
                    {gap['title']}
                </p>
                <div style='display:flex;gap:20px;margin-bottom:10px;'>
                    <div>
                        <span style='color:{T['muted']};
                                     font-size:0.7rem;
                                     text-transform:uppercase;
                                     letter-spacing:1px;'>
                            Year
                        </span><br>
                        <span style='color:{T['accent']};
                                     font-weight:700;'>
                            {gap['year']}
                        </span>
                    </div>
                    <div>
                        <span style='color:{T['muted']};
                                     font-size:0.7rem;
                                     text-transform:uppercase;
                                     letter-spacing:1px;'>
                            Citations
                        </span><br>
                        <span style='color:{T['accent']};
                                     font-weight:700;'>
                            {gap['citations']}
                        </span>
                    </div>
                    <div>
                        <span style='color:{T['muted']};
                                     font-size:0.7rem;
                                     text-transform:uppercase;
                                     letter-spacing:1px;'>
                            Score
                        </span><br>
                        <span style='color:{T['accent']};
                                     font-weight:700;'>
                            {sc:.4f}
                        </span>
                    </div>
                </div>
                <p style='color:{T['muted']};font-size:0.84rem;
                          line-height:1.6;margin:0 0 10px 0;'>
                    {abstract[:320]}...
                </p>
                <div>
                    {''.join([accent_tag(kw)
                     for kw in gap['bci_keywords']])}
                </div>
                """, unsafe_allow_html=True)
            with r:
                s = gap["scores"]
                sn = ["Momentum","Cit. Density","Novelty",
                      "Isolation","Patent Gap"]
                sv = [s["research_momentum"],s["citation_density"],
                      s["keyword_novelty"],s["isolation_score"],
                      s.get("patent_gap_score",0)]
                fig_b = go.Figure(go.Bar(
                    x=sv,y=sn,orientation="h",
                    marker_color=T['chart_seq'],
                    text=[f"{v:.3f}" for v in sv],
                    textposition="outside",
                    textfont=dict(color=T['muted'],size=10)
                ))
                fig_b.update_layout(
                    paper_bgcolor=T['surface2'],
                    plot_bgcolor=T['surface2'],height=200,
                    margin=dict(l=0,r=40,t=8,b=8),
                    xaxis=dict(range=[0,1.1],showgrid=False,
                               zeroline=False,color=T['muted']),
                    yaxis=dict(color=T['muted'],tickfont_size=10),
                    font_color=T['muted']
                )
                st.plotly_chart(fig_b, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "Opportunities":
    section_header(
        "🏆 Top 10 Innovation Opportunities",
        "Highest combined score across all 5 signals"
    )

    for rank,paper in enumerate(all_scores[:10],1):
        sc    = paper["scores"]["innovation_opportunity_score"]
        medal = "🥇" if rank==1 else "🥈" if rank==2 \
                else "🥉" if rank==3 else f"#{rank}"
        bc    = T['accent'] if rank<=3 else T['border']
        bg    = T['surface2'] if rank<=3 else T['surface']

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {bc};
                    border-left:3px solid {bc};border-radius:8px;
                    padding:12px 18px;margin-bottom:7px;'>
            <span style='font-size:1rem;'>{medal}</span>
            <span style='color:{T['accent']};font-weight:700;
                         margin-left:8px;'>{sc:.4f}</span>
            <span style='color:{T['text']};margin-left:10px;
                         font-size:0.9rem;'>{paper['title']}</span>
            <br>
            <span style='color:{T['muted']};font-size:0.75rem;'>
                {paper['year']} · {paper['citations']} citations ·
                {', '.join(paper['bci_keywords'][:4])}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style='color:{T['muted']};font-size:0.75rem;
              text-transform:uppercase;letter-spacing:1px;
              margin:20px 0 10px 0;font-weight:600;'>
        Signal Breakdown — Top 3
    </p>
    """, unsafe_allow_html=True)

    cats  = ["Momentum","Cit. Density","Novelty",
             "Isolation","Patent Gap"]
    fig_r = go.Figure()
    rcols = T['chart_seq'][:3]

    for i,paper in enumerate(all_scores[:3]):
        s = paper["scores"]
        fig_r.add_trace(go.Scatterpolar(
            r=[s["research_momentum"],s["citation_density"],
               s["keyword_novelty"],s["isolation_score"],
               s.get("patent_gap_score",0)],
            theta=cats,fill="toself",
            name=paper["title"][:35]+"...",
            line_color=rcols[i],
            fillcolor=hex_to_rgba(rcols[i],0.13)
        ))

    fig_r.update_layout(
        polar=dict(
            bgcolor=T['surface2'],
            radialaxis=dict(visible=True,range=[0,1],
                            gridcolor=T['border'],
                            color=T['muted']),
            angularaxis=dict(gridcolor=T['border'],
                             color=T['muted'])
        ),
        paper_bgcolor=T['surface'],font_color=T['muted'],
        height=420,
        legend=dict(font_size=10,bgcolor=T['surface'],
                    bordercolor=T['border'])
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI PROPOSALS
# ─────────────────────────────────────────────────────────────────────────────
elif selected == "AI Proposals":
    section_header(
        "🤖 AI-Generated Innovation Proposals",
        "Generated by Llama 3.3-70B · "
        "Analyzing underexplored BCI research gaps"
    )

    inv_file = "data/processed/bci_innovations.json"
    if not os.path.exists(inv_file):
        st.warning("No proposals found. "
                   "Run `python src/llm_generator.py` first.")
    else:
        with open(inv_file,"r",encoding="utf-8") as f:
            innovations = json.load(f)

        st.markdown(f"""
        <div style='background:{T['success']}12;
                    border:1px solid {T['success']}33;
                    border-radius:8px;padding:10px 16px;
                    margin-bottom:16px;display:inline-block;'>
            <span style='color:{T['success']};font-weight:700;'>
                ✓ {len(innovations)} proposals
            </span>
            <span style='color:{T['muted']};font-size:0.82rem;
                         margin-left:6px;'>
                by Llama 3.3-70B via Groq
            </span>
        </div>
        """, unsafe_allow_html=True)

        for i,inv in enumerate(innovations,1):
            p     = inv["proposal"]
            sc    = inv["gap_score"]
            title = p.get("innovation_title","Untitled") \
                    or "Untitled"

            with st.expander(
                f"{'⭐ ' if i<=3 else ''}#{i} · {title}",
                expanded=(i==1)
            ):
                kpi_row([
                    ("Gap Score", f"{sc:.4f}"),
                    ("Year",      inv.get("gap_year","N/A")),
                    ("Citations", inv.get("gap_citations",0)),
                ])

                l,r = st.columns([3,2])
                with l:
                    for key,label in [
                        ("problem_it_solves","Problem"),
                        ("proposed_solution","Solution"),
                        ("why_novel","Why Novel"),
                        ("research_contribution","Contribution"),
                    ]:
                        val = (p.get(key,"") or "").strip()
                        if val:
                            st.markdown(f"""
                            <div style='margin-bottom:14px;'>
                                <p style='color:{T['accent']};
                                          font-weight:700;
                                          font-size:0.72rem;
                                          text-transform:uppercase;
                                          letter-spacing:1px;
                                          margin:0 0 4px 0;'>
                                    {label}
                                </p>
                                <p style='color:{T['text']};
                                          font-size:0.87rem;
                                          line-height:1.65;
                                          margin:0;'>
                                    {val}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                with r:
                    for key,label in [
                        ("technology_stack","Tech Stack"),
                        ("target_users","Target Users"),
                    ]:
                        val = (p.get(key,"") or "").strip()
                        if val:
                            st.markdown(f"""
                            <div style='background:{T['surface2']};
                                        border:1px solid {T['border']};
                                        border-radius:8px;
                                        padding:12px 14px;
                                        margin-bottom:10px;'>
                                <p style='color:{T['accent']};
                                          font-weight:700;
                                          font-size:0.7rem;
                                          text-transform:uppercase;
                                          letter-spacing:1px;
                                          margin:0 0 5px 0;'>
                                    {label}
                                </p>
                                <p style='color:{T['muted']};
                                          font-size:0.83rem;
                                          line-height:1.55;
                                          margin:0;'>
                                    {val}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    patent = (p.get("patent_opportunity","")
                              or "").strip()
                    if patent:
                        st.markdown(f"""
                        <div style='background:{T['accent']}08;
                                    border:1px solid {T['accent']}22;
                                    border-radius:8px;
                                    padding:12px 14px;
                                    margin-bottom:10px;'>
                            <p style='color:{T['accent']};
                                      font-weight:700;
                                      font-size:0.7rem;
                                      text-transform:uppercase;
                                      letter-spacing:1px;
                                      margin:0 0 5px 0;'>
                                Patent Opportunity
                            </p>
                            <p style='color:{T['text']};
                                      font-size:0.83rem;
                                      line-height:1.55;
                                      margin:0;
                                      font-style:italic;'>
                                {patent}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    impact = p.get("impact_score","")
                    if impact:
                        num = re.search(r'\d+',str(impact))
                        if num:
                            n = int(num.group())
                            st.markdown(f"""
                            <div style='background:{T['surface2']};
                                        border:1px solid {T['border']};
                                        border-radius:8px;
                                        padding:12px 14px;'>
                                <p style='color:{T['muted']};
                                          font-weight:700;
                                          font-size:0.7rem;
                                          text-transform:uppercase;
                                          letter-spacing:1px;
                                          margin:0 0 6px 0;'>
                                    Impact Score
                                </p>
                                <p style='color:{T['accent']};
                                          font-weight:800;
                                          font-size:1.4rem;
                                          margin:0 0 6px 0;'>
                                    {n}
                                    <span style='font-size:0.8rem;
                                                 color:{T['muted']};'>
                                        /10
                                    </span>
                                </p>
                            """, unsafe_allow_html=True)
                            st.progress(n/10)
                            st.markdown("</div>",
                                        unsafe_allow_html=True)

                st.markdown(f"""
                <div style='margin-top:12px;padding-top:10px;
                            border-top:1px solid {T['border']};'>
                    <p style='color:{T['muted']};font-size:0.7rem;
                              text-transform:uppercase;
                              letter-spacing:1px;margin:0 0 4px 0;'>
                        Source Paper
                    </p>
                    <p style='color:{T['text']};font-size:0.83rem;
                              font-style:italic;margin:0 0 6px 0;'>
                        {inv['gap_title']}
                    </p>
                    {''.join([accent_tag(kw)
                     for kw in inv.get('gap_keywords',[])])}
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:20px 0 6px 0;
            border-top:1px solid {T['border']};
            margin-top:24px;'>
    <span style='color:{T['muted']};font-size:0.74rem;'>
        NIIE · National Innovation Intelligence Engine ·
        Python · Streamlit · NetworkX ·
        Sentence-Transformers · Plotly · Groq
    </span>
</div>
""", unsafe_allow_html=True)