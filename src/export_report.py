import json
import os
import io
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas


# ── BRAND COLORS ──────────────────────────────────────────────────────────────
NIIE_DARK    = colors.HexColor("#0a0a0f")
NIIE_ACCENT  = colors.HexColor("#6366f1")
NIIE_ACCENT2 = colors.HexColor("#8b5cf6")
NIIE_TEXT    = colors.HexColor("#1e293b")
NIIE_MUTED   = colors.HexColor("#64748b")
NIIE_SURFACE = colors.HexColor("#f8fafc")
NIIE_BORDER  = colors.HexColor("#e2e8f0")
NIIE_SUCCESS = colors.HexColor("#10b981")
NIIE_WHITE   = colors.white


# ── PAGE TEMPLATE WITH HEADER/FOOTER ─────────────────────────────────────────
class NIIEPageTemplate:
    def __init__(self, domain, generated_at):
        self.domain       = domain
        self.generated_at = generated_at

    def on_page(self, canv, doc):
        w, h = A4

        # Top bar
        canv.setFillColor(NIIE_DARK)
        canv.rect(0, h - 1.2*cm, w, 1.2*cm, fill=1, stroke=0)

        # Logo text
        canv.setFillColor(NIIE_WHITE)
        canv.setFont("Helvetica-Bold", 10)
        canv.drawString(1.5*cm, h - 0.85*cm, "NIIE")
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#94a3b8"))
        canv.drawString(3.0*cm, h - 0.85*cm,
                        "National Innovation Intelligence Engine")

        # Domain pill (top right)
        canv.setFont("Helvetica", 7)
        canv.setFillColor(colors.HexColor("#6366f1"))
        domain_text = f"Domain: {self.domain}"
        canv.drawRightString(w - 1.5*cm, h - 0.85*cm, domain_text)

        # Bottom bar
        canv.setFillColor(NIIE_DARK)
        canv.rect(0, 0, w, 0.8*cm, fill=1, stroke=0)

        # Footer text
        canv.setFillColor(colors.HexColor("#64748b"))
        canv.setFont("Helvetica", 7)
        canv.drawString(1.5*cm, 0.28*cm,
                        f"Generated: {self.generated_at}  ·  NIIE v1.0  ·  Confidential")
        canv.setFillColor(colors.HexColor("#6366f1"))
        canv.drawRightString(w - 1.5*cm, 0.28*cm,
                             f"Page {doc.page}")

        # Accent line under top bar
        canv.setStrokeColor(NIIE_ACCENT)
        canv.setLineWidth(2)
        canv.line(0, h - 1.2*cm, w, h - 1.2*cm)


def build_styles():
    styles = getSampleStyleSheet()

    custom = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=NIIE_WHITE,
            leading=34,
            spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Helvetica",
            fontSize=13,
            textColor=colors.HexColor("#94a3b8"),
            leading=18,
            spaceAfter=6,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#64748b"),
            leading=14,
        ),
        "section_title": ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=NIIE_ACCENT,
            leading=18,
            spaceBefore=16,
            spaceAfter=6,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=NIIE_TEXT,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=NIIE_TEXT,
            leading=14,
            spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "muted",
            fontName="Helvetica",
            fontSize=8,
            textColor=NIIE_MUTED,
            leading=12,
            spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=NIIE_MUTED,
            leading=10,
            spaceAfter=2,
        ),
        "value_accent": ParagraphStyle(
            "value_accent",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=NIIE_ACCENT,
            leading=14,
        ),
        "patent": ParagraphStyle(
            "patent",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=NIIE_TEXT,
            leading=13,
            leftIndent=12,
            spaceAfter=6,
        ),
    }
    return custom


def divider(color=NIIE_BORDER, thickness=0.5):
    return HRFlowable(
        width="100%",
        thickness=thickness,
        color=color,
        spaceAfter=6,
        spaceBefore=6
    )


def kpi_table(data, styles):
    """4-column KPI row."""
    header_row = []
    value_row  = []
    for label, value in data:
        header_row.append(Paragraph(label, styles["label"]))
        value_row.append(Paragraph(str(value), styles["value_accent"]))

    t = Table(
        [header_row, value_row],
        colWidths=[4.2*cm]*4
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NIIE_SURFACE),
        ("BOX",         (0,0), (-1,-1), 0.5, NIIE_BORDER),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, NIIE_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING",(0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


def scores_table(all_scores, styles):
    """Top 10 scores table."""
    headers = ["Rank","Title","Year","Cit.","Momentum",
               "Novelty","Isolation","Score"]
    rows = [headers]

    for i, p in enumerate(all_scores[:10], 1):
        s = p["scores"]
        rows.append([
            str(i),
            p["title"][:42] + ("..." if len(p["title"]) > 42 else ""),
            str(p["year"] or "N/A"),
            str(p["citations"]),
            f"{s['research_momentum']:.3f}",
            f"{s['keyword_novelty']:.3f}",
            f"{s['isolation_score']:.3f}",
            f"{s['innovation_opportunity_score']:.4f}",
        ])

    col_w = [0.8*cm, 6.2*cm, 1.1*cm, 1.0*cm,
             1.6*cm, 1.4*cm, 1.5*cm, 1.6*cm]

    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0,0), (-1,0), NIIE_DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), NIIE_WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0), 7),
        ("TOPPADDING",   (0,0), (-1,0), 6),
        ("BOTTOMPADDING",(0,0), (-1,0), 6),
        # Body
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,1), (-1,-1), 7),
        ("TEXTCOLOR",    (0,1), (-1,-1), NIIE_TEXT),
        ("TOPPADDING",   (0,1), (-1,-1), 5),
        ("BOTTOMPADDING",(0,1), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        # Alternating rows
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[NIIE_WHITE, NIIE_SURFACE]),
        # Score column highlight
        ("TEXTCOLOR",    (-1,1), (-1,-1), NIIE_ACCENT),
        ("FONTNAME",     (-1,1), (-1,-1), "Helvetica-Bold"),
        # Grid
        ("GRID",         (0,0), (-1,-1), 0.3, NIIE_BORDER),
    ]))
    return t


def gap_block(gap, rank, styles):
    """Single research gap block."""
    s     = gap["scores"]
    score = s["innovation_opportunity_score"]
    kws   = ", ".join(gap["bci_keywords"][:5])
    abstract = (gap.get("abstract","") or "")[:280]

    elements = []

    # Gap header
    elements.append(Paragraph(
        f"Gap #{rank} — Innovation Score: {score:.4f}",
        styles["subsection"]
    ))
    elements.append(Paragraph(gap["title"], styles["body"]))
    elements.append(Spacer(1, 4))

    # Meta row
    meta_data = [
        [Paragraph("YEAR", styles["label"]),
         Paragraph("CITATIONS", styles["label"]),
         Paragraph("MOMENTUM", styles["label"]),
         Paragraph("NOVELTY", styles["label"]),
         Paragraph("PATENT GAP", styles["label"])],
        [Paragraph(str(gap["year"] or "N/A"), styles["value_accent"]),
         Paragraph(str(gap["citations"]), styles["value_accent"]),
         Paragraph(f"{s['research_momentum']:.3f}", styles["value_accent"]),
         Paragraph(f"{s['keyword_novelty']:.3f}", styles["value_accent"]),
         Paragraph(f"{s.get('patent_gap_score',0):.3f}", styles["value_accent"])],
    ]
    meta_t = Table(meta_data, colWidths=[3.4*cm]*5)
    meta_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), NIIE_SURFACE),
        ("BOX",          (0,0), (-1,-1), 0.3, NIIE_BORDER),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, NIIE_BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ]))
    elements.append(meta_t)
    elements.append(Spacer(1, 6))

    if abstract:
        elements.append(Paragraph(abstract + "...", styles["muted"]))

    if kws:
        elements.append(Paragraph(f"Keywords: {kws}", styles["muted"]))

    elements.append(divider())
    return elements


def proposal_block(inv, rank, styles):
    """Single AI innovation proposal block."""
    p     = inv["proposal"]
    score = inv["gap_score"]

    elements = []
    title = p.get("innovation_title","Untitled") or "Untitled"

    elements.append(Paragraph(
        f"Proposal #{rank}: {title}",
        styles["subsection"]
    ))
    elements.append(Paragraph(
        f"Source gap score: {score:.4f}  ·  Year: {inv.get('gap_year','N/A')}  "
        f"·  Citations: {inv.get('gap_citations',0)}",
        styles["muted"]
    ))
    elements.append(Spacer(1, 6))

    sections = [
        ("problem_it_solves",    "Problem It Solves"),
        ("proposed_solution",    "Proposed Solution"),
        ("why_novel",            "Why It Is Novel"),
        ("technology_stack",     "Technology Stack"),
        ("target_users",         "Target Users"),
        ("research_contribution","Research Contribution"),
    ]

    for key, label in sections:
        val = (p.get(key,"") or "").strip()
        if val:
            elements.append(Paragraph(label.upper(), styles["label"]))
            elements.append(Paragraph(val[:400], styles["body"]))
            elements.append(Spacer(1, 4))

    patent = (p.get("patent_opportunity","") or "").strip()
    if patent:
        elements.append(Paragraph("PATENT OPPORTUNITY", styles["label"]))
        elements.append(Paragraph(patent[:300], styles["patent"]))
        elements.append(Spacer(1, 4))

    impact = p.get("impact_score","")
    if impact:
        import re
        num = re.search(r'\d+', str(impact))
        if num:
            n = int(num.group())
            elements.append(Paragraph(
                f"IMPACT SCORE: {n}/10",
                styles["label"]
            ))
            # Simple bar
            bar_data = [["" * n + " " * (10-n)]]
            bar_t = Table(bar_data, colWidths=[17*cm])
            filled = colors.HexColor("#6366f1")
            bar_t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ]))
            elements.append(Spacer(1,2))

    elements.append(divider(NIIE_ACCENT, thickness=0.3))
    elements.append(Spacer(1, 4))
    return elements


def generate_report(
    scores_data,
    graph_data,
    innovations,
    domain="Brain-Computer Interfaces",
    output_path="reports/NIIE_Innovation_Report.pdf"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")
    page_tmpl    = NIIEPageTemplate(domain, generated_at)
    styles       = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin    = 1.8*cm,
        bottomMargin = 1.4*cm,
        leftMargin   = 1.5*cm,
        rightMargin  = 1.5*cm,
    )

    story = []
    W, H  = A4

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    # Dark cover background via a full-page colored table
    cover_bg = Table(
        [[Paragraph("", styles["body"])]],
        colWidths=[17*cm], rowHeights=[22*cm]
    )
    cover_bg.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NIIE_DARK),
        ("BOX",        (0,0), (-1,-1), 0, NIIE_DARK),
    ]))
    story.append(cover_bg)

    # Overlay content — use a nested table on top
    story = []  # restart — we'll do cover inline

    # Cover block
    cover_content = [
        Spacer(1, 3*cm),
        Paragraph("NIIE", ParagraphStyle(
            "logo_big", fontName="Helvetica-Bold",
            fontSize=48, textColor=NIIE_ACCENT, leading=52
        )),
        Spacer(1, 0.3*cm),
        Paragraph("National Innovation Intelligence Engine",
                  styles["cover_title"]),
        Spacer(1, 0.3*cm),
        Paragraph("Innovation Opportunity Discovery Report",
                  styles["cover_sub"]),
        Spacer(1, 0.8*cm),
        divider(color=NIIE_ACCENT, thickness=1.5),
        Spacer(1, 0.8*cm),
        Paragraph(f"Domain: {domain}", ParagraphStyle(
            "domain_pill", fontName="Helvetica-Bold",
            fontSize=12, textColor=NIIE_ACCENT2
        )),
        Spacer(1, 0.4*cm),
        Paragraph(
            f"Generated on {generated_at}",
            styles["cover_meta"]
        ),
        Paragraph(
            f"Papers Analysed: {graph_data['num_nodes']}  ·  "
            f"Graph Edges: {graph_data['num_edges']}  ·  "
            f"Research Clusters: {graph_data['num_clusters']}  ·  "
            f"Gaps Detected: {scores_data['total_gaps']}",
            styles["cover_meta"]
        ),
        Spacer(1, 2*cm),
        Paragraph(
            "Powered by Python · Streamlit · NetworkX · "
            "Sentence-Transformers · Plotly · Groq LLaMA 3.3",
            styles["cover_meta"]
        ),
        PageBreak(),
    ]

    story.extend(cover_content)

    # ── SECTION 1: EXECUTIVE SUMMARY ─────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles["section_title"]))
    story.append(divider())
    story.append(Paragraph(
        f"This report presents the findings of the National Innovation Intelligence Engine (NIIE) "
        f"applied to the domain of <b>{domain}</b>. The system analysed {graph_data['num_nodes']} "
        f"research papers from Semantic Scholar, constructed a semantic knowledge graph with "
        f"{graph_data['num_edges']} edges across {graph_data['num_clusters']} research clusters, "
        f"and applied a 5-signal innovation scoring model to identify "
        f"{scores_data['total_gaps']} high-potential research gaps.",
        styles["body"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The Innovation Opportunity Score combines five weighted signals: Research Momentum (25%), "
        "Citation Density (20%), Keyword Novelty (20%), Graph Isolation Score (20%), and "
        "Patent Gap Score (15%). Areas scoring highest represent the most underexplored, "
        "high-momentum research directions with low patent coverage — the optimal zones for "
        "novel research, startup development, and patent filing.",
        styles["body"]
    ))
    story.append(Spacer(1, 12))

    # KPI row
    story.append(kpi_table([
        ("PAPERS ANALYSED",   graph_data["num_nodes"]),
        ("GRAPH CONNECTIONS", graph_data["num_edges"]),
        ("RESEARCH CLUSTERS", graph_data["num_clusters"]),
        ("GAPS DETECTED",     scores_data["total_gaps"]),
    ], styles))
    story.append(Spacer(1, 16))

    # ── SECTION 2: TOP 10 OPPORTUNITIES ──────────────────────────────────────
    story.append(Paragraph("2. Top 10 Innovation Opportunities",
                           styles["section_title"]))
    story.append(divider())
    story.append(Paragraph(
        "The following papers represent the highest-scoring innovation opportunities "
        "identified by the NIIE scoring engine across all five signals.",
        styles["body"]
    ))
    story.append(Spacer(1, 8))
    story.append(scores_table(scores_data["all_scores"], styles))
    story.append(Spacer(1, 16))

    # ── SECTION 3: RESEARCH GAPS ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Research Gap Analysis",
                           styles["section_title"]))
    story.append(divider())
    story.append(Paragraph(
        "Research gaps are identified as areas with high keyword novelty, "
        "low citation density, graph isolation, and low patent coverage. "
        "These represent the most promising directions for future research and innovation.",
        styles["body"]
    ))
    story.append(Spacer(1, 8))

    gaps = scores_data.get("research_gaps", [])
    for i, gap in enumerate(gaps[:5], 1):
        story.extend(gap_block(gap, i, styles))

    # ── SECTION 4: AI INNOVATION PROPOSALS ────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. AI-Generated Innovation Proposals",
                           styles["section_title"]))
    story.append(divider())
    story.append(Paragraph(
        f"The following innovation proposals were autonomously generated by "
        f"Llama 3.3-70B (via Groq) based on the top-scoring research gaps. "
        f"Each proposal includes a problem statement, solution architecture, "
        f"novelty analysis, technology stack, target users, and a patentable claim.",
        styles["body"]
    ))
    story.append(Spacer(1, 8))

    for i, inv in enumerate(innovations[:3], 1):
        story.extend(proposal_block(inv, i, styles))

    # ── SECTION 5: METHODOLOGY ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Methodology", styles["section_title"]))
    story.append(divider())

    method_steps = [
        ("Data Acquisition",
         "Research papers were collected from Semantic Scholar using domain-specific "
         "queries. Each paper's title, abstract, year, and citation count were retrieved."),
        ("Entity Extraction",
         "Domain-specific keywords were extracted from each paper using pattern matching "
         "against a curated BCI keyword taxonomy covering 30 technical terms."),
        ("Embedding Generation",
         "Paper abstracts and titles were encoded into 384-dimensional semantic vectors "
         "using the all-MiniLM-L6-v2 sentence transformer model, accelerated on GPU."),
        ("Knowledge Graph Construction",
         "A semantic similarity graph was constructed using cosine similarity between "
         "paper embeddings. Edges were created between papers exceeding a 0.65 similarity "
         "threshold. Graph centrality metrics (degree, betweenness) were computed for "
         "each node using NetworkX."),
        ("Innovation Opportunity Scoring",
         "Each paper was scored on five signals: Research Momentum (publication recency), "
         "Citation Density (normalized citation count), Keyword Novelty (emerging vs. "
         "saturated keyword ratio), Isolation Score (low centrality, high betweenness), "
         "and Patent Gap Score (inverse patent landscape density). Signals were combined "
         "using a weighted linear combination."),
        ("AI Proposal Generation",
         "Top-scoring research gaps were submitted to Llama 3.3-70B via Groq API with "
         "structured prompts requesting innovation proposals covering problem statement, "
         "solution, novelty, technology stack, target users, patent opportunity, "
         "research contribution, and impact score."),
    ]

    for step, desc in method_steps:
        story.append(Paragraph(step, styles["subsection"]))
        story.append(Paragraph(desc, styles["body"]))
        story.append(Spacer(1, 4))

    # ── SECTION 6: SIGNAL WEIGHT RATIONALE ───────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Signal Weight Rationale",
                           styles["section_title"]))
    story.append(divider())

    weight_data = [
        ["Signal", "Weight", "Rationale"],
        ["Research Momentum", "25%",
         "Recent papers indicate active, growing research areas"],
        ["Citation Density",  "20%",
         "High citations confirm domain importance and visibility"],
        ["Keyword Novelty",   "20%",
         "Emerging keywords signal underexplored technical directions"],
        ["Isolation Score",   "20%",
         "Graph-isolated papers represent bridge concepts with high gap potential"],
        ["Patent Gap Score",  "15%",
         "Low patent density indicates open commercialization opportunity"],
    ]

    wt = Table(weight_data, colWidths=[4.5*cm, 2*cm, 10.5*cm])
    wt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), NIIE_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), NIIE_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), NIIE_TEXT),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [NIIE_WHITE, NIIE_SURFACE]),
        ("GRID",          (0,0), (-1,-1), 0.3, NIIE_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("TEXTCOLOR",     (1,1), (1,-1), NIIE_ACCENT),
        ("FONTNAME",      (1,1), (1,-1), "Helvetica-Bold"),
    ]))
    story.append(wt)
    story.append(Spacer(1, 16))

    # ── CLOSING ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(divider(NIIE_ACCENT, thickness=1))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This report was generated automatically by NIIE — National Innovation "
        "Intelligence Engine. For research, patent, or commercial inquiries "
        "regarding the identified innovation opportunities, contact the NIIE team.",
        styles["muted"]
    ))

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage=page_tmpl.on_page,
        onLaterPages=page_tmpl.on_page
    )

    return output_path


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("data/processed/bci_scores.json","r",encoding="utf-8") as f:
        scores_data = json.load(f)
    with open("data/processed/bci_graph.json","r",encoding="utf-8") as f:
        graph_data = json.load(f)
    with open("data/processed/bci_innovations.json","r",encoding="utf-8") as f:
        innovations = json.load(f)

    path = generate_report(scores_data, graph_data, innovations)
    print(f"Report generated: {path}")