import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"


def generate_section(client, section_name, prompt):
    """Generate a single paper section."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic writer specializing in "
                    "IEEE-format research papers. Write in formal academic "
                    "English. Be specific, cite concepts precisely, use "
                    "passive voice where appropriate. Never use bullet points "
                    "in academic prose — write in paragraphs only."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


def generate_paper(paper_data, progress_callback=None):
    """
    Generate a complete IEEE-format research paper.
    progress_callback(section_name, content) called after each section.
    """
    client = Groq(api_key=GROQ_API_KEY)

    title        = paper_data.get("title", "")
    authors      = paper_data.get("authors", "")
    domain       = paper_data.get("domain", "")
    problem      = paper_data.get("problem", "")
    solution     = paper_data.get("solution", "")
    methodology  = paper_data.get("methodology", "")
    results      = paper_data.get("results", "")
    keywords     = paper_data.get("keywords", "")
    gaps_found   = paper_data.get("gaps_found", "")
    top_score    = paper_data.get("top_score", "")
    num_papers   = paper_data.get("num_papers", 100)
    num_clusters = paper_data.get("num_clusters", 12)
    num_gaps     = paper_data.get("num_gaps", 20)

    sections = {}

    section_prompts = {
        "abstract": f"""Write a 150-word IEEE-format abstract.
Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}
Key Results: {results}
Structure: background → problem → method → results → significance.
Past tense. No citations. No bullet points. Paragraphs only.""",

        "introduction": f"""Write a 300-word IEEE-format Introduction.
Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}
Paragraph 1: Why does this domain matter?
Paragraph 2: The specific problem or gap.
Paragraph 3: Our proposed approach and novelty.
Paragraph 4: Main contributions as (1)... (2)... (3)...
Paragraph 5: Paper organization.
Include citation placeholders [1],[2],[3]. Academic English. Paragraphs only.""",

        "related_work": f"""Write a 300-word IEEE-format Related Work section.
Domain: {domain}
Our approach: {solution}
Subsection A: Existing research discovery tools
Subsection B: Knowledge graphs in research analysis
Subsection C: AI-based innovation scoring
Subsection D: Limitations of Elicit, ResearchRabbit, Connected Papers, Semantic Scholar
Include [4],[5],[6],[7],[8]. Academic English. Paragraphs only.""",

        "methodology": f"""Write a 350-word IEEE-format Methodology section.
System: {solution}
Approach: {methodology}
Subsection A: Data Acquisition — Semantic Scholar, {num_papers} papers
Subsection B: Entity Extraction — 30 BCI keywords
Subsection C: Knowledge Graph — cosine similarity 0.65, NetworkX
Subsection D: Innovation Scoring — 5 signals, formula: IOS = 0.25*M + 0.20*C + 0.20*N + 0.20*I + 0.15*P
Subsection E: AI Generation — Llama 3.3-70B via Groq
Academic English. Paragraphs only.""",

        "results": f"""Write a 250-word IEEE-format Results section.
Papers: {num_papers}, Clusters: {num_clusters}, Gaps: {num_gaps}, Top score: {top_score}
Top gaps: {gaps_found}
Subsection A: Knowledge Graph Analysis
Subsection B: Innovation Scoring Results
Subsection C: Research Gap Detection
Reference specific numbers. Academic English. Paragraphs only.""",

        "discussion": f"""Write a 200-word IEEE-format Discussion section.
Domain: {domain}, Gaps: {num_gaps}, Top score: {top_score}
Paragraph 1: What the gaps mean for the field
Paragraph 2: How NIIE compares to Elicit, ResearchRabbit, Connected Papers
Paragraph 3: Limitations and how to address them
Paragraph 4: Broader implications
Academic English. Analytical tone. Paragraphs only.""",

        "conclusion": f"""Write a 150-word IEEE-format Conclusion.
Title: {title}
Results: {num_papers} papers, {num_gaps} gaps, top score {top_score}
Paragraph 1: Summary of contributions
Paragraph 2: Most significant finding
Paragraph 3: Future work — real-time streaming, cross-domain detection, automated patents
Past tense for completed work. Future tense for future work.""",

        "references": f"""Generate 12 IEEE-format references for a paper about:
- AI research gap detection and knowledge graphs
- Innovation opportunity scoring
- {domain}
- Large language models for research
Format: [1] A. Author, "Title," Journal, vol. X, pp. XX, Year.
Mix of IEEE journals, NeurIPS, AAAI, ICML. Years 2019-2025. Realistic.""",
    }

    section_order = [
        "abstract","introduction","related_work",
        "methodology","results","discussion",
        "conclusion","references"
    ]

    for key in section_order:
        try:
            content = generate_section(client, key, section_prompts[key])
            sections[key] = content
            if progress_callback:
                progress_callback(key, content)
        except Exception as e:
            sections[key] = f"[Error generating {key}: {e}]"
            if progress_callback:
                progress_callback(key, sections[key])

    return sections

    # ── ABSTRACT ─────────────────────────────────────────────────────────────
    sections["abstract"] = generate_section(client, "Abstract", f"""
Write a 150-word IEEE-format abstract for this research paper.

Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}
Key Results: {results}

Structure: 1 sentence on background/motivation, 1-2 sentences on the
problem, 2-3 sentences on the proposed method, 1-2 sentences on
experimental results, 1 sentence on significance/contribution.
Use past tense. No citations. No bullet points.
""")

    # ── INTRODUCTION ─────────────────────────────────────────────────────────
    sections["introduction"] = generate_section(
        client, "Introduction", f"""
Write a 300-word IEEE-format Introduction section for this paper.

Title: {title}
Domain: {domain}
Problem: {problem}
Solution: {solution}

Structure:
Paragraph 1: Motivate the problem — why does this domain matter?
Paragraph 2: State the specific problem or gap this paper addresses.
Paragraph 3: Briefly describe the proposed approach and its novelty.
Paragraph 4: State the key contributions as numbered points
             (e.g., "The main contributions of this paper are:
             (1)... (2)... (3)...").
Paragraph 5: Describe the organization of the paper.

Use formal academic English. Include plausible citation placeholders
like [1], [2], [3].
""")

    # ── RELATED WORK ─────────────────────────────────────────────────────────
    sections["related_work"] = generate_section(
        client, "Related Work", f"""
Write a 300-word IEEE-format Related Work section for this paper.

Domain: {domain}
Our approach: {solution}

Cover these subsections:
A. Existing approaches to {domain} research discovery
B. Knowledge graph applications in research analysis
C. AI-based innovation scoring and gap detection
D. Limitations of existing tools (mention Elicit, ResearchRabbit,
   Connected Papers, Semantic Scholar) and how our work differs.

Use formal academic English. Include plausible citation placeholders
like [4], [5], [6], [7], [8].
Write in paragraphs, not bullet points.
""")

    # ── METHODOLOGY ──────────────────────────────────────────────────────────
    sections["methodology"] = generate_section(
        client, "Methodology", f"""
Write a 400-word IEEE-format Methodology section for this paper.

System: {solution}
Approach: {methodology}

Cover these subsections with IEEE headings (A, B, C, D, E):
A. Data Acquisition — Semantic Scholar API, {num_papers} papers,
   domain-specific queries
B. Entity Extraction — keyword taxonomy, 30 BCI terms,
   pattern matching
C. Knowledge Graph Construction — cosine similarity threshold 0.65,
   NetworkX, degree and betweenness centrality
D. Innovation Opportunity Scoring Engine — 5 signals
   (Research Momentum 25%, Citation Density 20%,
   Keyword Novelty 20%, Isolation Score 20%, Patent Gap 15%),
   weighted linear combination formula
E. AI Proposal Generation — Llama 3.3-70B via Groq API,
   structured prompting, 6-section output format

Include the scoring formula:
IOS = 0.25*M + 0.20*C + 0.20*N + 0.20*I + 0.15*P

Use formal academic English. Write in paragraphs.
""")

    # ── RESULTS ──────────────────────────────────────────────────────────────
    sections["results"] = generate_section(
        client, "Results", f"""
Write a 300-word IEEE-format Results and Evaluation section.

Experimental results:
- Papers analysed: {num_papers}
- Knowledge graph: {num_papers} nodes, edges constructed at
  similarity threshold 0.65
- Research clusters identified: {num_clusters}
- Innovation gaps detected: {num_gaps}
- Top innovation opportunity score: {top_score}
- Top identified gaps: {gaps_found}
- AI proposals generated with patent claims and impact scores

User-provided results: {results}

Cover:
A. Knowledge Graph Analysis — cluster distribution,
   connectivity metrics
B. Innovation Scoring Results — score distribution,
   top opportunities
C. Research Gap Detection — gap characteristics,
   signal breakdown
D. AI Proposal Quality — examples of generated proposals,
   patent claims

Use formal academic English. Reference specific numbers.
""")

    # ── DISCUSSION ───────────────────────────────────────────────────────────
    sections["discussion"] = generate_section(
        client, "Discussion", f"""
Write a 250-word IEEE-format Discussion section.

Domain: {domain}
Key findings: {num_gaps} gaps detected, top score {top_score}
Solution: {solution}

Cover:
Paragraph 1: Interpret the key findings — what do the identified
             gaps mean for the field?
Paragraph 2: Compare to existing tools (Elicit, ResearchRabbit,
             Connected Papers) — what does NIIE do that they cannot?
Paragraph 3: Limitations of the current approach and how they
             can be addressed.
Paragraph 4: Broader implications for research methodology,
             innovation policy, and patent strategy.

Use formal academic English. Be analytical, not descriptive.
""")

    # ── CONCLUSION ───────────────────────────────────────────────────────────
    sections["conclusion"] = generate_section(
        client, "Conclusion", f"""
Write a 150-word IEEE-format Conclusion section.

Title: {title}
Key contributions: {solution}
Results: {num_papers} papers, {num_gaps} gaps,
         top score {top_score}

Paragraph 1: Summarize the paper and its key contributions.
Paragraph 2: State the most significant finding.
Paragraph 3: Describe future work directions
             (real-time streaming, multi-modal data,
             cross-domain collision detection,
             automated patent filing).

Use past tense for completed work, future tense for future work.
""")

    # ── REFERENCES ───────────────────────────────────────────────────────────
    sections["references"] = generate_section(
        client, "References", f"""
Generate 15 plausible IEEE-format references for a paper about:
- AI-powered research gap detection
- Knowledge graphs for scientific literature
- Innovation opportunity scoring
- Brain-computer interfaces (if domain is BCI)
- Large language models for research assistance
- Patent analysis using AI

Domain: {domain}

Format each as IEEE reference style:
[1] A. Author, "Title," Journal/Conference, vol. X, pp. XX-XX, Year.

Include a mix of:
- Journal papers (IEEE, Nature, Science)
- Conference papers (NeurIPS, AAAI, ICML, IEEE conferences)
- Books
- Recent papers (2020-2025)

Make them realistic and plausible.
""")

    return sections


def format_paper_latex(paper_data, sections):
    """Format the paper as LaTeX for Overleaf."""
    title    = paper_data.get("title", "NIIE Research Paper")
    authors  = paper_data.get("authors", "Author Name")
    keywords = paper_data.get("keywords", "")

    latex = f"""\\documentclass[conference]{{IEEEtran}}
\\IEEEoverridecommandlockouts

\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}

\\begin{{document}}

\\title{{{title}}}

\\author{{\\IEEEauthorblockN{{{authors}}}}}

\\maketitle

\\begin{{abstract}}
{sections.get('abstract', '')}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{keywords}
\\end{{IEEEkeywords}}

\\section{{Introduction}}
{sections.get('introduction', '')}

\\section{{Related Work}}
{sections.get('related_work', '')}

\\section{{Methodology}}
{sections.get('methodology', '')}

\\section{{Results and Evaluation}}
{sections.get('results', '')}

\\section{{Discussion}}
{sections.get('discussion', '')}

\\section{{Conclusion}}
{sections.get('conclusion', '')}

\\begin{{thebibliography}}{{00}}
{sections.get('references', '')}
\\end{{thebibliography}}

\\end{{document}}
"""
    return latex


def format_paper_markdown(paper_data, sections):
    """Format the paper as clean Markdown."""
    title   = paper_data.get("title", "")
    authors = paper_data.get("authors", "")
    domain  = paper_data.get("domain", "")

    md = f"""# {title}

**Authors:** {authors}
**Domain:** {domain}

---

## Abstract

{sections.get('abstract', '')}

---

## 1. Introduction

{sections.get('introduction', '')}

---

## 2. Related Work

{sections.get('related_work', '')}

---

## 3. Methodology

{sections.get('methodology', '')}

---

## 4. Results and Evaluation

{sections.get('results', '')}

---

## 5. Discussion

{sections.get('discussion', '')}

---

## 6. Conclusion

{sections.get('conclusion', '')}

---

## References

{sections.get('references', '')}
"""
    return md