import os
import json
import time
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GROQ_API_KEY             = os.getenv("GROQ_API_KEY")
GROQ_MODEL               = "llama-3.3-70b-versatile"


def search_papers(query, limit=20):
    """Search Semantic Scholar for papers matching a natural language query."""
    url     = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params  = {
        "query" : query,
        "limit" : limit,
        "fields": "title,abstract,year,authors,citationCount,"
                  "externalIds,tldr,publicationTypes"
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


def summarize_paper(paper, groq_client):
    """Generate a concise AI summary of a paper."""
    title    = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    tldr     = paper.get("tldr", {})
    tldr_text = tldr.get("text", "") if tldr else ""

    if not abstract and not tldr_text:
        return "No abstract available for summarization."

    prompt = f"""Summarize this research paper in exactly 3 sentences.
Sentence 1: What problem does it solve?
Sentence 2: What method or approach did they use?
Sentence 3: What was the key finding or contribution?

Title: {title}
Abstract: {abstract[:600]}
{f'TLDR: {tldr_text}' if tldr_text else ''}

Be specific and technical. No filler phrases."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a research paper summarizer. "
                            "Be concise, specific, and technical."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {e}"


def generate_landscape(query, papers, groq_client):
    """Generate an overall research landscape summary."""
    titles = "\n".join([
        f"- {p.get('title','')[:80]}" for p in papers[:10]
    ])

    prompt = f"""You are a research analyst. Based on these papers about "{query}",
write a Research Landscape Overview in exactly 4 paragraphs:

Paragraph 1: Current state of this research field (2-3 sentences)
Paragraph 2: Major themes and approaches being used (2-3 sentences)
Paragraph 3: Key challenges and open problems (2-3 sentences)
Paragraph 4: Most promising future directions (2-3 sentences)

Papers found:
{titles}

Be specific, technical, and insightful."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are an expert research analyst with deep "
                            "knowledge across all scientific domains."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Landscape overview unavailable: {e}"


def run_literature_search(query, summarize_top=5,
                           status_callback=None):
    def update(msg):
        if status_callback:
            status_callback(msg)

    groq_client = Groq(api_key=GROQ_API_KEY)

    update("📡 Searching Semantic Scholar...")
    papers = search_papers(query, limit=20)
    if not papers:
        return None, "No papers found. Try a different query."
    update(f"✅ Found {len(papers)} papers")

    # Sort by citation count
    papers = sorted(
        papers,
        key=lambda x: x.get("citationCount", 0) or 0,
        reverse=True
    )

    update("🌐 Generating research landscape overview...")
    landscape = generate_landscape(query, papers, groq_client)
    update("✅ Landscape generated")

    update(f"🤖 Summarizing top {summarize_top} papers...")
    for i, paper in enumerate(papers[:summarize_top]):
        paper["ai_summary"] = summarize_paper(paper, groq_client)
        update(f"  Summarized {i+1}/{summarize_top}: "
               f"{paper.get('title','')[:40]}...")

    update("🎉 Literature search complete!")

    return {
        "query"    : query,
        "papers"   : papers,
        "landscape": landscape,
        "total"    : len(papers)
    }, None