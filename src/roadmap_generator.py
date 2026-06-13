import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"


def generate_roadmap(gap, domain="Brain-Computer Interfaces"):
    """Generate a complete 6-step research execution roadmap for a gap."""
    client = Groq(api_key=GROQ_API_KEY)

    title    = gap.get("title", "")
    abstract = (gap.get("abstract", "") or "")[:500]
    keywords = ", ".join(gap.get("bci_keywords", []))
    score    = gap.get("scores", {}).get(
        "innovation_opportunity_score", 0
    )
    year     = gap.get("year", "N/A")
    citations = gap.get("citations", 0)

    prompt = f"""You are a world-class research strategist and innovation advisor.

A research gap has been identified in the domain of "{domain}".

Gap Details:
- Paper Title: {title}
- Year: {year}
- Citations: {citations}
- Innovation Score: {score:.4f}
- Keywords: {keywords}
- Abstract: {abstract}

Generate a complete, actionable 6-step Research Execution Roadmap.
Be extremely specific — no generic advice.

Format your response EXACTLY like this:

STEP 1: HYPOTHESIS
[Write one specific, testable research hypothesis that extends or challenges this paper. Include the independent variable, dependent variable, and expected relationship.]

STEP 2: METHODOLOGY
[Specify the exact research methodology: experimental design, control conditions, sample size, measurement instruments, and statistical tests to use.]

STEP 3: DATASET
[Name specific datasets, databases, or data collection methods needed. Include exact sources, how to access them, and estimated data size.]

STEP 4: EXPECTED RESULTS
[Describe what results would confirm the hypothesis. Include specific metrics, thresholds, and what would constitute a significant finding.]

STEP 5: PUBLICATION STRATEGY
[Name 3 specific journals or conferences to target. Include impact factor if known, submission timeline, and which section of the paper to emphasize.]

STEP 6: PATENT STRATEGY
[Write the specific independent claim for a patent filing. Include what is novel, what it improves upon, and which patent class to file under.]

BONUS - RISK ASSESSMENT
[Identify the 2 biggest risks in this research plan and how to mitigate them.]"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class research strategist with expertise "
                    "in AI, neuroscience, and innovation management. "
                    "Your roadmaps are specific, actionable, and publishable-quality. "
                    "Never give generic advice. Always be technically precise."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()
    return parse_roadmap(raw)


def parse_roadmap(raw_text):
    """Parse the structured roadmap response into sections."""
    sections = {
        "hypothesis"   : "",
        "methodology"  : "",
        "dataset"      : "",
        "expected"     : "",
        "publication"  : "",
        "patent"       : "",
        "risk"         : "",
        "raw"          : raw_text
    }

    markers = {
        "STEP 1"  : "hypothesis",
        "STEP 2"  : "methodology",
        "STEP 3"  : "dataset",
        "STEP 4"  : "expected",
        "STEP 5"  : "publication",
        "STEP 6"  : "patent",
        "BONUS"   : "risk",
    }

    lines   = raw_text.split("\n")
    current = None
    buffer  = []

    for line in lines:
        line_upper = line.upper().strip()
        matched    = False

        for marker, key in markers.items():
            if line_upper.startswith(marker):
                if current and buffer:
                    sections[current] = " ".join(buffer).strip()
                current = key
                buffer  = []
                # Check for inline content
                if ":" in line:
                    inline = line.split(":", 1)[-1].strip()
                    if inline and len(inline) > 3:
                        buffer.append(inline)
                matched = True
                break

        if not matched and current:
            stripped = line.strip()
            if stripped and stripped not in ["-","—","*","**","["]:
                buffer.append(stripped)

    if current and buffer:
        sections[current] = " ".join(buffer).strip()

    return sections


def generate_roadmaps_for_gaps(gaps, domain, top_n=3):
    """Generate roadmaps for the top N gaps."""
    roadmaps = []
    for i, gap in enumerate(gaps[:top_n]):
        print(f"Generating roadmap {i+1}/{top_n}: "
              f"{gap['title'][:50]}...")
        roadmap = generate_roadmap(gap, domain)
        roadmaps.append({
            "gap_title"   : gap["title"],
            "gap_score"   : gap["scores"][
                "innovation_opportunity_score"
            ],
            "gap_year"    : gap.get("year"),
            "gap_keywords": gap.get("bci_keywords", []),
            "roadmap"     : roadmap
        })
    return roadmaps


if __name__ == "__main__":
    with open(
        "data/processed/bci_scores.json", "r", encoding="utf-8"
    ) as f:
        scores_data = json.load(f)

    gaps     = scores_data["research_gaps"][:3]
    roadmaps = generate_roadmaps_for_gaps(
        gaps, "Brain-Computer Interfaces"
    )

    for i, r in enumerate(roadmaps, 1):
        print(f"\n{'='*60}")
        print(f"ROADMAP #{i}: {r['gap_title'][:55]}...")
        print(f"{'='*60}")
        rm = r["roadmap"]
        print(f"HYPOTHESIS:  {rm['hypothesis'][:150]}...")
        print(f"METHODOLOGY: {rm['methodology'][:150]}...")
        print(f"PATENT:      {rm['patent'][:150]}...")