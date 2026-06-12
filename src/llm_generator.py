import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File paths
SCORES_FILE = "data/processed/bci_scores.json"
OUTPUT_FILE = "data/processed/bci_innovations.json"

# Model to use — completely free on Groq
MODEL = "llama-3.3-70b-versatile"

def load_gaps():
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["research_gaps"]

def build_prompt(gap):
    title    = gap["title"]
    abstract = gap.get("abstract", "") or "No abstract available."
    keywords = ", ".join(gap["bci_keywords"])
    year     = gap.get("year", "Unknown")
    citations = gap.get("citations", 0)

    prompt = f"""You are an expert AI innovation analyst specializing in Brain-Computer Interface (BCI) research.

A research paper has been identified as an UNDEREXPLORED area in BCI science — meaning it has low citation density, high novelty keywords, and is isolated in the research knowledge graph. This makes it a strong candidate for future innovation.

Paper Details:
- Title: {title}
- Year: {year}
- Citations: {citations}
- Keywords: {keywords}
- Abstract: {abstract[:500]}

Based on this underexplored research area, generate a structured innovation proposal with exactly these sections:

1. INNOVATION TITLE
A compelling name for the proposed innovation (not the paper title).

2. PROBLEM IT SOLVES
What specific real-world problem does this innovation address? Be concrete.

3. PROPOSED SOLUTION
What should be built? Describe the system, product, or technology in 3-4 sentences.

4. WHY IT IS NOVEL
What makes this different from existing solutions? What gap does it fill?

5. TECHNOLOGY STACK
List 5-7 specific technologies needed to build this.

6. TARGET USERS
Who benefits from this innovation? Be specific.

7. PATENT OPPORTUNITY
Describe one specific patentable claim for this innovation in one sentence.

8. RESEARCH CONTRIBUTION
What would be the key academic contribution if this were published?

9. IMPACT SCORE
Rate the potential impact from 1-10 and explain why in one sentence.

Be specific, technical, and forward-thinking. Avoid generic statements."""

    return prompt

def generate_innovation(client, gap):
    prompt = build_prompt(gap)

    print(f"  Generating proposal for: {gap['title'][:60]}...")

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role"   : "system",
                "content": "You are an expert AI innovation analyst. Always respond with structured, technical, specific innovation proposals. Never give vague or generic answers."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature = 0.7,
        max_tokens  = 1024
    )

    return response.choices[0].message.content

def parse_proposal(raw_text):
    """Extract sections from the generated text."""
    sections = {
        "innovation_title"      : "",
        "problem_it_solves"     : "",
        "proposed_solution"     : "",
        "why_novel"             : "",
        "technology_stack"      : "",
        "target_users"          : "",
        "patent_opportunity"    : "",
        "research_contribution" : "",
        "impact_score"          : "",
        "raw"                   : raw_text
    }

    # Map of possible section headers to keys
    section_map = {
        "INNOVATION TITLE"      : "innovation_title",
        "PROBLEM IT SOLVES"     : "problem_it_solves",
        "PROPOSED SOLUTION"     : "proposed_solution",
        "WHY IT IS NOVEL"       : "why_novel",
        "TECHNOLOGY STACK"      : "technology_stack",
        "TARGET USERS"          : "target_users",
        "PATENT OPPORTUNITY"    : "patent_opportunity",
        "RESEARCH CONTRIBUTION" : "research_contribution",
        "IMPACT SCORE"          : "impact_score"
    }

    lines = raw_text.split("\n")
    current_section = None
    buffer = []

    for line in lines:
        line_upper = line.upper().strip()
        matched = False

        for key, val in section_map.items():
            if key in line_upper:
                # Save previous section
                if current_section and buffer:
                    sections[current_section] = " ".join(buffer).strip()
                current_section = val
                buffer = []
                # Check if content is on same line after the header
                if ":" in line:
                    inline = line.split(":", 1)[-1].strip()
                    if inline and len(inline) > 2:
                        buffer.append(inline)
                matched = True
                break

        if not matched and current_section:
            stripped = line.strip()
            # Skip lines that are just numbers or dashes
            if stripped and stripped not in ["-", "—", "*", "**"]:
                buffer.append(stripped)

    # Save last section
    if current_section and buffer:
        sections[current_section] = " ".join(buffer).strip()

    return sections
def generate_all(gaps, max_gaps=5):
    """Generate innovation proposals for top N gaps."""
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in .env file")
        return []

    client = Groq(api_key=GROQ_API_KEY)
    results = []

    print(f"Generating innovation proposals for top {max_gaps} research gaps...\n")

    for i, gap in enumerate(gaps[:max_gaps]):
        print(f"[{i+1}/{max_gaps}]", end=" ")
        try:
            raw      = generate_innovation(client, gap)
            proposal = parse_proposal(raw)
            results.append({
                "gap_id"          : gap["id"],
                "gap_title"       : gap["title"],
                "gap_score"       : gap["scores"]["innovation_opportunity_score"],
                "gap_year"        : gap["year"],
                "gap_citations"   : gap["citations"],
                "gap_keywords"    : gap["bci_keywords"],
                "proposal"        : proposal
            })
            print("Done")
        except Exception as e:
            print(f"Error: {e}")
            continue

    return results

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} innovation proposals to {output_file}")

def print_summary(results):
    print("\n" + "="*70)
    print("INNOVATION PROPOSALS GENERATED")
    print("="*70)

    for i, r in enumerate(results, 1):
        p = r["proposal"]
        print(f"\n{'─'*70}")
        print(f"GAP #{i} — Score: {r['gap_score']}")
        print(f"Research Paper : {r['gap_title'][:65]}...")
        print(f"{'─'*70}")
        print(f"INNOVATION     : {p.get('innovation_title', 'N/A')}")
        print(f"PROBLEM        : {p.get('problem_it_solves', 'N/A')[:120]}...")
        print(f"PATENT ANGLE   : {p.get('patent_opportunity', 'N/A')[:120]}...")
        print(f"IMPACT SCORE   : {p.get('impact_score', 'N/A')[:80]}")

def main():
    print("Starting LLM Innovation Generator...\n")

    gaps = load_gaps()
    print(f"Loaded {len(gaps)} research gaps from scoring engine")

    results = generate_all(gaps, max_gaps=5)

    if not results:
        print("No proposals generated. Check your API key.")
        return

    save_results(results, OUTPUT_FILE)
    print_summary(results)

    print("\nInnovation generation complete.")

if __name__ == "__main__":
    main()