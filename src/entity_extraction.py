import os
import json
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# File paths
INPUT_FILE  = "data/raw/papers/bci_papers.json"
OUTPUT_FILE = "data/processed/bci_entities.json"

# Keywords specific to BCI domain we care about
BCI_KEYWORDS = [
    "neural", "electrode", "eeg", "ecog", "meg", "fmri", "bci",
    "brain", "cortex", "neuron", "signal", "implant", "prosthetic",
    "motor", "cognitive", "attention", "seizure", "stroke", "paralysis",
    "deep learning", "machine learning", "classification", "decoding",
    "wireless", "invasive", "non-invasive", "feedback", "rehabilitation"
]

def extract_entities(text):
    """Extract named entities and BCI keywords from text."""
    if not text:
        return [], []

    doc = nlp(text[:1000000])  # spaCy limit safety

    # Named entities (organizations, people, places, etc.)
    named_entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "WORK_OF_ART", "LAW"]:
            named_entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    # BCI domain keywords found in text
    text_lower = text.lower()
    found_keywords = [kw for kw in BCI_KEYWORDS if kw in text_lower]

    return named_entities, found_keywords


def process_papers(input_file, output_file):
    # Load papers
    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Processing {len(papers)} papers...")

    processed = []

    for i, paper in enumerate(papers):
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")
        year     = paper.get("year")
        citations = paper.get("citationCount", 0)

        # Combine title and abstract for extraction
        full_text = f"{title}. {abstract}"

        named_entities, keywords = extract_entities(full_text)

        processed.append({
            "title"         : title,
            "year"          : year,
            "citations"     : citations,
            "abstract"      : abstract,
            "named_entities": named_entities,
            "bci_keywords"  : keywords,
            "keyword_count" : len(keywords)
        })

        # Progress update every 20 papers
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(papers)} papers")

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\nSaved processed data to {output_file}")
    return processed


def print_summary(processed):
    print("\n--- Extraction Summary ---")

    # Most common keywords across all papers
    from collections import Counter
    all_keywords = []
    for p in processed:
        all_keywords.extend(p["bci_keywords"])

    keyword_counts = Counter(all_keywords)
    print("\nTop 10 BCI keywords found:")
    for kw, count in keyword_counts.most_common(10):
        print(f"  {kw:<25} → {count} papers")

    # Papers with most keywords (most domain-relevant)
    print("\nTop 5 most domain-relevant papers:")
    sorted_papers = sorted(processed, key=lambda x: x["keyword_count"], reverse=True)
    for p in sorted_papers[:5]:
        print(f"  [{p['keyword_count']} keywords] {p['title'][:70]}...")


def main():
    print("Starting entity extraction...\n")
    processed = process_papers(INPUT_FILE, OUTPUT_FILE)
    print_summary(processed)
    print("\nEntity extraction complete.")


if __name__ == "__main__":
    main()