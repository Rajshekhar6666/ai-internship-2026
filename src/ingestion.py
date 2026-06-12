import os
import json
import time
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Settings
QUERY = "brain-computer interface"
OUTPUT_FILE = "data/raw/papers/bci_papers.json"

def fetch_papers(query, api_key):
    print(f"Fetching papers for: '{query}'")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    headers = {
        "x-api-key": api_key
    }

    params = {
        "query": query,
        "limit": 100,
        "fields": "title,abstract,year,authors,citationCount,externalIds"
    }

    # Retry up to 5 times
    for attempt in range(1, 6):
        print(f"Attempt {attempt}...")
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            print(f"Successfully fetched {len(papers)} papers")
            return papers

        elif response.status_code == 429:
            wait_time = 15 * attempt  # 15s, 30s, 45s, 60s, 75s
            print(f"Rate limited. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        else:
            print(f"Unexpected error: {response.status_code}")
            print(response.text)
            return []

    print("Failed after 5 attempts.")
    return []

def save_papers(papers, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(papers)} papers to {output_file}")

def fetch_patents(query, output_file="data/raw/patents/bci_patents.json"):
    """Fetch patent data from Google Patents via SerpAPI free alternative."""
    print(f"\nFetching patents for: '{query}'")

    # Using patents.google.com public search via requests
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # We use Semantic Scholar with patent-specific terms as proxy
    # This is the free approach — no patent API key needed
    patent_queries = [
        f"{query} patent system method",
        f"{query} apparatus device invention",
        f"{query} neural interface patent"
    ]

    all_patents = []

    headers = {"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")}

    for pq in patent_queries:
        params = {
            "query" : pq,
            "limit" : 20,
            "fields": "title,abstract,year,authors,citationCount,externalIds"
        }
        for attempt in range(1, 4):
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json().get("data", [])
                all_patents.extend(data)
                print(f"  Got {len(data)} results for: '{pq[:40]}'")
                time.sleep(2)
                break
            elif response.status_code == 429:
                print(f"  Rate limited, waiting {15 * attempt}s...")
                time.sleep(15 * attempt)

    # Deduplicate by title
    seen   = set()
    unique = []
    for p in all_patents:
        t = p.get("title", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(p)

    print(f"Total unique patent-adjacent papers: {len(unique)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")
    return unique
def main():
    if not API_KEY:
        print("ERROR: API key not found. Check your .env file.")
        return

    print("API key loaded successfully")

    papers = fetch_papers(QUERY, API_KEY)

    if not papers:
        print("No papers fetched. Something went wrong.")
        return

    save_papers(papers, OUTPUT_FILE)

    # Preview first paper
    print("\n--- Preview of first paper ---")
    first = papers[0]
    print(f"Title   : {first.get('title')}")
    print(f"Year    : {first.get('year')}")
    print(f"Citations: {first.get('citationCount')}")
    abstract = first.get('abstract') or 'N/A'
    print(f"Abstract: {abstract[:200]}...")

if __name__ == "__main__":
    main()
    print("\n--- Now fetching patent data ---")
    fetch_patents("brain-computer interface")
    