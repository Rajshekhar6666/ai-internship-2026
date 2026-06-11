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