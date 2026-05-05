"""
Generate EA Forum top-post ingestion config via GraphQL API.

Usage:
  python fetch_ea_top_posts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests

GRAPHQL_URL = "https://forum.effectivealtruism.org/graphql"
OUT_PATH = Path(__file__).resolve().parent / "config" / "ea_top_posts.json"
NAMESPACE = "animal_policies"
TOP_N = 10

TOPICS: Dict[str, Dict[str, str]] = {
    "animal_welfare": {
        "tag_id": "QdH9f8TC6G8oGYdgt",
        "name": "Animal welfare",
        "primary_focus": "effective_advocacy",
    },
    "farmed_animal_welfare": {
        "tag_id": "TteDwtS2DckL4kLpT",
        "name": "Farmed animal welfare",
        "primary_focus": "farmed_animals",
    },
    "wild_animal_welfare": {
        "tag_id": "MWwcDEyw2ThZDReGF",
        "name": "Wild animal welfare",
        "primary_focus": "wildlife",
    },
    "effective_animal_advocacy": {
        "tag_id": "c8w7qjiHku7nDrZYp",
        "name": "Effective animal advocacy",
        "primary_focus": "effective_advocacy",
    },
}


def gql(query: str, variables: Dict) -> Dict:
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=30,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def fetch_top_posts(tag_id: str, limit: int = TOP_N) -> List[Dict[str, str]]:
    query = """
    query TopPosts($terms: JSON) {
      posts(input: { terms: $terms }) {
        results {
          _id
          title
          pageUrl
          postedAt
        }
      }
    }
    """
    terms = {"view": "top", "limit": limit, "tagId": tag_id}
    data = gql(query, {"terms": terms})
    return data["posts"]["results"]


def to_forum_bots_url(url: str) -> str:
    """
    Normalize EA post URLs to forum-bots domain for scraping.
    """
    parsed = urlparse(url)
    path = parsed.path or ""
    if path.startswith("/posts/"):
        return f"https://forum-bots.effectivealtruism.org{path}"
    return url


def build_config() -> List[Dict]:
    entries: List[Dict] = []
    for topic_key, topic in TOPICS.items():
        posts = fetch_top_posts(topic["tag_id"], TOP_N)
        for post in posts:
            post_url = post.get("pageUrl")
            post_title = post.get("title")
            posted_at = post.get("postedAt")
            if not post_url or not post_title:
                continue
            post_url = to_forum_bots_url(post_url)
            publication_date = posted_at[:10] if posted_at else "1970-01-01"

            entries.append(
                {
                    "type": "web",
                    "source": post_url,
                    "namespace": NAMESPACE,
                    "meta": {
                        "name": post_title,
                        "organization": "Effective Altruism Forum",
                        "primary_focus": topic["primary_focus"],
                        "doc_type": "forum",
                        "publication_date": publication_date,
                        "tags": [
                            "effective_altruism",
                            "animal_advocacy",
                            topic_key,
                            "top_posts",
                            "api_collected",
                        ],
                    },
                }
            )
    return entries


def main() -> None:
    config = build_config()
    OUT_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {len(config)} topic entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
