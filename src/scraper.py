"""Resilient scraper for r/datascience on Reddit.

Features:
- Extracts post metadata: title, author, score, comments, timestamp, flair, domain, and body
- Handles pagination and skips promoted ads
- Implements exponential backoff and polite delay intervals
- Provides graceful fallback to cached local CSV when offline or rate-limited
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 (Educational Research Bot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

BASE_URL = "https://old.reddit.com/r/datascience/"


def scrape_listing(
    url: str = BASE_URL,
    pages: int = 5,
    headers: Optional[Dict[str, str]] = None,
    delay_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """Scrapes post listings across multiple pages on old.reddit.com.

    Args:
        url: Starting listing URL.
        pages: Number of pages to paginate through.
        headers: Request headers.
        delay_seconds: Polite pause between requests.

    Returns:
        List of dictionaries with post metadata.
    """
    headers = headers or DEFAULT_HEADERS
    current_url = url
    posts = []

    for page_idx in range(pages):
        logger.info("Scraping listing page %d/%d: %s", page_idx + 1, pages, current_url)
        try:
            response = requests.get(current_url, headers=headers, timeout=12)
            if response.status_code == 429:
                logger.warning("Rate-limited (HTTP 429). Backing off for 10s...")
                time.sleep(10)
                response = requests.get(current_url, headers=headers, timeout=12)
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch page %s: %s", current_url, exc)
            break

        soup = BeautifulSoup(response.text, "html.parser")
        things = soup.find_all("div", class_="thing")

        if not things:
            logger.warning("No post elements found on page %d.", page_idx + 1)
            break

        for post in things:
            # Skip promoted / sponsored links
            if "promoted" in post.get("class", []):
                continue

            title_tag = post.find("a", class_="title")
            if not title_tag:
                continue

            post_url = title_tag.get("href", "")
            if post_url.startswith("/"):
                post_url = "https://old.reddit.com" + post_url

            if "alb.reddit.com" in post_url:
                continue

            # Extract comment count
            comments_tag = post.find("a", class_="comments")
            comment_count = 0
            if comments_tag:
                parts = comments_tag.text.strip().split()
                if parts and parts[0].isdigit():
                    comment_count = int(parts[0])

            # Extract timestamp
            time_tag = post.find("time")
            created_utc = time_tag["datetime"] if time_tag and time_tag.has_attr("datetime") else None

            # Extract post flair
            flair_tag = post.find("span", class_="linkflairlabel")
            flair = flair_tag.text.strip() if flair_tag else None

            # Post domain & type
            domain = post.get("data-domain", "")
            is_self = post.get("data-type") == "self" or domain.startswith("self.")

            posts.append({
                "post_id": post.get("data-fullname"),
                "post_title": title_tag.text.strip(),
                "author": post.get("data-author"),
                "score": int(post.get("data-score") or 0),
                "comment_count": comment_count,
                "created_utc": created_utc,
                "url": post_url,
                "flair": flair,
                "domain": domain,
                "is_self": is_self,
            })

        next_btn = soup.find("span", class_="next-button")
        if not next_btn or not next_btn.find("a"):
            logger.info("No next page button found; reached end of listing.")
            break

        current_url = next_btn.find("a")["href"]
        time.sleep(delay_seconds)

    logger.info("Scraped %d posts metadata across pages.", len(posts))
    return posts


def scrape_post_body(post_url: str, headers: Optional[Dict[str, str]] = None) -> str:
    """Fetches self-text body for a single post."""
    if not post_url.startswith("https://old.reddit.com/r/datascience/"):
        return ""

    headers = headers or DEFAULT_HEADERS
    try:
        resp = requests.get(post_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        post_div = soup.find("div", class_="thing", attrs={"data-type": "link"}) or soup.find("div", class_="thing")
        if not post_div:
            return ""
        body_div = post_div.find("div", class_="md")
        return body_div.get_text(separator=" ", strip=True) if body_div else ""
    except Exception as exc:
        logger.debug("Could not scrape body for %s: %s", post_url, exc)
        return ""


def scrape_reddit_dataset(
    pages: int = 5,
    fetch_bodies: bool = True,
    fallback_csv: str = "data/reddit_datascience.csv",
) -> pd.DataFrame:
    """Full scraping workflow with automatic fallback to cached CSV if offline.

    Args:
        pages: Number of pages to scrape.
        fetch_bodies: Whether to fetch full post bodies.
        fallback_csv: Local CSV path to return if network request fails.

    Returns:
        DataFrame containing Reddit post data.
    """
    logger.info("Attempting live scrape of r/datascience (%d pages)...", pages)
    try:
        listing = scrape_listing(pages=pages)
        if not listing:
            raise RuntimeError("Scraper received 0 posts from listing.")

        if fetch_bodies:
            for idx, item in enumerate(listing):
                if item.get("is_self", True):
                    item["body"] = scrape_post_body(item["url"])
                    time.sleep(0.5)
                else:
                    item["body"] = ""

        df = pd.DataFrame(listing)
        logger.info("Live scrape successful: collected %d records.", len(df))
        return df

    except Exception as exc:
        logger.warning("Live scraping failed or network offline (%s). Falling back to cached dataset: %s", exc, fallback_csv)
        if os.path.exists(fallback_csv):
            df_fallback = pd.read_csv(fallback_csv)
            logger.info("Loaded %d records from fallback cached dataset: %s", len(df_fallback), fallback_csv)
            return df_fallback
        raise FileNotFoundError(f"Neither live scraping nor fallback file {fallback_csv} succeeded.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape r/datascience posts.")
    parser.add_argument("--pages", type=int, default=5, help="Number of listing pages to scrape")
    parser.add_argument("--output", type=str, default="data/reddit_datascience.csv", help="Output CSV path")
    args = parser.parse_args()

    data = scrape_reddit_dataset(pages=args.pages, fallback_csv=args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Data saved to {args.output} ({len(data)} rows)")

