"""Legacy scraper wrapper forwarding to modernized src.scraper module.

Maintains backward-compatibility for existing scripts and notebooks.
"""

import sys
import os

# Add parent directory to sys.path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scraper import (
    scrape_listing,
    scrape_post_body as scrape_body,
    scrape_reddit_dataset as scrape_reddit_full,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Legacy Reddit scraper entrypoint")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    parser.add_argument("--output", type=str, default="data/reddit_datascience.csv", help="Output path")
    args = parser.parse_args()

    df = scrape_reddit_full(pages=args.pages, fallback_csv=args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
