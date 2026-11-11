"""Legacy helper compatibility package.

Maintains backward-compatibility with existing notebooks and scripts.
"""

from helper.reddit_scraper import (
    scrape_listing,
    scrape_body,
    scrape_reddit_full,
)

__all__ = [
    "scrape_listing",
    "scrape_body",
    "scrape_reddit_full",
]
