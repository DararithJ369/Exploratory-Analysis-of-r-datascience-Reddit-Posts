"""Reddit Data Science Scraping & Analysis Package.

Modular components for scraping, cleaning, NLP, topic modeling, and visualization.
"""

from src.cleaning import (
    clean_reddit_data,
    detect_iqr_outliers,
    normalize_text,
)
from src.nlp import (
    assign_nmf_topics,
    categorize_posts,
    categorize_title,
    compute_sentiment,
    fit_nmf_topic_model,
)
from src.scraper import scrape_reddit_dataset
from src.visualization import generate_all_visualizations

__version__ = "0.2.0"

__all__ = [
    "clean_reddit_data",
    "normalize_text",
    "detect_iqr_outliers",
    "categorize_title",
    "categorize_posts",
    "compute_sentiment",
    "fit_nmf_topic_model",
    "assign_nmf_topics",
    "scrape_reddit_dataset",
    "generate_all_visualizations",
]
