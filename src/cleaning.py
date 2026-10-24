"""Data cleaning and feature engineering module for Reddit posts.

Adheres to rigorous data science principles:
- Zero data loss for title-only submissions (preserves 100% of organic posts)
- Systematic missing value imputation and text composition
- System/bot account filtering with audit logging
- Non-linear log transformations for heavy-tailed engagement distributions
- Temporal and text-based feature extraction
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_reddit_data(
    df: pd.DataFrame,
    remove_bots: bool = True,
    bot_usernames: Tuple[str, ...] = ("AutoModerator",),
) -> pd.DataFrame:
    """Cleans raw Reddit dataset and engineers analytical features.

    Args:
        df: Raw DataFrame loaded from CSV or scraper.
        remove_bots: Whether to filter out bot accounts.
        bot_usernames: Tuple of bot account usernames to filter.

    Returns:
        Enriched and cleaned DataFrame.
    """
    initial_count = len(df)
    df = df.copy()

    # 1. Drop records lacking essential identifier or title
    df = df.dropna(subset=["post_title"]).reset_index(drop=True)
    if "post_id" in df.columns:
        df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # 2. Filter system / bot accounts
    if remove_bots and "author" in df.columns:
        bot_mask = df["author"].isin(bot_usernames)
        num_bots = bot_mask.sum()
        if num_bots > 0:
            logger.info("Filtered %d system/bot posts (%s)", num_bots, bot_usernames)
            df = df[~bot_mask].reset_index(drop=True)

    # 3. Handle body text without dropping records
    df["has_body"] = df["body"].notna() & (df["body"].astype(str).str.strip() != "")
    df["body_clean"] = df["body"].fillna("").astype(str).str.strip()

    # 4. Create unified text representation for NLP/sentiment
    # If body exists, combine title + body; otherwise use title
    df["full_text"] = np.where(
        df["has_body"],
        df["post_title"] + " " + df["body_clean"],
        df["post_title"]
    )

    # 5. Datetime parsing & temporal features
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
        df["date"] = pd.to_datetime(df["created_utc"].dt.date)
        df["hour"] = df["created_utc"].dt.hour
        df["day_of_week"] = df["created_utc"].dt.dayofweek
        df["day_name"] = df["created_utc"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["year_month"] = df["created_utc"].dt.strftime("%Y-%m")

    # 6. Engagement features & log-transforms (addressing skewness)
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df.get("comment_count", 0), errors="coerce").fillna(0).astype(int)

    df["log_score"] = np.log1p(np.maximum(df["score"], 0))
    df["log_comment_count"] = np.log1p(np.maximum(df["comment_count"], 0))
    df["comments_per_score"] = df["comment_count"] / (df["score"] + 1.0)

    # 7. Text length features
    df["title_char_length"] = df["post_title"].str.len()
    df["title_word_count"] = df["post_title"].str.split().str.len()
    df["body_char_length"] = df["body_clean"].str.len()
    df["body_word_count"] = df["body_clean"].str.split().str.len()

    logger.info(
        "Cleaned dataset: %d -> %d records retained (%.1f%% of raw data).",
        initial_count,
        len(df),
        (len(df) / max(initial_count, 1)) * 100,
    )
    return df
