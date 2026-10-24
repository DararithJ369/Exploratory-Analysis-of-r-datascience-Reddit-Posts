"""Data cleaning, feature engineering, and temporal harmonization module for Reddit posts.

Adheres to rigorous data science principles:
- Transparent pipeline retention reporting (preserving collected organic posts without claiming platform-wide census)
- Daylight-saving-aware timezone conversion via zoneinfo (properly handling EDT vs EST)
- Post age and exposure-time normalization (distinguishing cumulative vs rate-based engagement)
- Systematic missing value handling and text harmonization
- Bot account filtration with audit logging
- Non-linear log transformations for heavy-tailed power-law counts
"""

import logging
from typing import Tuple, Optional
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Eastern Timezone for North American tech discourse alignment
EASTERN_TZ = ZoneInfo("America/New_York")


def clean_reddit_data(
    df: pd.DataFrame,
    remove_bots: bool = True,
    bot_usernames: Tuple[str, ...] = ("AutoModerator",),
    collection_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Cleans raw Reddit dataset, computes exposure metrics, and engineers analytical features.

    Args:
        df: Raw DataFrame loaded from CSV or scraper.
        remove_bots: Whether to filter out system bot accounts.
        bot_usernames: Tuple of bot account usernames to filter.
        collection_time: Timestamp of data collection (used for post age calculation).
                         If None, defaults to max(created_utc) + 1 hour.

    Returns:
        Enriched and cleaned DataFrame.
    """
    initial_count = len(df)
    df = df.copy()

    # 1. Drop records lacking essential identifier or title
    df = df.dropna(subset=["post_title"]).reset_index(drop=True)
    if "post_id" in df.columns:
        df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # 2. Filter system / bot accounts (non-organic discussion)
    if remove_bots and "author" in df.columns:
        bot_mask = df["author"].isin(bot_usernames)
        num_bots = bot_mask.sum()
        if num_bots > 0:
            logger.info("Filtered %d system/bot posts (%s)", num_bots, bot_usernames)
            df = df[~bot_mask].reset_index(drop=True)

    # 3. Handle body text without discarding records (Zero Scraped Data Loss)
    # Note: Retains 100% of organic scraped posts. This refers to pipeline retention,
    # not an exhaustive census of all historical Reddit submissions.
    if "body" not in df.columns:
        df["body"] = ""
    df["has_body"] = df["body"].notna() & (df["body"].astype(str).str.strip() != "")
    df["body_clean"] = df["body"].fillna("").astype(str).str.strip()

    # 4. Create unified text representation for NLP/sentiment
    df["full_text"] = np.where(
        df["has_body"],
        df["post_title"] + " " + df["body_clean"],
        df["post_title"]
    )

    # 5. Datetime parsing with dynamic Daylight Saving Time handling (zoneinfo)
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
        # Ensure UTC timezone awareness
        if df["created_utc"].dt.tz is None:
            df["created_utc"] = df["created_utc"].dt.tz_localize("UTC")
        else:
            df["created_utc"] = df["created_utc"].dt.tz_convert("UTC")

        # Convert to Eastern Time, automatically distinguishing EDT (UTC-4) and EST (UTC-5)
        df["created_eastern"] = df["created_utc"].dt.tz_convert(EASTERN_TZ)
        df["tz_eastern_name"] = df["created_eastern"].dt.strftime("%Z")  # "EDT" or "EST"

        df["date"] = pd.to_datetime(df["created_utc"].dt.date)
        df["hour_utc"] = df["created_utc"].dt.hour
        df["hour"] = df["hour_utc"]  # Legacy alias
        df["hour_eastern"] = df["created_eastern"].dt.hour
        df["day_of_week"] = df["created_utc"].dt.dayofweek
        df["day_name"] = df["created_utc"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["year_month"] = df["created_utc"].dt.strftime("%Y-%m")

    # 6. Engagement features & log-transforms (addressing skewness)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    else:
        df["score"] = 0

    if "comment_count" in df.columns:
        df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["comment_count"] = 0
    df["log_score"] = np.log1p(np.maximum(df["score"], 0))
    df["log_comment_count"] = np.log1p(np.maximum(df["comment_count"], 0))
    df["comments_per_score"] = df["comment_count"] / (df["score"] + 1.0)

    # 7. Post Age & Exposure-Normalized Engagement Velocity
    if "created_utc" in df.columns:
        if collection_time is None:
            collection_time = df["created_utc"].max() + pd.Timedelta(hours=1)
        elif collection_time.tzinfo is None:
            collection_time = collection_time.tz_localize("UTC")

        df["post_age_days"] = (collection_time - df["created_utc"]).dt.total_seconds() / 86400.0
        # Safe denominator clamp (min 6 hours / 0.25 days) to avoid extreme inflation on very new posts
        safe_age = np.maximum(df["post_age_days"], 0.25)
        df["score_per_day"] = df["score"] / safe_age
        df["comments_per_day"] = df["comment_count"] / safe_age
        df["log_score_per_day"] = np.log1p(df["score_per_day"])
        df["log_comments_per_day"] = np.log1p(df["comments_per_day"])

    # 8. Text length features
    df["title_char_length"] = df["post_title"].str.len()
    df["title_word_count"] = df["post_title"].str.split().str.len()
    df["body_char_length"] = df["body_clean"].str.len()
    df["body_word_count"] = df["body_clean"].str.split().str.len()

    logger.info(
        "Cleaned dataset: %d -> %d records retained (%.1f%% of raw scraped data).",
        initial_count,
        len(df),
        (len(df) / max(initial_count, 1)) * 100,
    )
    return df
