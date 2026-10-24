"""Natural Language Processing, Categorization, and Topic Modeling module.

Features:
- Regex-based post categorization with strict word boundaries (prevents false matches like 'r')
- Automatic offline NLTK data resolution for VADER sentiment analyzer
- Unsupervised NMF (Non-Negative Matrix Factorization) Topic Modeling via TF-IDF
- N-gram and keyphrase extraction with domain-specific stopwords
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk

logger = logging.getLogger(__name__)

# Register local nltk data path if available
LOCAL_NLTK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "nltk_data"))
if os.path.exists(LOCAL_NLTK_DIR) and LOCAL_NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DIR)

# Domain-specific stopwords to remove noisy common terms
DOMAIN_STOPWORDS = {
    "data", "science", "datascience", "subreddit", "reddit", "post", "posts",
    "people", "like", "just", "would", "get", "one", "think", "also", "know",
    "anyone", "time", "thread", "discussion", "official", "weekly", "daily",
    "year", "years", "much", "even", "really", "going", "want", "work", "good"
}


# Pre-compiled regex patterns with strict word boundaries
CATEGORY_PATTERNS = {
    "AI/LLM": re.compile(
        r"\b(ai|llm|llms|genai|gpt|chatgpt|claude|rag|agent|agents|agentic|transformer|transformers|copilot|prompt|prompting|tabpfn)\b",
        re.IGNORECASE,
    ),
    "Salary": re.compile(
        r"\b(salary|salaries|compensation|total comp|comp|pay|bonus|stock options|offer|offers|remuneration)\b",
        re.IGNORECASE,
    ),
    "Career": re.compile(
        r"\b(job|jobs|career|careers|hiring|hired|resume|cv|interview|interviews|faang|transition|break into|entry level|promotion|promoted|layoff|layoffs|bootcamp|phd|masters|degree|internship|intern)\b",
        re.IGNORECASE,
    ),
    "Project": re.compile(
        r"\b(project|projects|portfolio|github|kaggle|side project|capstone|dissertation|showcase|demo|repo|dataset|datasets|pipeline)\b",
        re.IGNORECASE,
    ),
    "Learning": re.compile(
        r"\b(learn|learning|tutorial|tutorials|course|courses|book|books|mooc|study|studying|roadmap|beginner|beginners|statistics|math|machine learning|deep learning|xgboost|tensorflow|pytorch|pandas|sql|\br\b|rstats)\b",
        re.IGNORECASE,
    ),
    "Advice": re.compile(
        r"\b(advice|advise|help|tips|question|questions|guidance|recommend|recommendation|suggest|suggestion|thoughts|feedback|opinion)\b",
        re.IGNORECASE,
    ),
}


def categorize_title(title: str, flair: Optional[str] = None) -> str:
    """Categorizes a post using strict word-bounded regex and flair heuristics.

    Args:
        title: Post title string.
        flair: Optional post flair label from Reddit.

    Returns:
        Assigned category string.
    """
    if flair and isinstance(flair, str) and flair.strip():
        f = flair.strip().lower()
        if "career" in f or "job" in f:
            return "Career"
        if "project" in f or "portfolio" in f:
            return "Project"
        if "education" in f or "learning" in f:
            return "Learning"
        if "discussion" in f:
            # Let title regex refine discussion flairs
            pass

    if not title or not isinstance(title, str):
        return "Other"

    # Evaluate patterns in prioritized order
    for cat, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(title):
            return cat

    return "Other"


def categorize_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Applies topic categorization to post titles."""
    df = df.copy()
    flairs = df["flair"] if "flair" in df.columns else [None] * len(df)
    df["category"] = [
        categorize_title(title, flair)
        for title, flair in zip(df["post_title"], flairs)
    ]
    return df


def get_sentiment_analyzer():
    """Initializes and returns SentimentIntensityAnalyzer."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


def compute_sentiment(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """Computes VADER sentiment scores for posts."""
    df = df.copy()
    sia = get_sentiment_analyzer()

    scores = df[text_col].fillna("").astype(str).apply(sia.polarity_scores)
    df["sentiment_compound"] = [s["compound"] for s in scores]
    df["sentiment_pos"] = [s["pos"] for s in scores]
    df["sentiment_neg"] = [s["neg"] for s in scores]
    df["sentiment_neu"] = [s["neu"] for s in scores]

    df["sentiment_label"] = pd.cut(
        df["sentiment_compound"],
        bins=[-float("inf"), -0.05, 0.05, float("inf")],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


def fit_nmf_topic_model(
    texts: List[str],
    n_topics: int = 5,
    n_top_words: int = 8,
    random_state: int = 42,
) -> Tuple[NMF, TfidfVectorizer, List[Dict[str, Any]]]:
    """Fits an unsupervised NMF topic model on post texts.

    Args:
        texts: List of text documents.
        n_topics: Number of latent topics to discover.
        n_top_words: Number of top keywords per topic.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (fitted NMF model, fitted TfidfVectorizer, list of topic summaries).
    """
    try:
        from nltk.corpus import stopwords
        base_stopwords = set(stopwords.words("english"))
    except Exception:
        base_stopwords = set()

    all_stopwords = list(base_stopwords | DOMAIN_STOPWORDS)

    tfidf = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=all_stopwords,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    dtm = tfidf.fit_transform(texts)

    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=400,
    )
    nmf.fit(dtm)

    feature_names = tfidf.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = [float(topic[i]) for i in top_features_ind]
        topics.append({
            "topic_id": topic_idx + 1,
            "name": f"Topic {topic_idx + 1}: {', '.join(top_features[:3])}",
            "keywords": top_features,
            "weights": weights,
        })

    return nmf, tfidf, topics
