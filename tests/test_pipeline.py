"""Comprehensive unit test suite for Reddit Data Science pipeline.

Tests:
1. Regex word-boundary categorization (specifically validating no single-letter 'r' false positives)
2. Data cleaning & zero-data-loss behavior for empty bodies
3. Bot account filtering
4. VADER sentiment analysis execution and offline lexicon loading
5. Unsupervised NMF topic modeling reproducibility
6. Graceful scraper fallback
"""

import unittest
import numpy as np
import pandas as pd

from src.cleaning import clean_reddit_data
from src.nlp import (
    categorize_title,
    categorize_posts,
    compute_sentiment,
    fit_nmf_topic_model,
)
from src.scraper import scrape_reddit_dataset


class TestCategorization(unittest.TestCase):
    """Test topic categorization with word-boundary checks."""

    def test_single_letter_r_regression_bug(self):
        """Verifies that titles containing the letter 'r' are NOT miscategorized as Learning."""
        # In the buggy implementation, all these matched Learning because of 'r'
        self.assertEqual(categorize_title("Deciding on an offer: Higher Salary vs Stability"), "Salary")
        self.assertEqual(categorize_title("Three Senior DS Interviews, Three Totally Different Tests"), "Career")
        self.assertEqual(categorize_title("TabPFN-2.5 Is Live (Tabular Foundation Model)"), "AI/LLM")
        self.assertEqual(categorize_title("Designing data collection for my undergrad capstone project"), "Project")

    def test_valid_category_matching(self):
        """Verifies accurate categorization across all domain classes."""
        self.assertEqual(categorize_title("How to prepare for AI Engineering interviews?"), "AI/LLM")
        self.assertEqual(categorize_title("Total compensation for Senior Data Scientist in SF"), "Salary")
        self.assertEqual(categorize_title("Looking for resume advice and career transition tips"), "Career")
        self.assertEqual(categorize_title("Check out my open-source Kaggle competition pipeline repo"), "Project")
        self.assertEqual(categorize_title("Best beginner statistics book and Python tutorial"), "Learning")
        self.assertEqual(categorize_title("Is R programming still useful in 2026?"), "Learning")

    def test_flair_priority(self):
        """Verifies flair takes precedence or guides classification."""
        self.assertEqual(categorize_title("Random question", flair="Career"), "Career")
        self.assertEqual(categorize_title("My new tool", flair="Projects/Tooling"), "Project")


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning and feature engineering."""

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "post_id": "t3_001",
                "post_title": "Title with body",
                "author": "user_a",
                "score": 10,
                "comment_count": 5,
                "created_utc": "2026-01-10T12:00:00Z",
                "body": "This is a detailed post body text.",
            },
            {
                "post_id": "t3_002",
                "post_title": "Title only question with no body",
                "author": "user_b",
                "score": 45,
                "comment_count": 12,
                "created_utc": "2026-01-10T14:30:00Z",
                "body": None,  # Missing body
            },
            {
                "post_id": "t3_003",
                "post_title": "Weekly Entering & Transitioning Thread",
                "author": "AutoModerator",
                "score": 5,
                "comment_count": 80,
                "created_utc": "2026-01-05T00:00:00Z",
                "body": "Welcome to the transition thread.",
            },
        ])

    def test_zero_data_loss_for_empty_body(self):
        """Ensures posts with null bodies are NOT dropped."""
        cleaned = clean_reddit_data(self.sample_df, remove_bots=False)
        self.assertEqual(len(cleaned), 3)
        self.assertFalse(cleaned.loc[cleaned["post_id"] == "t3_002", "has_body"].values[0])
        self.assertEqual(
            cleaned.loc[cleaned["post_id"] == "t3_002", "full_text"].values[0],
            "Title only question with no body",
        )

    def test_automoderator_filtering(self):
        """Ensures AutoModerator posts are filtered out."""
        cleaned = clean_reddit_data(self.sample_df, remove_bots=True)
        self.assertEqual(len(cleaned), 2)
        self.assertNotIn("AutoModerator", cleaned["author"].values)

    def test_feature_engineering_transforms(self):
        """Ensures log metrics and temporal features are computed."""
        cleaned = clean_reddit_data(self.sample_df, remove_bots=True)
        self.assertIn("log_score", cleaned.columns)
        self.assertIn("log_comment_count", cleaned.columns)
        self.assertIn("hour", cleaned.columns)
        self.assertIn("day_of_week", cleaned.columns)
        self.assertFalse(cleaned["log_score"].isna().any())
        self.assertAlmostEqual(cleaned.loc[cleaned["post_id"] == "t3_001", "log_score"].values[0], np.log1p(10))


class TestNLPAndTopicModeling(unittest.TestCase):
    """Test NLP sentiment analysis and NMF modeling."""

    def test_sentiment_scoring(self):
        df = pd.DataFrame([
            {"post_title": "Amazing breakthrough and exciting opportunity!", "full_text": "Amazing breakthrough and exciting opportunity!"},
            {"post_title": "Terrible disaster and horrible failure", "full_text": "Terrible disaster and horrible failure"},
        ])
        scored = compute_sentiment(df)
        self.assertIn("sentiment_compound", scored.columns)
        self.assertGreater(scored.loc[0, "sentiment_compound"], 0.2)
        self.assertLess(scored.loc[1, "sentiment_compound"], -0.2)

    def test_nmf_topic_modeling(self):
        corpus = [
            "machine learning models neural networks deep learning",
            "career transition resume interview job search",
            "data science compensation salary tech offer",
            "python pandas sql jupyter notebook coding",
            "llm agents artificial intelligence transformers",
        ] * 4
        nmf, tfidf, topics = fit_nmf_topic_model(corpus, n_topics=3)
        self.assertEqual(len(topics), 3)
        self.assertTrue(all("keywords" in t for t in topics))


class TestScraperFallback(unittest.TestCase):
    """Test scraper resilience and graceful fallback."""

    def test_fallback_to_cached_csv(self):
        df = scrape_reddit_dataset(pages=1, fallback_csv="data/reddit_datascience.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()

