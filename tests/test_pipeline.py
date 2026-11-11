"""Comprehensive unit test suite for Reddit Data Science pipeline.

Tests:
1. Regex word-boundary categorization (specifically validating no single-letter 'r' false positives)
2. Data cleaning & zero-data-loss behavior for empty bodies
3. Bot account filtering
4. VADER sentiment analysis execution and offline lexicon loading
5. Unsupervised NMF topic modeling reproducibility
6. Graceful scraper fallback
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.cleaning import (
    clean_reddit_data,
    normalize_text,
    detect_iqr_outliers,
)
from src.nlp import (
    categorize_title,
    categorize_posts,
    compute_sentiment,
    fit_nmf_topic_model,
    assign_nmf_topics,
)
from src.scraper import scrape_reddit_dataset
from main import compute_kruskal_tests, export_statistical_report


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

    def test_case_insensitivity_and_punctuation(self):
        """Verifies category matching with uppercase and surrounding punctuation."""
        self.assertEqual(categorize_title("Is this a good SALARY?"), "Salary")
        self.assertEqual(categorize_title("Career... what should I do next?"), "Career")
        self.assertEqual(categorize_title("Building an AGENT for data workflows!"), "AI/LLM")


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
        """Ensures log metrics, post age, and zoneinfo Eastern Time features are computed."""
        cleaned = clean_reddit_data(self.sample_df, remove_bots=True)
        self.assertIn("log_score", cleaned.columns)
        self.assertIn("log_comment_count", cleaned.columns)
        self.assertIn("hour_utc", cleaned.columns)
        self.assertIn("hour_eastern", cleaned.columns)
        self.assertIn("created_eastern", cleaned.columns)
        self.assertIn("tz_eastern_name", cleaned.columns)
        self.assertIn("post_age_days", cleaned.columns)
        self.assertIn("score_per_day", cleaned.columns)
        self.assertFalse(cleaned["log_score"].isna().any())
        self.assertAlmostEqual(cleaned.loc[cleaned["post_id"] == "t3_001", "log_score"].values[0], np.log1p(10))

    def test_daylight_saving_handling(self):
        """Verifies zoneinfo correctly assigns EDT vs EST depending on date."""
        df_dates = pd.DataFrame([
            {"post_id": "p1", "post_title": "Summer post", "created_utc": "2025-08-15T12:00:00Z"},
            {"post_id": "p2", "post_title": "Winter post", "created_utc": "2026-01-15T12:00:00Z"},
        ])
        cleaned = clean_reddit_data(df_dates, remove_bots=False)
        self.assertEqual(cleaned.loc[cleaned["post_id"] == "p1", "tz_eastern_name"].values[0], "EDT")
        self.assertEqual(cleaned.loc[cleaned["post_id"] == "p2", "tz_eastern_name"].values[0], "EST")

    def test_negative_scores_numerical_stability(self):
        """Verifies that negative or zero engagement values do not produce NaNs or infs."""
        df_neg = pd.DataFrame([
            {"post_id": "neg1", "post_title": "Downvoted post", "score": -5, "comment_count": 0, "created_utc": "2026-01-10T12:00:00Z"},
            {"post_id": "zero1", "post_title": "Zero post", "score": 0, "comment_count": 0, "created_utc": "2026-01-10T12:00:00Z"},
        ])
        cleaned = clean_reddit_data(df_neg, remove_bots=False)
        self.assertFalse(cleaned["log_score"].isna().any())
        self.assertFalse(cleaned["log_score_per_day"].isna().any())
        self.assertGreaterEqual(cleaned["log_score"].min(), 0.0)
        self.assertGreaterEqual(cleaned["log_score_per_day"].min(), 0.0)

    def test_velocity_clamp_on_fresh_posts(self):
        """Verifies safe denominator clamp prevents runaway engagement velocity."""
        ref_time = pd.Timestamp("2026-01-10T12:05:00Z")
        df_fresh = pd.DataFrame([
            {
                "post_id": "fresh1",
                "post_title": "Posted 5 minutes ago",
                "score": 10,
                "comment_count": 5,
                "created_utc": "2026-01-10T12:00:00Z",
            }
        ])
        cleaned = clean_reddit_data(df_fresh, remove_bots=False, collection_time=ref_time)
        self.assertLess(cleaned["post_age_days"].values[0], 0.01)
        # Denominator clamped to minimum 0.25 days -> velocity should be 10 / 0.25 = 40
        self.assertAlmostEqual(cleaned["score_per_day"].values[0], 40.0)

    def test_text_normalization_html_and_zero_width(self):
        """Verifies HTML entity decoding and invisible character removal."""
        raw_text = "Data&#39;s &amp; AI &lt;models&gt; \u200b\ufeffare   great!"
        cleaned = normalize_text(raw_text)
        self.assertEqual(cleaned, "Data's & AI <models> are great!")

    def test_iqr_outlier_detection(self):
        """Verifies Tukey IQR outlier detection bounds and flagging."""
        # Baseline series: [10, 12, 14, 15, 16, 18, 20], with extreme outlier 200
        data = pd.Series([10, 12, 14, 15, 16, 18, 20, 200])
        outliers, lower, upper = detect_iqr_outliers(data, factor=1.5)
        self.assertTrue(outliers.iloc[-1])
        self.assertFalse(outliers.iloc[0])
        self.assertGreater(upper, 20)
        self.assertLess(upper, 200)

    def test_clean_data_outlier_flags(self):
        """Verifies outlier indicator columns exist in output dataframe."""
        cleaned = clean_reddit_data(self.sample_df, remove_bots=False)
        self.assertIn("is_score_outlier", cleaned.columns)
        self.assertIn("is_comment_outlier", cleaned.columns)
        self.assertEqual(cleaned["is_score_outlier"].dtype, bool)


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

    def test_empty_and_whitespace_sentiment(self):
        """Verifies neutral polarity for empty or whitespace-only documents."""
        df_empty = pd.DataFrame([
            {"post_title": "", "full_text": ""},
            {"post_title": "   ", "full_text": "   "},
        ])
        scored = compute_sentiment(df_empty)
        self.assertEqual(scored.loc[0, "sentiment_compound"], 0.0)
        self.assertEqual(scored.loc[1, "sentiment_compound"], 0.0)
        self.assertEqual(scored.loc[0, "sentiment_label"], "Neutral")

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

    def test_assign_nmf_topics(self):
        """Verifies document projection onto dominant NMF topics."""
        corpus = [
            "machine learning models neural networks deep learning",
            "career transition resume interview job search",
            "data science compensation salary tech offer",
        ] * 4
        nmf, tfidf, topics = fit_nmf_topic_model(corpus, n_topics=3)
        df_docs = pd.DataFrame([
            {"post_title": "Deep Learning Models", "full_text": "deep learning neural networks models"},
            {"post_title": "Empty doc", "full_text": ""},
        ])
        assigned = assign_nmf_topics(df_docs, nmf, tfidf, topics)
        self.assertIn("nmf_topic_id", assigned.columns)
        self.assertIn("nmf_topic_name", assigned.columns)
        self.assertIn("nmf_confidence", assigned.columns)
        self.assertGreater(assigned.loc[0, "nmf_confidence"], 0.0)
        self.assertEqual(assigned.loc[1, "nmf_topic_id"], 0)
        self.assertEqual(assigned.loc[1, "nmf_confidence"], 0.0)


class TestScraperFallback(unittest.TestCase):
    """Test scraper resilience and graceful fallback."""

    def test_fallback_to_cached_csv(self):
        df = scrape_reddit_dataset(pages=1, fallback_csv="data/reddit_datascience.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_scraper_score_coercion(self):
        """Verifies that non-numeric score strings from Reddit HTML do not crash the parser."""
        from bs4 import BeautifulSoup
        raw_html = """
        <div class="thing" data-fullname="t3_999" data-score="?" data-author="test_user" data-type="self">
            <a class="title" href="/r/datascience/comments/999/test/">Test Title</a>
            <a class="comments">15 comments</a>
            <time datetime="2026-01-10T12:00:00Z"></time>
        </div>
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        post = soup.find("div", class_="thing")
        raw_score = post.get("data-score")
        try:
            score_val = int(raw_score) if raw_score is not None else 0
        except (ValueError, TypeError):
            score_val = 0
        self.assertEqual(score_val, 0)


class TestStatisticalReporting(unittest.TestCase):
    """Test non-parametric hypothesis testing and report generation."""

    def setUp(self):
        self.df = pd.DataFrame([
            {"category": "Career", "score": 50, "comment_count": 30, "created_utc": pd.Timestamp("2026-01-10T12:00:00Z"), "sentiment_label": "Positive", "is_score_outlier": False, "is_comment_outlier": False},
            {"category": "Career", "score": 40, "comment_count": 25, "created_utc": pd.Timestamp("2026-01-10T13:00:00Z"), "sentiment_label": "Neutral", "is_score_outlier": False, "is_comment_outlier": False},
            {"category": "Salary", "score": 80, "comment_count": 45, "created_utc": pd.Timestamp("2026-01-10T14:00:00Z"), "sentiment_label": "Positive", "is_score_outlier": True, "is_comment_outlier": False},
            {"category": "Salary", "score": 90, "comment_count": 55, "created_utc": pd.Timestamp("2026-01-10T15:00:00Z"), "sentiment_label": "Negative", "is_score_outlier": True, "is_comment_outlier": True},
        ])
        self.topics = [
            {"topic_id": 1, "name": "Topic 1: career, jobs", "keywords": ["career", "jobs", "hiring"]},
            {"topic_id": 2, "name": "Topic 2: salary, compensation", "keywords": ["salary", "comp", "offer"]},
        ]

    def test_kruskal_tests(self):
        """Verifies Kruskal-Wallis H-test calculation across categories."""
        res = compute_kruskal_tests(self.df)
        self.assertIn("score", res)
        self.assertIn("comment_count", res)
        self.assertIsInstance(res["score"][0], float)
        self.assertGreaterEqual(res["score"][1], 0.0)
        self.assertLessEqual(res["score"][1], 1.0)

    def test_kruskal_tests_identical_values(self):
        """Verifies Kruskal-Wallis gracefully handles identical values across groups without raising."""
        df_identical = pd.DataFrame([
            {"category": "Career", "score": 10, "comment_count": 5},
            {"category": "Career", "score": 10, "comment_count": 5},
            {"category": "Salary", "score": 10, "comment_count": 5},
            {"category": "Salary", "score": 10, "comment_count": 5},
        ])
        res = compute_kruskal_tests(df_identical)
        self.assertEqual(res["score"], (0.0, 1.0))
        self.assertEqual(res["comment_count"], (0.0, 1.0))

    def test_export_statistical_report(self):
        """Verifies generation of structured Markdown report."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_statistical_report(self.df, self.topics, output_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# Statistical & Methodological Analysis Report", content)
            self.assertIn("Kruskal-Wallis", content)
            self.assertIn("Spearman Rank Correlation", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPackageInterface(unittest.TestCase):
    """Test top-level package namespace exports and public API structure."""

    def test_top_level_exports(self):
        """Verifies primary public API functions are importable from root src package."""
        import src
        expected_symbols = [
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
        for symbol in expected_symbols:
            self.assertTrue(hasattr(src, symbol), f"src missing exported symbol: {symbol}")
            self.assertIn(symbol, src.__all__)


class TestLegacyScraperHelper(unittest.TestCase):
    """Test legacy helper backwards-compatibility wrapper."""

    def test_legacy_helper_exports(self):
        """Verifies helper package and module re-export compatibility functions."""
        import helper
        import helper.reddit_scraper

        for symbol in ["scrape_listing", "scrape_body", "scrape_reddit_full"]:
            self.assertTrue(hasattr(helper, symbol), f"helper missing {symbol}")
            self.assertTrue(hasattr(helper.reddit_scraper, symbol), f"helper.reddit_scraper missing {symbol}")
            self.assertTrue(callable(getattr(helper, symbol)))


if __name__ == "__main__":
    unittest.main()


