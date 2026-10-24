"""Reddit Data Science Scraping & Analysis Orchestrator.

Main entry point for automated execution inside Docker or local CLI.
Runs the end-to-end data pipeline:
1. Ingestion (Live scraping with automatic fallback to cached dataset)
2. Cleaning & Feature Engineering (Zero data loss, bot removal, log transforms)
3. Topic Categorization (Word boundaries regex + flairs)
4. Unsupervised NMF Topic Modeling (TF-IDF latent theme discovery)
5. VADER Sentiment Scoring (Domain-aware NLP)
6. Statistical Summary & Verification
7. Exporting Publication-Grade Visualizations to `img/`
"""

import os
import sys
import argparse
import logging
from scipy import stats
import pandas as pd

from src.scraper import scrape_reddit_dataset
from src.cleaning import clean_reddit_data
from src.nlp import categorize_posts, compute_sentiment, fit_nmf_topic_model
from src.visualization import generate_all_visualizations

# Set writable matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    scrape: bool = True,
    pages: int = 5,
    input_csv: str = "data/reddit_datascience.csv",
    output_clean_csv: str = "data/reddit_datascience_clean.csv",
    img_dir: str = "img",
    export_images: bool = True,
) -> pd.DataFrame:
    """Executes the complete end-to-end data science pipeline."""
    print("=" * 70)
    print("🚀 STARTING REDDIT DATA SCIENCE ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Data Ingestion
    if scrape:
        logger.info("Step 1/6: Ingesting data via live scraper (with cached fallback)...")
        df_raw = scrape_reddit_dataset(pages=pages, fallback_csv=input_csv)
    else:
        logger.info("Step 1/6: Loading existing dataset from %s...", input_csv)
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file not found: {input_csv}")
        df_raw = pd.read_csv(input_csv)

    print(f"\n📊 Raw Dataset Ingested: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # 2. Cleaning & Feature Engineering
    logger.info("Step 2/6: Cleaning data and engineering features (preserving title-only posts)...")
    df_clean = clean_reddit_data(df_raw, remove_bots=True)
    retention_rate = (len(df_clean) / max(len(df_raw), 1)) * 100
    print(f"✅ Organic Post Retention: {len(df_clean)} posts retained ({retention_rate:.1f}%)")

    # 3. Topic Categorization
    logger.info("Step 3/6: Categorizing posts via word-bounded regex patterns...")
    df_cat = categorize_posts(df_clean)

    # 4. Sentiment Analysis
    logger.info("Step 4/6: Computing VADER sentiment compound scores...")
    df_sent = compute_sentiment(df_cat)

    # 5. Unsupervised NMF Topic Modeling
    logger.info("Step 5/6: Fitting NMF Topic Model on post full texts...")
    nmf_model, tfidf_vec, topics = fit_nmf_topic_model(df_sent["full_text"].tolist(), n_topics=5)

    # 6. Save Cleaned Dataset
    os.makedirs(os.path.dirname(output_clean_csv), exist_ok=True)
    df_sent.to_csv(output_clean_csv, index=False)
    logger.info("Saved cleaned & enriched dataset to %s", output_clean_csv)

    # Statistical Overview
    print("\n" + "=" * 70)
    print("📈 STATISTICAL SUMMARY & INSIGHTS")
    print("=" * 70)

    print("\n[Topic Category Breakdown]")
    cat_summary = df_sent["category"].value_counts()
    for cat, count in cat_summary.items():
        print(f"  • {cat:<12}: {count:>3} posts ({count/len(df_sent)*100:>5.1f}%)")

    print("\n[Engagement Metrics Summary (Non-Parametric & Parametric)]")
    for metric in ["score", "comment_count"]:
        s = df_sent[metric]
        print(f"  • {metric.replace('_', ' ').title():<14}: Mean={s.mean():.1f}, Std={s.std():.1f} | Median={s.median():.0f}, IQR={s.quantile(0.75) - s.quantile(0.25):.0f}, Max={s.max()}")

    rho, p_val = stats.spearmanr(df_sent["score"], df_sent["comment_count"])
    print(f"\n[Correlation (Score vs Comments)]")
    print(f"  • Spearman's Rank Correlation: rho = {rho:.3f} (p = {p_val:.2e})")

    print("\n[Discovered Latent NMF Topics (Unsupervised)]")
    for t in topics:
        print(f"  • {t['name']:<35} -> [{', '.join(t['keywords'][:6])}]")

    print("\n[VADER Sentiment Polarity Distribution]")
    sent_summary = df_sent["sentiment_label"].value_counts()
    for lbl, count in sent_summary.items():
        print(f"  • {lbl:<10}: {count:>3} posts ({count/len(df_sent)*100:>5.1f}%)")

    # 7. Generate Visualizations
    if export_images:
        print("\n" + "=" * 70)
        print("🎨 EXPORTING PUBLICATION VISUALIZATIONS TO img/")
        print("=" * 70)
        generate_all_visualizations(df_sent, topics, output_dir=img_dir)
        print(f"✅ All figures successfully exported to '{img_dir}/'")

    print("\n" + "=" * 70)
    print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_sent


def main():
    parser = argparse.ArgumentParser(description="Reddit Data Science Scraping & Analysis Orchestrator")
    parser.add_argument("--scrape", action="store_true", default=True, help="Attempt live scraping with fallback")
    parser.add_argument("--no-scrape", action="store_false", dest="scrape", help="Skip scraping and use cached CSV")
    parser.add_argument("--pages", type=int, default=5, help="Number of listing pages to scrape (if live)")
    parser.add_argument("--input", type=str, default="data/reddit_datascience.csv", help="Input raw CSV path")
    parser.add_argument("--output", type=str, default="data/reddit_datascience_clean.csv", help="Output cleaned CSV path")
    parser.add_argument("--img-dir", type=str, default="img", help="Directory to save exported charts")
    parser.add_argument("--no-images", action="store_false", dest="export_images", help="Skip generating images")

    args = parser.parse_args()
    run_pipeline(
        scrape=args.scrape,
        pages=args.pages,
        input_csv=args.input,
        output_clean_csv=args.output,
        img_dir=args.img_dir,
        export_images=args.export_images,
    )


if __name__ == "__main__":
    main()

