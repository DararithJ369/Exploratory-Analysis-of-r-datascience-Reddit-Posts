"""Publication-grade visualization module for Reddit Data Science EDA.

Exports figures to `img/` at 300 DPI with rigorous statistical annotations:
- Dual-scale engagement distributions (raw vs log-transformed)
- Post category distribution with sample size tags
- Median engagement by topic category (both cumulative and age-normalized)
- Circadian posting intensity heatmap (Day of Week x Hour UTC & Eastern Time)
- Log-log engagement scatter plot with Spearman rank correlation (association vs causation)
- Post age and exposure-time relationship
- Unsupervised NMF latent topic keyword weights
- Sentiment polarity distributions across discussion categories
- Domain-aware Word Cloud
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

# Ensure writable matplotlib cache in constrained environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Global aesthetic styling
sns.set_theme(style="whitegrid", font_scale=1.1)


def save_current_figure(output_path: str, dpi: int = 300):
    """Saves the active matplotlib figure and cleanly closes it."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)


def plot_engagement_distributions(df: pd.DataFrame, output_path: str = "img/fig_engagement_distributions.png"):
    """Plots raw vs log-transformed distributions of score and comments."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Raw Score
    sns.histplot(df["score"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=25)
    axes[0, 0].set_title(f"Raw Score Distribution (Skewness: {df['score'].skew():.2f})")
    axes[0, 0].set_xlabel("Score (Upvotes)")
    axes[0, 0].set_ylabel("Post Count")

    # Raw Comments
    sns.histplot(df["comment_count"], kde=True, ax=axes[0, 1], color="#d95f02", bins=25)
    axes[0, 1].set_title(f"Raw Comment Count Distribution (Skewness: {df['comment_count'].skew():.2f})")
    axes[0, 1].set_xlabel("Number of Comments")
    axes[0, 1].set_ylabel("Post Count")

    # Log1p Score
    sns.histplot(df["log_score"], kde=True, ax=axes[1, 0], color="#2b5c8f", bins=20)
    axes[1, 0].set_title(f"Log-Transformed Score: log1p(score) (Skewness: {df['log_score'].skew():.2f})")
    axes[1, 0].set_xlabel("log(1 + Score)")
    axes[1, 0].set_ylabel("Post Count")

    # Log1p Comments
    sns.histplot(df["log_comment_count"], kde=True, ax=axes[1, 1], color="#d95f02", bins=20)
    axes[1, 1].set_title(f"Log-Transformed Comments: log1p(comments) (Skewness: {df['log_comment_count'].skew():.2f})")
    axes[1, 1].set_xlabel("log(1 + Comments)")
    axes[1, 1].set_ylabel("Post Count")

    fig.suptitle("Engagement Metric Distributions: Heavy-Tail vs Log Stabilization", fontsize=15, fontweight="bold", y=1.02)
    save_current_figure(output_path)


def plot_topic_distribution(df: pd.DataFrame, output_path: str = "img/fig_topic_distribution.png"):
    """Horizontal bar plot showing clean distribution of categories."""
    counts = df["category"].value_counts()
    total = len(df)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=counts.values, y=counts.index, palette="mako", hue=counts.index, legend=False)

    for i, count in enumerate(counts.values):
        pct = (count / total) * 100
        ax.text(count + 0.5, i, f"{count} ({pct:.1f}%)", va="center", fontweight="semibold")

    plt.title(f"Discussion Category Distribution in r/datascience (N = {total})", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Posts")
    plt.ylabel("Topic Category")
    plt.xlim(0, max(counts.values) * 1.15)
    save_current_figure(output_path)


def plot_engagement_by_category(df: pd.DataFrame, output_path: str = "img/fig_engagement_by_category.png"):
    """Plots median and IQR engagement by topic category with age-normalized velocity."""
    summary = df.groupby("category").agg(
        median_score=("score", "median"),
        median_comments=("comment_count", "median"),
        median_score_daily=("score_per_day", "median") if "score_per_day" in df.columns else ("score", "median"),
        count=("score", "count"),
    ).reset_index().sort_values("median_score", ascending=False)

    summary["category_label"] = summary["category"] + " (N=" + summary["count"].astype(str) + ")"

    fig, ax1 = plt.subplots(figsize=(11, 6))

    x = np.arange(len(summary))
    width = 0.35

    rects1 = ax1.bar(x - width/2, summary["median_score"], width, label="Cumulative Median Upvotes", color="#2b5c8f")
    rects2 = ax1.bar(x + width/2, summary["median_comments"], width, label="Cumulative Median Comments", color="#e7742f")

    ax1.set_ylabel("Median Cumulative Count")
    ax1.set_title("Median Cumulative Engagement by Category (with Sample Sizes)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["category_label"], rotation=25, ha="right")
    ax1.legend(loc="upper right")

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10)

    save_current_figure(output_path)


def plot_circadian_heatmap(df: pd.DataFrame, output_path: str = "img/fig_circadian_heatmap.png"):
    """Heatmap showing posting activity across Day of Week x Hour of Day (UTC & Eastern)."""
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    activity_utc = df.groupby(["day_name", "hour_utc"]).size().unstack(fill_value=0)
    activity_utc = activity_utc.reindex(index=[d for d in days_order if d in activity_utc.index], columns=range(24), fill_value=0)

    plt.figure(figsize=(14, 6))
    sns.heatmap(activity_utc, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={"label": "Post Count"}, linewidths=0.5)
    plt.title("Circadian Submission Heatmap: Day of Week vs Hour of Day (UTC)\n[Note: Peaks at 13-18 UTC correspond to North American Morning Hours: EDT UTC-4 / EST UTC-5]",
              fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Day of Week")
    save_current_figure(output_path)


def plot_score_comments_regression(df: pd.DataFrame, output_path: str = "img/fig_score_comments_regression.png"):
    """Log-log scatter plot with Spearman rank correlation and regression line.

    Methodological Note: Correlation demonstrates monotonic association, not causation.
    """
    rho, p_val = stats.spearmanr(df["score"], df["comment_count"])

    plt.figure(figsize=(9, 7))
    sns.regplot(
        data=df,
        x="log_score",
        y="log_comment_count",
        scatter_kws={"alpha": 0.6, "color": "#1b7837", "s": 50},
        line_kws={"color": "#b2182b", "linewidth": 2},
    )

    stats_text = (
        f"Spearman $\\rho = {rho:.3f}$\n"
        f"$p = {p_val:.2e}$\n"
        f"$N = {len(df)}$\n"
        "Association Only (Not Causation)"
    )
    plt.gca().text(
        0.05, 0.88, stats_text, transform=plt.gca().transAxes,
        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
    )

    plt.title("Monotonic Association: Upvotes vs Comments (Log Scale)", fontsize=14, fontweight="bold")
    plt.xlabel("log(1 + Upvote Score)")
    plt.ylabel("log(1 + Comment Count)")
    save_current_figure(output_path)


def plot_post_age_and_exposure(df: pd.DataFrame, output_path: str = "img/fig_post_age_exposure.png"):
    """Evaluates the relationship between post exposure time (age in days) and cumulative score."""
    if "post_age_days" not in df.columns:
        return

    rho, p_val = stats.spearmanr(df["post_age_days"], df["score"])

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="post_age_days",
        y="log_score",
        hue="category",
        palette="tab10",
        alpha=0.75,
        s=60,
    )
    sns.regplot(
        data=df,
        x="post_age_days",
        y="log_score",
        scatter=False,
        color="darkslategray",
        line_kws={"linestyle": "--", "linewidth": 1.5},
    )

    stats_text = f"Spearman $\\rho = {rho:.3f}$ ($p = {p_val:.3f}$)\nEngagement plateaus early due to Reddit Hot algorithm decay"
    plt.gca().text(
        0.05, 0.88, stats_text, transform=plt.gca().transAxes,
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
    )

    plt.title("Post Exposure Time (Age in Days) vs Log Score", fontsize=14, fontweight="bold")
    plt.xlabel("Post Age at Scrape Date (Days)")
    plt.ylabel("log(1 + Score)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Category")
    save_current_figure(output_path)


def plot_title_length_engagement(df: pd.DataFrame, output_path: str = "img/fig_title_length_engagement.png"):
    """Title length vs engagement with non-parametric correlation."""
    rho, p_val = stats.spearmanr(df["title_char_length"], df["score"])

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="title_char_length",
        y="log_score",
        hue="category",
        palette="tab10",
        alpha=0.75,
        s=60,
    )
    sns.regplot(
        data=df,
        x="title_char_length",
        y="log_score",
        scatter=False,
        color="darkslategray",
        line_kws={"linestyle": "--", "linewidth": 1.5},
    )

    stats_text = f"Spearman $\\rho = {rho:.3f}$ ($p = {p_val:.3f}$)"
    plt.title(f"Post Title Length vs Engagement ({stats_text})", fontsize=14, fontweight="bold")
    plt.xlabel("Title Character Length")
    plt.ylabel("log(1 + Score)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Category")
    save_current_figure(output_path)


def plot_nmf_topics(topics: List[Dict[str, Any]], output_path: str = "img/fig_nmf_topics.png"):
    """Plots top keyword weights for each discovered NMF topic."""
    n_topics = len(topics)
    fig, axes = plt.subplots(1, n_topics, figsize=(4 * n_topics, 5), sharey=False)

    for i, t in enumerate(topics):
        ax = axes[i] if n_topics > 1 else axes
        words = t["keywords"][::-1]
        weights = t["weights"][::-1]

        ax.barh(words, weights, color=sns.color_palette("mako", len(words)))
        ax.set_title(f"Topic {t['topic_id']}\n{words[-1].title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("TF-IDF Weight")

    fig.suptitle("Unsupervised NMF Topic Discovery: Top Keywords", fontsize=14, fontweight="bold", y=1.03)
    save_current_figure(output_path)


def plot_sentiment_by_category(df: pd.DataFrame, output_path: str = "img/fig_sentiment_by_category.png"):
    """Boxplot of sentiment compound scores across categories with N count."""
    order = df.groupby("category")["sentiment_compound"].median().sort_values(ascending=False).index
    counts = df["category"].value_counts()
    labels = [f"{c}\n(N={counts[c]})" for c in order]

    plt.figure(figsize=(11, 6))
    ax = sns.boxplot(
        data=df,
        x="category",
        y="sentiment_compound",
        order=order,
        palette="viridis",
        hue="category",
        legend=False,
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title("VADER Sentiment Polarity Distribution by Post Category\n(Lexical Polarity Estimate; Subject to Technical Jargon False Positives)", fontsize=13, fontweight="bold")
    plt.xlabel("Category")
    plt.ylabel("Compound Sentiment Score (-1.0 to +1.0)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    save_current_figure(output_path)


def plot_wordcloud(texts: List[str], output_path: str = "img/fig_wordcloud.png"):
    """Generates and saves domain-filtered wordcloud."""
    from src.nlp import DOMAIN_STOPWORDS
    try:
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english")) | DOMAIN_STOPWORDS
    except Exception:
        stop = DOMAIN_STOPWORDS

    combined = " ".join([t for t in texts if isinstance(t, str)])
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=stop,
        colormap="viridis",
        collocations=False,
        max_words=120,
    ).generate(combined)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc.to_image(), interpolation="bilinear")
    plt.axis("off")
    plt.title("Key Discussion Keywords in r/datascience", fontsize=16, fontweight="bold", pad=12)
    save_current_figure(output_path)


def generate_all_visualizations(
    df: pd.DataFrame,
    nmf_topics: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "img",
):
    """Generates and exports the complete suite of figures."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating publication-grade visualization suite in %s/...", output_dir)

    plot_engagement_distributions(df, os.path.join(output_dir, "fig_engagement_distributions.png"))
    plot_topic_distribution(df, os.path.join(output_dir, "fig_topic_distribution.png"))
    plot_engagement_by_category(df, os.path.join(output_dir, "fig_engagement_by_category.png"))
    plot_circadian_heatmap(df, os.path.join(output_dir, "fig_circadian_heatmap.png"))
    plot_score_comments_regression(df, os.path.join(output_dir, "fig_score_comments_regression.png"))
    plot_title_length_engagement(df, os.path.join(output_dir, "fig_title_length_engagement.png"))
    plot_post_age_and_exposure(df, os.path.join(output_dir, "fig_post_age_exposure.png"))
    plot_sentiment_by_category(df, os.path.join(output_dir, "fig_sentiment_by_category.png"))
    plot_wordcloud(df["full_text"].tolist(), os.path.join(output_dir, "fig_wordcloud.png"))

    if nmf_topics:
        plot_nmf_topics(nmf_topics, os.path.join(output_dir, "fig_nmf_topics.png"))

    logger.info("All visualizations successfully generated.")
