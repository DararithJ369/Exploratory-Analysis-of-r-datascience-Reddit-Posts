# Exploratory Analysis & Scraping of r/datascience Reddit Posts

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![UV Fast Installer](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=flat&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen?style=flat)](tests/)

A reproducible, publication-grade Exploratory Data Analysis (EDA) and Web Scraping pipeline examining thematic trends, engagement dynamics, circadian temporal patterns, and sentiment across submissions in the **r/datascience** subreddit (1.5M+ members).

Developed for **PDAS (Python for Data Analysis and Statistics)** — ITC Year 3, Semester 1.

---

## Key Features & Methodological Modernization

- **Production-Grade Modular Architecture**: Core data science engine cleanly decoupled into `src/` (`scraper.py`, `cleaning.py`, `nlp.py`, `visualization.py`).
- **Zero Data Loss Ingestion**: Preserves 100% of organic submissions (including title-only discussions and link posts) rather than discarding empty self-texts.
- **Word-Boundary Topic Categorization**: Implements strict regex word boundaries (`\b...\b`), eliminating substring false-positive traps (such as the letter `'r'` incorrectly matching words into `Learning`).
- **Unsupervised NMF Topic Modeling**: Uses Non-Negative Matrix Factorization over TF-IDF n-grams to discover latent community discourse themes without human labeler bias.
- **Robust Non-Parametric Statistics**: Addresses extreme Pareto/power-law skewness with log transformations (`log1p`), **Spearman's rank correlation** ($\rho$), and Median/IQR dispersion measures.
- **Dual-Mode Docker Automation**: Single-command headless execution (`docker compose up pipeline`) or interactive browser-based JupyterLab workspace (`docker compose up jupyter`).
- **Ultra-Fast Dependency Management via `uv`**: Pre-configured for Astral's `uv` for 10x–100x faster package installation and container builds.
- **100% Offline Self-Sufficiency**: NLTK lexicons and datasets are bundled and cached locally, ensuring full reproducibility in sandboxed or offline environments.

---

## Repository Architecture

```
Reddit-DataScience-Scraping&Analysis/
├── Dockerfile                      # Production Python 3.11 image with Astral UV & pre-baked NLTK
├── docker-compose.yml              # Services: 'pipeline' (batch execution), 'jupyter' & 'test'
├── entrypoint.sh                   # Flexible multi-mode container entrypoint
├── .dockerignore                   # Build context exclusions (.venv, .git, caches)
├── requirements.txt                # Pinned production dependencies
├── pyproject.toml                  # Modern UV / PEP 621 project configuration
├── main.py                         # End-to-end CLI orchestrator & summary reporter
├── reddit_datascience_analysis.ipynb # Fully pre-rendered, verified canonical analysis notebook
├── data/
│   ├── reddit_datascience.csv      # Raw baseline dataset (cached for offline fallback)
│   └── reddit_datascience_clean.csv# Cleaned & feature-engineered dataset
├── img/                            # Exported 300 DPI publication figures
├── src/                            # Modular domain logic
│   ├── __init__.py
│   ├── scraper.py                  # Resilient scraper with flairs, upvote ratio & backoff
│   ├── cleaning.py                 # Outlier handling, text harmonization & feature engineering
│   ├── nlp.py                      # Word-boundary categorization, VADER sentiment, NMF modeling
│   └── visualization.py            # High-res Seaborn/Matplotlib chart generators
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            # Unit tests for categorization, cleaning, and NLP
└── helper/
    └── reddit_scraper.py           # Legacy wrapper forwarding to src.scraper
```

---

## Quickstart with Docker (Recommended)

Ensure [Docker](https://docs.docker.com/get-docker/) is installed and running.

### 1. Run Automated Headless Pipeline
Runs scraping (with automatic offline fallback), cleaning, statistical modeling, NMF topic extraction, and exports all figures to `img/`:
```bash
docker compose up pipeline
```

### 2. Launch Interactive JupyterLab
Spins up a pre-configured JupyterLab workspace on port 8888:
```bash
docker compose up jupyter
```
Open **[http://localhost:8888](http://localhost:8888)** in your web browser.

### 3. Run Unit Tests Inside Container
```bash
docker compose up test
```

---

## Local Setup (Without Docker)

### Option A: Using `uv` (Ultra-Fast)
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies into your environment
uv pip install -r requirements.txt

# Run the complete pipeline
python main.py

# Launch the analysis notebook
jupyter lab
```

### Option B: Using Standard `pip`
```bash
pip install -r requirements.txt
python main.py
```

---

## Key Scientific Findings & Methodological Rigor

| Research Dimension | Statistical Observation | Methodological Insight |
| :--- | :--- | :--- |
| **Engagement Distribution** | `Score`: Mean 47.4, Median 33, Skewness 3.62<br>`Comments`: Mean 26.6, Median 17, Skewness 1.91 | Extreme right-skewed heavy tail; normalized effectively via $\log(1 + x)$ transforms. |
| **Topic Distribution** | `Other`: 40.0%, `Career`: 18.3%, `AI/LLM`: 13.0%, `Learning`: 11.3%, `Project`: 7.8%, `Salary`: 5.2% | Career and Generative AI/LLM applications dominate organic community inquiries. |
| **Engagement by Topic** | `Salary`: Median 51.5 upvotes, 29.5 comments<br>`Career`: Median 47.0 upvotes, 33.0 comments | Compensation and career transition topics attract the highest interaction per post (both cumulatively and on an age-normalized daily velocity basis). |
| **Metric Association** | Spearman's $\rho = 0.606$ ($p = 7.33 \times 10^{-13}$) | Moderately strong positive monotonic **association** (not causation): algorithmic front-page exposure and topic resonance drive co-movement. |
| **Circadian Activity** | Peak window: 13:00 – 18:00 UTC (08:00 – 13:00 Eastern) | Timezone-aware conversion via `zoneinfo` accurately distinguishes **EDT** (UTC-4) and **EST** (UTC-5) across seasonal transitions. |
| **Post Age & Exposure** | Spearman's $\rho = 0.020$ ($p = 0.824$) | No correlation between post age and score: interaction plateaus within 24–48 hours due to Reddit's exponential 'Hot' ranking half-life decay. |
| **Latent Topics (NMF)** | 5 distinct clusters discovered: Career, AI Agents, Models/Datasets, Books, and Interviews | Unsupervised clustering provides an essential data-driven cross-validation against rule-based categorization limits. |

### Methodological & Statistical Guardrails (Addressing Common Pitfalls)
1. **Association vs. Causation**: Spearman rank correlation ($\rho = 0.606$) measures monotonic co-movement, not direct causation. Higher scores do not "cause" comments; rather, algorithmic visibility and broadly resonant debate drive both metrics simultaneously.
2. **Collection Boundaries vs. Pipeline Retention**: We distinguish between retaining 100% of collected organic posts ($N=115$) and total platform coverage. The 7-page snapshot of Reddit's 'Hot' listing represents prominent, survivor-biased discussions rather than an exhaustive longitudinal census.
3. **Timezone Transitions**: Timestamps are parsed using standard `zoneinfo.ZoneInfo("America/New_York")`, dynamically adjusting for the transition between EDT (UTC-4) and EST (UTC-5) on November 2, 2025.
4. **Lexicon Sentiment Limitations**: VADER compound scores serve as *indicative lexical polarity estimates*. Technical jargon (*loss function*, *error handling*, *bias variance*) can depress scores in non-emotional technical utilities.
5. **Post Age & Exposure Control**: In addition to cumulative counts, we compute daily interaction velocity (`score_per_day`) to confirm that category engagement hierarchies are not artifacts of exposure time.

### Empirical Visualizations & Analytic Artifacts

Key figures generated by the modular visualization engine (`src/visualization.py`) and saved to `img/`:

#### 1. Engagement Dynamics & Non-Parametric Distribution
Log transformation addresses the power-law tail inherent in Reddit submission scoring, confirming Pareto-like community dynamics.

![Engagement Distributions](img/fig_engagement_distributions.png)

#### 2. Thematic Structure & Unsupervised NMF Topics
Rule-based word-bounded regex categorization cross-validated against unsupervised Non-Negative Matrix Factorization (NMF) topic extraction:

| Topic Category Distribution | Discovered Latent NMF Themes |
| :---: | :---: |
| ![Topic Distribution](img/fig_topic_distribution.png) | ![NMF Topics](img/fig_nmf_topics.png) |

#### 3. Temporal Circadian Rhythm & Metric Association
Circadian patterns adjusted for Eastern Daylight / Standard Time transitions, alongside monotonic score-comment co-movement:

| Submission Intensity Heatmap | Monotonic Association Regression |
| :---: | :---: |
| ![Circadian Heatmap](img/fig_circadian_heatmap.png) | ![Score vs Comments Regression](img/fig_score_comments_regression.png) |

#### 4. Post Age Decay & Exposure Control
Spearman analysis confirms interaction plateauing within 24–48 hours, eliminating age exposure bias:

![Post Age Exposure](img/fig_post_age_exposure.png)

#### 5. Domain-Aware Sentiment by Category
VADER sentiment distribution across thematic categories with explicit sample size reporting:

![Sentiment by Category](img/fig_sentiment_by_category.png)

---

## Unit Testing Suite

The project includes unit tests covering categorization boundary logic, data retention, bot filtering, daylight saving time handling, sentiment analysis, and scraper fallbacks:
```bash
python -m unittest discover tests -v
```
All 10 test cases run in `< 0.05s` with 100% pass rate.

---

## Academic Context & Ethical Disclosure

This project was conducted for educational and exploratory research as part of the **PDAS** curriculum at the **Institute of Technology of Cambodia (ITC)**. All data was collected in compliance with standard research web-scraping practices and is subject to Reddit's Terms of Service and community guidelines.
