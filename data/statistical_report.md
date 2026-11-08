# Statistical & Methodological Analysis Report

## 1. Dataset & Collection Overview
- **Analyzed Submissions (N)**: 115 organic posts
- **Primary Subreddit**: r/datascience
- **Temporal Range**: 2025-10-31 13:11:19+00:00 to 2026-01-09 01:16:05+00:00

## 2. Descriptive & Non-Parametric Dispersion
| Metric | Mean | Std | Median | IQR | Max | Outliers (Tukey 1.5x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Score** | 47.4 | 63.5 | 33 | 44 | 462 | 8 (7.0%) |
| **Comment Count** | 26.6 | 25.7 | 17 | 25 | 130 | 8 (7.0%) |

## 3. Correlation & Hypothesis Testing
- **Spearman Rank Correlation (Score vs Comments)**: rho = 0.606 (p = 7.33e-13)
  - *Interpretation*: Moderately strong monotonic association driven by mutual algorithmic visibility.
- **Kruskal-Wallis H-Test (Score across Categories)**: H = 16.46 (p = 0.011)
- **Kruskal-Wallis H-Test (Comments across Categories)**: H = 16.10 (p = 0.013)

## 4. Topic Category Breakdown & Engagement Medians
| Category | N | Share (%) | Median Score | Median Comments | Positive Sentiment (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI/LLM** | 15 | 13.0% | 36.0 | 15.0 | 73.3% |
| **Advice** | 5 | 4.3% | 22.0 | 22.0 | 40.0% |
| **Career** | 21 | 18.3% | 47.0 | 33.0 | 61.9% |
| **Learning** | 13 | 11.3% | 28.0 | 12.0 | 61.5% |
| **Other** | 46 | 40.0% | 18.5 | 13.5 | 65.2% |
| **Project** | 9 | 7.8% | 5.0 | 10.0 | 77.8% |
| **Salary** | 6 | 5.2% | 51.5 | 29.5 | 83.3% |

## 5. Unsupervised NMF Latent Themes
- **Topic 1: job, experience, career**: `job, experience, career, getting, masters, feel, role, company`
- **Topic 2: agent, python, google**: `agent, python, google, jupyter, agents, https, code, learning`
- **Topic 3: models, datasets, clean**: `models, datasets, clean, model, better, real, used, conversations`
- **Topic 4: books, reilly, read**: `books, reilly, read, code, prep, reading, recommendations, best`
- **Topic 5: interviews, prepare, roles**: `interviews, prepare, roles, leetcode, questions, engineering, different, preparing`
