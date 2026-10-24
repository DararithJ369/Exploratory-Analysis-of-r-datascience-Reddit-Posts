# Multi-stage Docker build using Astral UV for ultra-fast package installation
# Stage 1: Get UV binary
FROM ghcr.io/astral-sh/uv:latest AS uv_installer

# Stage 2: Main Python runtime
FROM python:3.11-slim

LABEL maintainer="PDAS Team"
LABEL description="Containerized Reddit Data Science Scraping & Analysis Workspace"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NLTK_DATA=/workspace/data/nltk_data \
    UV_SYSTEM_PYTHON=1

# Copy uv and uvx binaries from installer stage
COPY --from=uv_installer /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Copy requirements file first for layer caching
COPY requirements.txt pyproject.toml ./

# Install python dependencies via UV at ultra-fast speeds
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download NLTK data to ensure 100% offline self-sufficiency
RUN mkdir -p /workspace/data/nltk_data && \
    python -c "import nltk; nltk.download('vader_lexicon', download_dir='/workspace/data/nltk_data'); nltk.download('stopwords', download_dir='/workspace/data/nltk_data'); nltk.download('punkt', download_dir='/workspace/data/nltk_data'); nltk.download('punkt_tab', download_dir='/workspace/data/nltk_data')"

# Copy project files
COPY . /workspace

# Ensure entrypoint is executable
RUN chmod +x /workspace/entrypoint.sh

# Expose JupyterLab port
EXPOSE 8888

# Set entrypoint and default command
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["pipeline"]
