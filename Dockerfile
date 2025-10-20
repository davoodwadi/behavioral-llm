FROM ghcr.io/astral-sh/uv:python3.10-trixie

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ADD . /app

WORKDIR /app
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
