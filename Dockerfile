FROM python:3.12-slim

# - no .pyc files, logs appear immediately
# - NLTK_DATA tells NLTK where the stopwords/wordnet data lives
# - PORT is the port the app listens on (Render also uses 10000 by default)
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
NLTK_DATA=/app/nltk_data \
PORT=10000

WORKDIR /app
# 1. Install dependencies first. This layer is reused on later builds
#    as long as requirements.txt does not change.
copy requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Download the NLTK data once, while the image is built.
RUN python -m nltk.downloader -d /app/nltk_data stopwords wordnet

# 3. Copy only what the app needs at runtime.
COPY app.py .
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY saved_models ./saved_models

# 4. Do not run as root inside the container.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

# Shell form so that ${PORT} is expanded when the container starts.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
