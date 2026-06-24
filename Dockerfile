FROM python:3.11-slim

WORKDIR /app

# System deps for lxml / pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (lighter demo set — no docling)
COPY requirements.demo.txt .
RUN pip install --no-cache-dir -r requirements.demo.txt

# Pre-download the embedding model so first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source
COPY src/ ./src/
COPY demo.html .
COPY presentation.html .
COPY .env.example .

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
