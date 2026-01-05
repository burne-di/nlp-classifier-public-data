FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .
COPY README.md .


COPY packages/ packages/
COPY services/ services/
COPY scripts/ scripts/
COPY data/processed/ data/processed/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy model (if exists) - this will be overridden in docker-compose
COPY models/ models/

# Set Python path
ENV PYTHONPATH=/app/packages

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
