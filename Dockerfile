# ============================================================
# DataMind AI - Container Image
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8501

# Expose Streamlit port
EXPOSE 8501

# Run DataMind AI
CMD ["streamlit", "run", "src/main/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
