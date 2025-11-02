# Hybrid Recommender System - Dockerfile
# Creates a lightweight container using Python 3.11 slim image
# Optimized for Debian Linux (amd64/x86_64) platform

# Platform specification (Linux amd64 - Debian/Ubuntu)
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for numpy, scipy)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY config.py .
COPY utils.py .
COPY logging_config.py .
COPY error_handling.py .
COPY security_utils.py .
COPY performance_utils.py .
COPY data/ ./data/
COPY data_loader/ ./data_loader/
COPY recommenders/ ./recommenders/
COPY ui/ ./ui/

# Expose port for Streamlit
EXPOSE 8080

# Health check (optional - port check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8080)); s.close()" || exit 1

# Start Streamlit application
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

