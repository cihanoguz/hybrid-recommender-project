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
COPY download_data.py .
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/
COPY data_loader/ ./data_loader/
COPY recommenders/ ./recommenders/
COPY ui/ ./ui/

# Create startup script to handle dynamic PORT (Render.com sets PORT env var)
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'PORT=${PORT:-8080}' >> /app/start.sh && \
    echo 'exec streamlit run app.py \' >> /app/start.sh && \
    echo '    --server.port=$PORT \' >> /app/start.sh && \
    echo '    --server.address=0.0.0.0 \' >> /app/start.sh && \
    echo '    --server.headless=true \' >> /app/start.sh && \
    echo '    --server.enableCORS=false \' >> /app/start.sh && \
    echo '    --server.enableXsrfProtection=false' >> /app/start.sh && \
    chmod +x /app/start.sh

# Expose port (default 8080, Render sets PORT dynamically)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket, os; s=socket.socket(); port=int(os.getenv('PORT', 8080)); s.connect(('localhost', port)); s.close()" || exit 1

# Start Streamlit application
CMD ["/app/start.sh"]

