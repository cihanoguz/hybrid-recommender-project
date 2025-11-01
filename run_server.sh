#!/bin/bash

# Hybrid Recommender Project - Streamlit Server Startup Script

cd "$(dirname "$0")"

echo "🚀 Starting Streamlit application..."
echo "📍 Address: http://localhost:8080"
echo ""
echo "To stop: Ctrl+C"
echo ""

# Use Anaconda Streamlit
/opt/anaconda3/bin/streamlit run app.py \
    --server.port 8080 \
    --server.address localhost \
    --server.headless true

