#!/bin/bash

# Linux Server Deployment Script
# Run this script on your Linux server

set -e

echo "🚀 Hybrid Recommender - Linux Server Deployment"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Docker check
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    echo "To install Docker: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

echo -e "${GREEN}✅ Docker found${NC}"
docker --version

# Docker Compose check
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker Compose not found, installing...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo -e "${GREEN}✅ Docker Compose found${NC}"
docker-compose --version

# Docker service check
if ! systemctl is-active --quiet docker; then
    echo -e "${YELLOW}⚠️  Docker service is not running, starting...${NC}"
    sudo systemctl start docker
    sudo systemctl enable docker
fi

echo -e "${GREEN}✅ Docker service is running${NC}"

# Project directory check
PROJECT_DIR=$(dirname "$(readlink -f "$0")")
cd "$PROJECT_DIR"

echo ""
echo "📦 Building Docker image..."
echo "   (This may take 5-10 minutes)"
echo ""

# Build
docker build -t hybrid-recommender:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image built successfully${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

# Stop existing container (if any)
echo ""
echo "🛑 Stopping existing container (if any)..."
docker-compose down 2>/dev/null || true

# Start new container
echo ""
echo "🚀 Starting container..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container started successfully${NC}"
else
    echo -e "${RED}❌ Container failed to start!${NC}"
    exit 1
fi

# Status check
sleep 3
echo ""
echo "📊 Container status:"
docker ps | grep hybrid-recommender || echo -e "${YELLOW}⚠️  Container is not running${NC}"

# Port check
echo ""
echo "🔍 Port check:"
if command -v netstat &> /dev/null; then
    netstat -tuln | grep 8080 && echo -e "${GREEN}✅ Port 8080 is listening${NC}" || echo -e "${YELLOW}⚠️  Port 8080 is not yet active${NC}"
fi

echo ""
echo -e "${GREEN}================================================"
echo "✅ Deployment completed!"
echo "================================================${NC}"
echo ""
echo "📍 Application address: http://$(hostname -I | awk '{print $1}'):8080"
echo "   OR: http://localhost:8080"
echo ""
echo "📝 Useful commands:"
echo "   View logs: docker logs -f hybrid-recommender"
echo "   Stop: docker-compose down"
echo "   Restart: docker-compose restart"
echo ""

