#!/bin/bash

# Docker image build script for Debian Linux (amd64)

set -e

echo "🐧 Debian Linux (amd64) Docker Image Build"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker check
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found${NC}"

# Buildx check
if ! docker buildx version &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker buildx not found${NC}"
    echo "If you're using Docker Desktop, buildx comes automatically."
    exit 1
fi

echo -e "${GREEN}✅ Docker buildx found${NC}"

# Stop existing container (if any)
echo ""
echo -e "${BLUE}🛑 Stopping existing container (if any)...${NC}"
docker-compose down 2>/dev/null || true

# Clean old images (optional)
read -p "Do you want to remove old images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi hybrid-recommender:latest 2>/dev/null || true
    docker rmi hybrid-recommender-project-hybrid-recommender:latest 2>/dev/null || true
    echo -e "${GREEN}✅ Old images cleaned${NC}"
fi

echo ""
echo -e "${BLUE}📦 Building image for Debian Linux (amd64)...${NC}"
echo "   (This may take 5-10 minutes)"
echo ""

# Build for amd64 platform using buildx
cd "$(dirname "$0")"

docker buildx build \
    --platform linux/amd64 \
    --tag hybrid-recommender:latest \
    --load \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Image built successfully!${NC}"
    
    # Show image information
    echo ""
    echo -e "${BLUE}📊 Image information:${NC}"
    docker inspect hybrid-recommender:latest --format='Platform: {{.Architecture}}/{{.Os}}'
    docker images hybrid-recommender:latest --format 'Size: {{.Size}}'
    
    echo ""
    echo -e "${YELLOW}Next step: docker-compose up -d${NC}"
    read -p "Do you want to start the container now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "${BLUE}🚀 Starting container...${NC}"
        docker-compose up -d
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Container started successfully${NC}"
            sleep 3
            docker ps | grep hybrid-recommender
            echo ""
            echo -e "${GREEN}📍 Application: http://localhost:8080${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Process completed!"
echo "==========================================${NC}"

