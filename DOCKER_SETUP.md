# Docker Setup - Hybrid Recommender Project

## 📋 Prerequisites

1. **Docker** must be installed (https://docs.docker.com/get-docker/)
2. **Docker Compose** must be installed (usually comes with Docker)

## 🚀 Quick Start

### Step 1: Build Docker Image
```bash
docker build -t hybrid-recommender:latest .
```

### Step 2: Run Container
```bash
docker run -d \
  --name hybrid-recommender \
  -p 8080:8080 \
  hybrid-recommender:latest
```

**OR** using Docker Compose:
```bash
docker-compose up -d
```

### Step 3: Access Application
Open in your browser: `http://localhost:8080`

## 📦 Docker Files Description

### `Dockerfile`
- Uses Python 3.11-slim base image (small size)
- Installs dependencies from requirements.txt
- Runs application on port 8080

### `.dockerignore`
- Excludes unnecessary files from build context
- Files like venv, __pycache__, .git are not included
- Increases build speed and reduces image size

### `docker-compose.yml`
- For easy container management
- Port mapping and volume settings
- Restart policy

## 🔧 Commands

### Stop container
```bash
docker stop hybrid-recommender
```

### Start container
```bash
docker start hybrid-recommender
```

### View logs
```bash
docker logs -f hybrid-recommender
```

### Delete container
```bash
docker rm -f hybrid-recommender
```

### Delete image
```bash
docker rmi hybrid-recommender:latest
```

## 📊 Notes

- **First build:** May take ~5-10 minutes (downloading dependencies)
- **Image size:** ~1.5-2GB (numpy, pandas, scikit-learn are large)
- **Memory:** At least 2GB RAM recommended
- **Data files:** Pickle files are included in the image

## 🌐 Production Deployment

For production use:
1. Add environment variables using `.env` file
2. Use reverse proxy (nginx)
3. Add HTTPS/SSL certificate
4. Configure log rotation
5. Add health check endpoint

## 🔍 Troubleshooting

### Port already in use
```bash
# Check port
lsof -i :8080

# Use different port
docker run -p 8081:8080 hybrid-recommender:latest
```

### Memory error
```bash
# Give Docker more memory (Docker Desktop Settings)
```

### Check logs
```bash
docker logs hybrid-recommender
```
