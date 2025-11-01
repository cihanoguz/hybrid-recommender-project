# Linux Server Deployment Guide

## ⚠️ Important Note

The current Docker image was built for **ARM64** (Apple Silicon). To run on Linux server:

### Option 1: Rebuild on Linux Server (RECOMMENDED)
Build directly on Linux server to create image for the correct architecture.

### Option 2: Multi-Platform Build
Build for both ARM64 and AMD64 using buildx.

---

## 🚀 Linux Server Deployment Steps

### Step 1: Transfer Project Files to Linux Server

```bash
# Transfer using SCP (example)
scp -r /Users/cihanoguz/Downloads/hybrid-recommender-project user@linux-server:/opt/

# OR using Git
git clone <repo-url> /opt/hybrid-recommender-project
cd /opt/hybrid-recommender-project
```

### Step 2: Install Docker on Linux Server (if not installed)

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose installation
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 3: Build Docker Image

```bash
cd /opt/hybrid-recommender-project

# Will be built according to Linux server's architecture (automatic)
docker build -t hybrid-recommender:latest .
```

**Build time:** ~5-10 minutes (first build)

### Step 4: Run Container

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# OR manually
docker run -d \
  --name hybrid-recommender \
  -p 8080:8080 \
  --restart unless-stopped \
  hybrid-recommender:latest
```

### Step 5: Check Status

```bash
# Container status
docker ps

# Check logs
docker logs -f hybrid-recommender

# Health check
docker inspect hybrid-recommender | grep -A 10 Health
```

### Step 6: Firewall Settings

```bash
# If using UFW
sudo ufw allow 8080/tcp
sudo ufw reload

# OR firewalld
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

---

## 🌐 Nginx Reverse Proxy (Recommended for Production)

### Nginx Installation

```bash
sudo apt update
sudo apt install nginx -y
```

### Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/hybrid-recommender
```

Add the following content:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # OR server_ip_address

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Activate configuration:

```bash
sudo ln -s /etc/nginx/sites-available/hybrid-recommender /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔒 HTTPS/SSL (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## 📊 Systemd Service (Optional - Container Auto-start)

```bash
sudo nano /etc/systemd/system/hybrid-recommender.service
```

Content:

```ini
[Unit]
Description=Hybrid Recommender Docker Container
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/hybrid-recommender-project
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Activate:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hybrid-recommender
sudo systemctl start hybrid-recommender
```

---

## 🔍 Troubleshooting

### Port already in use

```bash
# Check port
sudo netstat -tulpn | grep 8080

# Use different port
docker run -p 8081:8080 ...
```

### Memory error

```bash
# Check Docker logs
docker logs hybrid-recommender

# Set memory limit
docker run --memory="4g" ...
```

### Permission error

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

---

## 📈 Monitoring

### Monitor logs

```bash
# Real-time logs
docker logs -f hybrid-recommender

# Last 100 lines
docker logs --tail 100 hybrid-recommender
```

### Resource usage

```bash
# Container resource usage
docker stats hybrid-recommender

# Disk usage
docker system df
```

---

## 🔄 Update Process

```bash
cd /opt/hybrid-recommender-project

# Pull new code (if using git)
git pull

# Stop container
docker-compose down

# Build new image
docker build -t hybrid-recommender:latest .

# Restart
docker-compose up -d
```

---

## 📝 Summary Commands

```bash
# Build and start
docker-compose up -d --build

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Enter container (for debugging)
docker exec -it hybrid-recommender bash
```
