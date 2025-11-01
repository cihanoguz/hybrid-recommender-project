# 🚀 Linux Server Quick Start

## Summary: Yes, It Works on Linux Server! ✅

The current Dockerfile is already **Linux-based** (Debian/Ubuntu). It will work directly on Linux server.

---

## ⚡ Quick Deployment (3 Steps)

### 1️⃣ Transfer Files to Linux Server

```bash
# Using SCP (from Mac/Windows)
scp -r hybrid-recommender-project user@linux-server:/opt/

# On Linux server
cd /opt/hybrid-recommender-project
```

### 2️⃣ Run Automatic Deployment Script

```bash
cd /opt/hybrid-recommender-project
chmod +x deploy-to-linux.sh
sudo ./deploy-to-linux.sh
```

**OR** Manually:

```bash
docker build -t hybrid-recommender:latest .
docker-compose up -d
```

### 3️⃣ Access

```bash
# Using server IP
http://SERVER_IP:8080

# Or from localhost
http://localhost:8080
```

---

## 📋 Requirements

- **Linux Server** (Ubuntu, Debian, CentOS, etc.)
- **Docker** (script installs automatically or manually: `curl -fsSL https://get.docker.com | sh`)
- **Minimum 4GB RAM** (recommended: 8GB)
- **Minimum 15GB Disk** (for image + data)

---

## 🔍 Architecture Compatibility

**Current status:**
- Image built on Mac: `arm64/linux` (Apple Silicon)

**For Linux server:**
- **AMD64/x86_64 server:** Rebuild on Linux server → Automatically becomes `amd64/linux` ✅
- **ARM64 server:** Current image works directly ✅

**Docker automatically builds according to the server's architecture.**

---

## 🎯 Extra Steps for Production

### Firewall

```bash
sudo ufw allow 8080/tcp
```

### Nginx Reverse Proxy

```bash
sudo apt install nginx
# Nginx config: see LINUX_DEPLOYMENT.md file
```

### Auto-start (Systemd)

```bash
sudo systemctl enable docker
# Container starts automatically (restart: unless-stopped in docker-compose.yml)
```

---

## ❓ Having Issues?

1. **Check logs:** `docker logs -f hybrid-recommender`
2. **Container status:** `docker ps`
3. **Detailed guide:** See `LINUX_DEPLOYMENT.md` file

---

## ✅ Summary

- ✅ **Works on Linux** (Docker is Linux native)
- ✅ **Architecture:** Automatically compatible when built on server
- ✅ **Dependencies:** All resolved within Docker
- ✅ **Production-ready:** Nginx, SSL, monitoring can be added

**When you build on Linux server, image will be created according to that server's architecture. No issues!**
