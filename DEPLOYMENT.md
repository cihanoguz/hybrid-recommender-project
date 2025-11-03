## Deployment Guide

### Hetzner Cloud (Recommended)

**Server specs:**
- OS: Debian 13
- RAM: 8GB (2GB minimum)
- Storage: 80GB SSD
- Primary IP: Optional (for static IP/domain)

**Quick deploy:**
```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker (official)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl start docker
systemctl enable docker

# Install docker-compose plugin
apt-get install docker-compose-plugin -y

# Clone and deploy
apt-get install git -y
git clone https://github.com/cihanoguz/hybrid-recommender-project.git
cd hybrid-recommender-project
docker compose up -d --build

# Check logs
docker compose logs -f

# Verify
docker ps
```

**Access:**
- App: `http://YOUR_SERVER_IP:8080` (direct)
- App: `http://YOUR_SERVER_IP` (port 80, requires nginx - see below)

**Port 80 access (nginx reverse proxy):**
```bash
# Install nginx
apt-get install nginx -y

# Copy config
cp nginx.conf /etc/nginx/sites-available/hybrid-recommender
ln -s /etc/nginx/sites-available/hybrid-recommender /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test and reload
nginx -t
systemctl reload nginx

# Firewall
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8080/tcp
ufw enable
```

**Update:**
```bash
cd hybrid-recommender-project
git pull
docker compose up -d --build
```

**Notes:**
- Data (643MB) baked into image during build (no runtime download)
- Build time: ~5-10 min (first time)
- Memory: ~1.3-1.6GB peak usage
- Auto-restart on reboot: `restart: unless-stopped` in docker-compose.yml

---

### Render.com (Alternative)

**Setup:**
- Runtime: Docker
- Plan: Starter ($7/month) - 2GB RAM required
- Free tier (512MB) insufficient

**Config:**
- Dockerfile auto-detected
- Port: Auto (uses `$PORT`)
- Data: Build-time download (baked into image)

**Env vars (optional):**
- `PICKLE_PATH`: `data/prepare_data_demo.pkl`
- `LOG_LEVEL`: `ERROR` (saves memory)
- `PYTHONUNBUFFERED`: `1`

**Memory issue:**
If "exceeded memory limit" → upgrade to Starter plan (2GB). Free tier too small for 643MB data + app overhead.

---

### Troubleshooting

**Docker not found:**
```bash
# Use official install script
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl start docker
```

**Port conflict:**
```bash
lsof -i :8080
# Change port in docker-compose.yml or kill process
```

**Container won't start:**
```bash
docker compose logs hybrid-recommender
docker ps -a
```

**Rebuild:**
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

### CI/CD & Container Management

**Portainer (Docker UI):**
```bash
# Start Portainer
docker compose -f docker-compose.portainer.yml up -d

# Access: http://YOUR_SERVER_IP:9000
# First login: create admin user
# Manage containers, images, logs via web UI
```

**CI/CD with GitHub Actions:**
- Auto-build Docker image on push
- Push to GitHub Container Registry (optional)
- Deploy to Hetzner via SSH (optional)

**Manual deploy workflow:**
```bash
# On server
cd hybrid-recommender-project
git pull
docker compose up -d --build
docker compose logs -f
```

**Portainer auto-update (optional):**
- Use watchtower or Portainer stacks
- Configure webhook for auto-deploy

---

### Domain Setup

1. Point A record to Primary IP (Hetzner)
2. Update nginx config: `server_name yourdomain.com;`
3. SSL: `certbot --nginx -d yourdomain.com`
4. Auto-renew: `certbot renew --dry-run`
