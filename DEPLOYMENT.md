# Deployment

## Hetzner Cloud

**Requirements:** Debian 13, 8GB RAM, 80GB SSD

### 1) Base setup (Docker + deploy user)
```bash
# System setup
apt-get update && apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
systemctl start docker && systemctl enable docker
apt-get install docker-compose-plugin git -y

# Create non-root deploy user
adduser deploy
usermod -aG docker deploy
usermod -aG sudo deploy

# (Optional) copy root SSH key to deploy user
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

### 2A) Recommended deploy (GHCR image, no git clone)

`docker-compose.ghcr.yml` uses prebuilt image from GHCR.

```bash
su - deploy
mkdir -p ~/hybrid-recommender-project
cd ~/hybrid-recommender-project

# Download compose file from repository
curl -fsSL https://raw.githubusercontent.com/cihanoguz/hybrid-recommender-project/main/docker-compose.ghcr.yml -o docker-compose.yml

# Pull and run
docker compose pull
docker compose up -d

# Verify
docker ps
docker compose logs -f
```

### 2B) Alternative deploy (build from source)

# Clone & run
git clone https://github.com/cihanoguz/hybrid-recommender-project.git
cd hybrid-recommender-project
docker compose up -d --build

# Verify
docker ps
docker compose logs -f
```

### 3) Nginx (port 80)
```bash
apt-get install nginx -y
cp nginx.conf /etc/nginx/sites-available/hybrid-recommender
ln -s /etc/nginx/sites-available/hybrid-recommender /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# Firewall
apt-get install ufw -y
ufw allow 80/tcp 443/tcp 8080/tcp 22/tcp
ufw enable
```

### 4) Update

**If using GHCR image:**
```bash
su - deploy
cd ~/hybrid-recommender-project
docker compose pull
docker compose up -d
```

**If using source clone:**
```bash
cd hybrid-recommender-project
git pull
docker compose up -d --build
```

**Access:**
- `http://YOUR_IP:8080` (direct)
- `http://YOUR_IP` (via nginx)

---

## Render.com

**Setup:**
- Runtime: Docker
- Plan: Starter (2GB RAM) - Free tier insufficient
- Port: Auto (uses `$PORT`)

**Env vars:**
- `PICKLE_PATH`: `data/prepare_data_demo.pkl`
- `LOG_LEVEL`: `ERROR`
- `PYTHONUNBUFFERED`: `1`

**Note:** Data (643MB) baked into image during build.

---

## Troubleshooting

**Rebuild:**
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

**Check logs:**
```bash
docker compose logs -f hybrid-recommender
docker ps -a
```

**Port conflict:**
```bash
lsof -i :8080
```

---

## Optional

**Portainer (Docker UI - secured):**
```bash
# 1. Get your IP
curl ifconfig.me

# 2. Update nginx.conf: Replace YOUR_IP_HERE with your IP, then:
nginx -t && systemctl reload nginx

# 3. Start Portainer (only accessible via nginx /portainer/)
su - deploy
cd ~/hybrid-recommender-project
curl -fsSL https://raw.githubusercontent.com/cihanoguz/hybrid-recommender-project/main/docker-compose.portainer.yml -o docker-compose.portainer.yml
docker compose -f docker-compose.portainer.yml up -d

# Access: http://YOUR_IP/portainer/
```

**Domain + SSL:**
```bash
# 1. Point A record to server IP
# 2. Update nginx.conf: server_name yourdomain.com;
# 3. Install certbot
apt-get install certbot python3-certbot-nginx -y
certbot --nginx -d yourdomain.com
```

---

## Security

See [SECURITY.md](SECURITY.md) for comprehensive security guide:
- SSH hardening
- Fail2ban setup
- SSL/TLS configuration
- Firewall rules
- Nginx security headers
- Automatic updates
- Log monitoring
