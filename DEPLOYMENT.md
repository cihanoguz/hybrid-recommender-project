# Deployment

## Hetzner Cloud

**Requirements:** Debian 13, 8GB RAM, 80GB SSD

**Deploy:**
```bash
# System setup
apt-get update && apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
systemctl start docker && systemctl enable docker
apt-get install docker-compose-plugin git -y

# Clone & run
git clone https://github.com/cihanoguz/hybrid-recommender-project.git
cd hybrid-recommender-project
docker compose up -d --build

# Verify
docker ps
docker compose logs -f
```

**Nginx (port 80):**
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

**Update:**
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

# 2. Update nginx.conf: Replace YOUR_IP_HERE with your IP
# 3. Reload nginx
nginx -t && systemctl reload nginx

# 4. Start Portainer (only accessible via nginx /portainer/)
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
