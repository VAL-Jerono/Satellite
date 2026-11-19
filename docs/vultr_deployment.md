## Vultr Deployment Guide — Kenya Land Readiness Atlas
### Using Vultr Cloud Compute with Docker

---

### 1. Create a Vultr Instance

**Recommended spec for $100 credits:**

| Component | Recommendation |
|---|---|
| Instance type | **Cloud Compute — Shared CPU** |
| Plan | **2 vCPU / 4 GB RAM** (~\$24/mo — your credits cover ~4 months) |
| OS | **Ubuntu 24.04 LTS** |
| Region | **Frankfurt** or **Singapore** (closest to Kenya) |
| SSH Key | Add your Mac public key |

---

### 2. First-time server setup (SSH in)

```bash
ssh root@YOUR_VULTR_IP

# Install Docker
curl -fsSL https://get.docker.com | bash
systemctl enable docker && systemctl start docker

# Install Docker Compose
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
     -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create app directory
mkdir -p /opt/land-atlas && cd /opt/land-atlas
```

---

### 3. Deploy from GitHub

```bash
cd /opt/land-atlas
git clone https://github.com/VAL-Jerono/Satellite.git .

# Upload your local model files (run this FROM YOUR MAC)
scp -r models/*.pkl root@YOUR_VULTR_IP:/opt/land-atlas/models/
scp -r data/atlas/ root@YOUR_VULTR_IP:/opt/land-atlas/data/
scp -r data/county_summaries/ root@YOUR_VULTR_IP:/opt/land-atlas/data/

# Back on the server — set env vars
cp .env.example .env
nano .env   # fill in GEMINI_API_KEY if you have one

# Build & launch
docker-compose up -d --build
docker-compose ps   # verify all services are running
```

---

### 4. Set up Nginx reverse proxy (clean URLs)

```bash
apt-get install -y nginx

cat > /etc/nginx/sites-available/land-atlas << 'EOF'
server {
    listen 80;
    server_name YOUR_VULTR_IP;

    # API
    location /api/ {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Streamlit (websocket-aware)
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

ln -s /etc/nginx/sites-available/land-atlas /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

**Your atlas is live at:** `http://YOUR_VULTR_IP`

---

### 5. Auto-restart on reboot

```bash
# Docker Compose already uses `restart: unless-stopped`
# This systemd service handles the initial boot
cat > /etc/systemd/system/land-atlas.service << 'EOF'
[Unit]
Description=Kenya Land Readiness Atlas
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/opt/land-atlas
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable land-atlas
```

---

### 6. Credit estimate breakdown

| Item | Cost/mo | Months on \$100 |
|---|---|---|
| 2vCPU/4GB instance | \$24 | **~4 months** |
| 1vCPU/2GB (minimum) | \$12 | **~8 months** |
| Bandwidth (500 GB/mo) | Included | — |
| Block storage (optional) | \$1/10GB | — |

> **Recommendation:** Use the 2vCPU/4GB plan. LightGBM inference is fast (< 50 ms/request) but Docker needs ~1.5 GB overhead. The 2 GB plan will OOM during build.

---

### 7. Useful Vultr commands

```bash
# Check status
docker-compose ps
docker-compose logs api --tail=50
docker-compose logs frontend --tail=50

# Update after a code push
git pull origin main
docker-compose up -d --build

# Check resource usage
docker stats
```

