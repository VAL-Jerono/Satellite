#!/bin/bash
# ─────────────────────────────────────────────────────────────
# deploy_vultr.sh  - Deploy Kenya Land Readiness Atlas to Vultr
# Usage: bash scripts/deploy_vultr.sh
# ─────────────────────────────────────────────────────────────
set -e

VULTR_IP="192.248.170.201"
REMOTE_USER="root"
REMOTE_DIR="/opt/land-atlas"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"

echo "🛰️  Kenya Land Readiness Atlas - Vultr Deployment"
echo "   Server: $VULTR_IP"
echo ""

# 1. Build React frontend locally
echo "📦 Building React frontend..."
cd web && npm run build && cd ..
echo "✅ Frontend built → web/dist/"

# 2. Sync files to Vultr (excluding heavy stuff)
echo "🚀 Syncing files to Vultr ($REMOTE_IP)..."
rsync -avz --progress \
  --exclude 'node_modules' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude 'mlruns' \
  --exclude '*.pdf' \
  --exclude '*.ipynb' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ./ $REMOTE_USER@$VULTR_IP:$REMOTE_DIR/

# 3. SSH in and restart containers
echo "🔄 Restarting containers on Vultr..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $REMOTE_USER@$VULTR_IP bash << 'ENDSSH'
  cd /opt/land-atlas
  
  # Install Docker if not present
  if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
  fi
  
  # Pull latest compose file and restart
  docker compose -f docker-compose.prod.yml down --remove-orphans
  docker compose -f docker-compose.prod.yml build --no-cache api
  docker compose -f docker-compose.prod.yml up -d
  
  echo "✅ Containers started:"
  docker compose -f docker-compose.prod.yml ps
ENDSSH

echo ""
echo "✅ Deployment complete!"
echo "   🌍 Frontend: http://$VULTR_IP"
echo "   ⚡ API:      http://$VULTR_IP/health"
echo "   📊 MLflow:   http://$VULTR_IP:5000"
