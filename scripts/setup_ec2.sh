#!/bin/bash
# scripts/setup_ec2.sh
#
# Deploys DemandSurge on a fresh AWS EC2 t2.micro (Ubuntu 22.04).
#
# Architecture:
#   - PostgreSQL  : Docker container (localhost:5432)
#   - FastAPI     : Python on host, managed by Supervisor
#   - Streamlit   : Python on host, managed by Supervisor
#   - Nginx       : Reverse proxy (port 80)
#
# Usage:
#   chmod +x scripts/setup_ec2.sh
#   demandsurge_REPO_URL=https://github.com/Ogeanni/demandsurge.git \
#     ./scripts/setup_ec2.sh

set -e
set -o pipefail

echo ""
echo "============================================================"
echo "  DemandSurge EC2 Deployment — Ubuntu 22.04 / t2.micro"
echo "  PostgreSQL in Docker"
echo "============================================================"
echo ""

# ── 1. Swap — t2.micro has 1GB RAM, Prophet training needs more ───────
echo "[1/10] Configuring 2GB swap..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  2GB swap created."
else
    echo "  Swap already exists."
fi

# ── 2. System packages ────────────────────────────────────────────────
echo ""
echo "[2/10] Installing system packages..."
sudo apt-get update -y -q
sudo apt-get install -y -q \
    python3.10 \
    python3.10-venv \
    python3-pip \
    docker.io \
    docker-compose \
    nginx \
    supervisor \
    git \
    curl \
    gcc \
    libpq-dev \
    postgresql-client

sudo usermod -aG docker ubuntu
echo "  Done."

# ── 3. Clone or update repository ────────────────────────────────────
echo ""
echo "[3/10] Setting up repository..."
cd /home/ubuntu

REPO_URL="${DEMANDSURGE_REPO_URL:-https://github.com/Ogeanni/demandsurge.git}"

if [ -d "demandsurge" ]; then
    echo "  Repo found — pulling latest..."
    cd demandsurge && git pull origin main
else
    git clone "$REPO_URL" demandsurge
    cd demandsurge
fi

# ── 4. Python virtual environment ────────────────────────────────────
echo ""
echo "[4/10] Setting up Python virtual environment..."
[ ! -d "venv" ] && python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."

# ── 5. Environment file ───────────────────────────────────────────────
echo ""
echo "[5/10] Checking .env..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_EC2_IP")
    # Auto-fill API_BASE_URL with the real EC2 IP
    sed -i "s|API_BASE_URL=http://YOUR_EC2_PUBLIC_IP/api|API_BASE_URL=http://${EC2_IP}/api|g" .env
    echo ""
    echo "  .env created. You must set:"
    echo "    OPENAI_API_KEY=sk-..."
    echo "  in /home/ubuntu/demandsurge/.env before continuing."
    echo ""
    read -rp "  Press Enter after editing .env..." _
fi

source .env

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    echo "  ERROR: OPENAI_API_KEY not set. Edit .env and re-run."
    exit 1
fi

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "  ERROR: POSTGRES_PASSWORD not set. Edit .env and re-run."
    exit 1
fi

if [ -z "$POSTGRES_USER" ]; then
    echo "  ERROR: POSTGRES_USER not set. Edit .env and re-run."
    exit 1
fi

if [ -z "$POSTGRES_DB" ]; then
    echo "  ERROR: POSTGRES_DB not set. Edit .env and re-run."
    exit 1
fi

echo "  .env validated."

# ── 6. Start PostgreSQL container ─────────────────────────────────────
echo ""
echo "[6/10] Starting PostgreSQL container..."
sudo systemctl enable docker
sudo systemctl start docker

# Use newgrp to apply docker group without logout
sudo docker-compose -f docker-compose.db.yml up -d

echo "  Waiting for Postgres..."
until sudo docker exec demandsurge_postgres pg_isready \
    -U "$POSTGRES_USER" -q 2>/dev/null; do
    printf '.'
    sleep 2
done
echo ""
echo "  PostgreSQL ready."

# ── 7. Database schema ────────────────────────────────────────────────
echo ""
echo "[7/10] Applying database schema..."
PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h 127.0.0.1 \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -f db/schema.sql

PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h 127.0.0.1 \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -c "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY category;"

# ── 8. ML pipeline ────────────────────────────────────────────────────
echo ""
echo "[8/10] Training ML pipeline (~10-15 mins on t2.micro)..."
mkdir -p data src models

echo "  [8a] Feature matrix..."
python src/features.py

echo "  [8b] Prophet demand models..."
python src/demand_forecast.py

echo "  [8c] XGBoost pricing model..."
python src/pricing_model.py

echo "  ML training complete."

# ── 9. Supervisor ─────────────────────────────────────────────────────
echo ""
echo "[9/10] Configuring Supervisor..."

# Build KEY="VALUE",... string from .env for Supervisor's environment= directive
# Supervisor does not source .env — vars must be injected explicitly
ENV_STRING=$(grep -v '^#' /home/ubuntu/demandsurge/.env \
    | grep -v '^[[:space:]]*$' \
    | grep '=' \
    | while IFS='=' read -r key val; do
        val=$(echo "$val" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
        printf '%s="%s",' "$key" "$val"
      done)
ENV_STRING="${ENV_STRING%,}"   # Strip trailing comma

sudo tee /etc/supervisor/conf.d/demandsurge_app.conf > /dev/null << SUPEOF
[program:demandsurge_app]
command=/home/ubuntu/demandsurge/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
directory=/home/ubuntu/demandsurge
user=ubuntu
autostart=true
autorestart=true
startretries=3
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/demandsurge_app.log
stderr_logfile=/var/log/demandsurge_app.err.log
stdout_logfile_maxbytes=10MB
environment=HOME="/home/ubuntu",USER="ubuntu",$ENV_STRING
SUPEOF

sudo tee /etc/supervisor/conf.d/demandsurge_ui.conf > /dev/null << SUPEOF
[program:demandsurge_ui]
command=/home/ubuntu/demandsurge/venv/bin/streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
directory=/home/ubuntu/demandsurge
user=ubuntu
autostart=true
autorestart=true
startretries=3
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/demandsurge_ui.log
stderr_logfile=/var/log/demandsurge_ui.err.log
stdout_logfile_maxbytes=10MB
environment=HOME="/home/ubuntu",USER="ubuntu",$ENV_STRING
SUPEOF

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start demandsurge_app  2>/dev/null || sudo supervisorctl restart demandsurge_app
sudo supervisorctl start demandsurge_ui   2>/dev/null || sudo supervisorctl restart demandsurge_ui
echo "  Services started."

# ── 10. Nginx ─────────────────────────────────────────────────────────
echo ""
echo "[10/10] Configuring Nginx..."

if [ ! -f "nginx/demandsurge.conf" ]; then
    echo "  ERROR: nginx/demandsurge.conf not found."
    echo "  Ensure it exists in your repo at nginx/demandsurge.conf"
    exit 1
fi

sudo cp nginx/demandsurge.conf /etc/nginx/sites-available/demandsurge
sudo ln -sf /etc/nginx/sites-available/demandsurge /etc/nginx/sites-enabled/demandsurge
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
echo "  Nginx ready."

# ── Final summary ─────────────────────────────────────────────────────
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_EC2_IP")

echo ""
echo "============================================================"
echo "  DemandSurge Deployed!"
echo "============================================================"
echo ""
echo "  Streamlit UI : http://${EC2_IP}"
echo "  FastAPI      : http://${EC2_IP}/api"
echo "  Swagger docs : http://${EC2_IP}/api/docs"
echo ""
echo "  Status:"
sudo supervisorctl status
echo ""
echo "  Logs:"
echo "    sudo tail -f /var/log/demandsurge_app.log"
echo "    sudo tail -f /var/log/demandsurge_ui.log"
echo "    sudo docker-compose -f docker-compose.db.yml logs postgres"
echo ""
echo "  Manage services:"
echo "    sudo supervisorctl restart demandsurge_app"
echo "    sudo supervisorctl restart demandsurge_ui"
echo "    sudo docker-compose -f docker-compose.db.yml restart"
echo ""