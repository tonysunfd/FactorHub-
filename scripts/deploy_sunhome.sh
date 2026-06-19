#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sunhome}"
REMOTE_DIR="${REMOTE_DIR:-~/apps/factorhub-v3}"

echo "==> Syncing project to ${REMOTE_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-rdagent' \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'log' \
  --exclude 'logs' \
  --exclude '*.db' \
  "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Building and starting docker service on ${REMOTE_HOST}"
ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR}/docker && DOCKER_BUILDKIT=1 docker compose build && docker compose up -d"

echo "==> Verifying deployment"
ssh "${REMOTE_HOST}" "docker ps --filter name=factorflow --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
ssh "${REMOTE_HOST}" "curl -fsS http://127.0.0.1:8000/health"

echo "==> Deployment complete"
