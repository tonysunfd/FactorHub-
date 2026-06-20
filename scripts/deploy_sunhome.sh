#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-physnowhere@100.73.5.96}"
REMOTE_DIR="${REMOTE_DIR:-~/apps/factorhub-v3}"
REMOTE_IDENTITY="${REMOTE_IDENTITY:-$HOME/.ssh/id_ed25519_sunhome}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new}"

if [[ -f "${REMOTE_IDENTITY}" ]]; then
  SSH_OPTS="${SSH_OPTS} -i ${REMOTE_IDENTITY}"
fi

echo "==> Syncing project to ${REMOTE_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  -e "ssh ${SSH_OPTS}" \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-rdagent' \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'log' \
  --exclude 'logs' \
  --exclude '*.db' \
  --exclude 'config/llm_config.json' \
  "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Building and starting docker service on ${REMOTE_HOST}"
ssh ${SSH_OPTS} "${REMOTE_HOST}" "cd ${REMOTE_DIR}/docker && DOCKER_BUILDKIT=1 docker compose build && docker compose up -d"

echo "==> Verifying deployment"
ssh ${SSH_OPTS} "${REMOTE_HOST}" "docker ps --filter name=factorflow --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
ssh ${SSH_OPTS} "${REMOTE_HOST}" "curl -fsS http://127.0.0.1:8000/health"

echo "==> Deployment complete"
