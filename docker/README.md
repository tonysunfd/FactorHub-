# FactorFlow Docker Deployment Guide

## File Description

| File | Description |
|------|------|
| `Dockerfile` | Multi-stage build: frontend build + backend runtime |
| `docker-compose.yml` | Service orchestration configuration |

## Quick Start

```bash
# Navigate to docker directory
cd docker

# Build image
docker-compose build

# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

## Deploy To SunHome

Use the helper script from the project root:

```bash
./scripts/deploy_sunhome.sh
```

Optional environment variables:

```bash
REMOTE_HOST=sunhome
REMOTE_DIR=~/apps/factorhub-v3
```

## Access URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Architecture

```
┌─────────────────────────────────────┐
│         FastAPI Server (:8000)       │
├─────────────────────────────────────┤
│  /api/*     → Backend API routes     │
│  /docs      → API documentation      │
│  /assets/*  → Frontend static files  │
│  /*         → SPA (index.html)       │
└─────────────────────────────────────┘
```

## Directory Structure

```
docker/
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Service orchestration
└── README.md           # This file
```

## Environment Variables

Configure via `.env` file:

```env
# Service port
PORT=8000

# Timezone
TZ=Asia/Shanghai
```

## Notes

1. **Data Persistence**: `data/` directory is mounted as a volume
2. **TA-Lib**: Pre-installed in the Dockerfile
3. **Production**: Consider restricting CORS allowed origins
