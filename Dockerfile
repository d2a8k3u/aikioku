# Aikioku — Multi-stage Dockerfile
# Targets: server (FastAPI backend), dashboard (Next.js frontend)

# ============================================
# Stage: server-builder — Python dependencies
# ============================================
FROM python:3.11 AS server-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY apps/server/pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install -e ".[dev]" "numpy>=1.24,<2.0"

# ============================================
# Stage: server — FastAPI runtime
# ============================================
FROM python:3.11-slim AS server

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=server-builder /install /usr/local

COPY apps/server/pyproject.toml .
COPY apps/server/src/ src/
COPY apps/server/tests/ tests/

RUN mkdir -p /data/notes /data/kuzu /data/chroma /data/sqlite

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ============================================
# Stage: dashboard-deps — Node dependencies
# ============================================
FROM node:20-alpine AS dashboard-deps

WORKDIR /app

COPY apps/dashboard/package.json apps/dashboard/package-lock.json ./
RUN npm install --production

# ============================================
# Stage: dashboard-builder — Next.js build
# ============================================
FROM node:20-alpine AS dashboard-builder

WORKDIR /app

COPY --from=dashboard-deps /app/node_modules ./node_modules
COPY apps/dashboard/ ./
RUN npm run build

# ============================================
# Stage: dashboard — Next.js runtime
# ============================================
FROM node:20-alpine AS dashboard

WORKDIR /app

ENV NODE_ENV=production

COPY --from=dashboard-builder /app/.next ./.next
COPY --from=dashboard-builder /app/public ./public
COPY --from=dashboard-builder /app/package.json ./package.json
COPY --from=dashboard-builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["npm", "start"]


# ============================================
# Stage: server-dev — FastAPI with hot reload
# Source is bind-mounted at runtime; only deps live in the image.
# ============================================
FROM python:3.11 AS server-dev

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY apps/server/pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" "numpy>=1.24,<2.0"

RUN mkdir -p /data/notes /data/kuzu /data/chroma /data/sqlite

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src"]


# ============================================
# Stage: dashboard-dev — Next.js with fast refresh
# Source is bind-mounted at runtime; only node_modules live in the image.
# ============================================
FROM node:20-alpine AS dashboard-dev

WORKDIR /app

COPY apps/dashboard/package.json apps/dashboard/package-lock.json ./
RUN npm install

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
