# AlphaEdge Local Development Setup

## Services Running Locally via Docker

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **ChromaDB** | 8001 | http://localhost:8001 | Vector database for SEC filings |
| **Redis** | 6379 | redis://localhost:6379 | Caching and rate limiting |
| **Jaeger** | 16686 | http://localhost:16686 | Distributed tracing UI |
| **OTLP Collector** | 4317/4318 | localhost:4317 | OpenTelemetry trace ingestion |

## Quick Start

```bash
# Start local services
cd /Users/krkaushikkumar/Desktop/Fin/alphaedge
docker compose -f docker-compose.local.yml up -d

# Verify services
docker ps

# Start the API (in a terminal)
source ../venv/bin/activate
uvicorn src.api.main:app --reload --port 8000

# Start the Streamlit UI (in another terminal)  
source ../venv/bin/activate
streamlit run frontend/app.py
```

## Service URLs

- **API**: http://localhost:8000
  - Health: http://localhost:8000/health
  - Metrics: http://localhost:8000/metrics
  - Query: POST http://localhost:8000/query
  
- **Streamlit UI**: http://localhost:8501

- **Jaeger Tracing**: http://localhost:16686
  - View traces for API requests
  - Monitor query performance
  - Debug slow requests

- **ChromaDB**: http://localhost:8001
  - Vector store API
  - Heartbeat: http://localhost:8001/api/v1/heartbeat

## What's Verified ✓

### OpenBB (Financial Data)
- ✅ Stock quotes (real-time prices)
- ✅ Fundamental metrics (P/E, Market Cap)
- ✅ Historical data
- ✅ Company news

### MLX-LM (Local LLM)
- ✅ Qwen2.5-3B-Instruct-4bit running on M1
- ✅ ~3s model load time (cached after first load)
- ✅ ~10-30s inference time depending on query

### OpenTelemetry
- ✅ Traces exported to Jaeger
- ✅ Request metrics tracked
- ✅ API instrumented with FastAPI middleware

### Docker Services
- ✅ ChromaDB v0.5.23 (persistent vector store)
- ✅ Redis 7 Alpine (caching)
- ✅ Jaeger All-in-One (tracing)

## Configuration

Environment variables in `.env`:

```env
# Local services
CHROMA_HOST=localhost
CHROMA_PORT=8001
REDIS_URL=redis://localhost:6379

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=alphaedge
```

## Full Production Stack

For production deployment with all observability:

```bash
docker compose up -d
```

This includes:
- API container
- Frontend container  
- ChromaDB
- Redis
- OpenTelemetry Collector
- Jaeger
- Prometheus
- Grafana

## Stopping Services

```bash
# Stop local dev services
docker compose -f docker-compose.local.yml down

# Stop and remove volumes (clean slate)
docker compose -f docker-compose.local.yml down -v
```
