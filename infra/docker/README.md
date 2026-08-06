# Docker

## Quick start

```bash
cp .env.example .env
docker compose -f infra/docker/docker-compose.yml up --build
```

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Multi-stage build for the Python module's service (extend for other modules as needed) |
| `docker-compose.yml` | Local dev stack: app + PostgreSQL + Redis |
| `docker-compose.prod.yml` | Production overrides (apply with `-f infra/docker/docker-compose.yml -f infra/docker/docker-compose.prod.yml`) |
| `entrypoint.sh` | Waits for the database before exec'ing the container command |

## Notes

- Build context is the **repository root**, not `infra/docker/` — the Dockerfile needs access to `python/` (and any other module it packages).
- `.dockerignore` lives at the repo root for the same reason.
