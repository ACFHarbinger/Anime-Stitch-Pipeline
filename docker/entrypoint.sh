#!/usr/bin/env bash
# Entrypoint for the app container: waits for the database, then execs the
# given command (defaults to the app's Dockerfile CMD).
set -euo pipefail

if [ -n "${DB_HOST:-}" ]; then
  echo "Waiting for database at ${DB_HOST}:${DB_PORT:-5432}..."
  until pg_isready -h "${DB_HOST}" -p "${DB_PORT:-5432}" -q; do
    sleep 1
  done
  echo "Database is ready."
fi

exec "$@"
