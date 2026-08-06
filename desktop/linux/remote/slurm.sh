#!/usr/bin/env bash
# Example SLURM batch submission for a long-running job on a remote cluster.
#SBATCH --job-name=app-job
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

(cd python && uv run python -m src.main "$@")
