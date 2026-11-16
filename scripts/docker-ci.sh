#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t retrieval-strategies:ci "$ROOT"
docker run --rm retrieval-strategies:ci evaluate-pipeline --profile ci-retrieval-eval