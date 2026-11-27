#!/bin/bash
set -eu
cd "$(dirname "$0")"
# docker-py reads DOCKER_HOST, while Docker Desktop selects its socket by context.
if [[ -z ${DOCKER_HOST:-} ]]; then
  DOCKER_HOST=$(docker context inspect --format '{{.Endpoints.docker.Host}}')
  export DOCKER_HOST
fi
if [[ $# -eq 0 ]]; then
  set -- --mode 5 --config qwen3vl4b.yaml
fi
exec ./llm_model_gen_v2 "$@"
