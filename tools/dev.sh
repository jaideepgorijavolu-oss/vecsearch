#!/usr/bin/env bash
# Run a command inside the vecsearch-dev container (source at /work, build trees in a volume at /build).
# Usage: tools/dev.sh "cmake --preset release && ..."
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 docker run --rm ${DEV_DOCKER_FLAGS:-} -v "$here:/work" -v vecsearch-build:/work/build -v vecsearch-data:/data -e VECSEARCH_DATA=/data \
  -w /work vecsearch-dev bash -lc "$*"
