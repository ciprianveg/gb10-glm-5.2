#!/usr/bin/env bash
# build.sh — Build the vllm-node-tf5-glm52-v18 image for 8× GB10 (Gilded Gnosis)
#
# Usage:
#   ./v18/build.sh                    # build + copy to all 7 workers
#   ./v18/build.sh --solo             # build only (no copy)
#   ./v18/build.sh --push             # build + push to registry (set REGISTRY env)
#
# Requires: local-inference-lab/blackwell-llm-docker cloned at ../blackwell-llm-docker
#           v18/Dockerfile adapted for aarch64/SM121
#
# Build time on GB10: ~30-40 min (cached base stages) or ~80-90 min (full rebuild)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BLACKWELL_DIR="${REPO_DIR}/../blackwell-llm-docker"

TAG="${TAG:-vllm-node-tf5-glm52-v18}"

SOLO=false
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --solo) SOLO=true; shift ;;
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$BLACKWELL_DIR" ]]; then
    echo "ERROR: blackwell-llm-docker not found at $BLACKWELL_DIR"
    echo "Clone it: git clone https://github.com/local-inference-lab/blackwell-llm-docker.git ../blackwell-llm-docker"
    echo "Then: cd ../blackwell-llm-docker && git checkout 7f3cbc6"
    exit 1
fi

# Copy adapted Dockerfile into blackwell-llm-docker
cp "$SCRIPT_DIR/Dockerfile" "$BLACKWELL_DIR/Dockerfile.vllm-b12x-cu132"

cd "$BLACKWELL_DIR"

echo "Building v18 image: $TAG"
echo "  Source: local-inference-lab/blackwell-llm-docker @ 7f3cbc6"
echo "  CUDA arch: sm_120 (forward-compat with SM121)"
echo "  MAX_JOBS: 12  VLLM_MAX_JOBS: 12"
echo "  Copy to workers: $([ "$SOLO" == false ] && echo yes || echo no)"
echo ""

BUILD_ARGS=(
    IMAGE="$TAG"
    BUILD_BASE_IMAGE=1
    PUSH_BASE_IMAGE=0
    MAX_JOBS=12
    VLLM_MAX_JOBS=12
    NVCC_THREADS=1
)

if [[ "$SOLO" == false ]]; then
    # Build + copy to workers
    export IMAGE="$TAG"
    export BUILD_BASE_IMAGE=1
    export PUSH_BASE_IMAGE=0
    export MAX_JOBS=12
    export VLLM_MAX_JOBS=12
    export NVCC_THREADS=1

    ./build-gilded-gnosis-v18-cu132.sh

    echo ""
    echo "=== Distributing to workers ==="
    echo "Save image: docker save $TAG | gzip > /tmp/v18.tar.gz"
    echo "Copy to each worker sequentially, removing old image first:"
    echo "  for host in 192.168.177.12 ... 192.168.177.18; do"
    echo "      ssh \"\$host\" 'docker rmi $TAG 2>/dev/null; docker image prune -f'"
    echo '      ssh "$host" "docker load" < /tmp/v18.tar.gz'
    echo "  done"
else
    # Build only
    ./build-gilded-gnosis-v18-cu132.sh
fi

echo ""
echo "Done. Image: $TAG:latest"
echo ""
echo "Post-build verification:"
echo '  docker run --rm $TAG bash -c '\''
    so=/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
    echo "sm_120a (must be 0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120a)"
    echo "sm_120 (should be >0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120)"
  '\'
