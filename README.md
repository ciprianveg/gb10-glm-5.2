# GLM-5.2 on DGX Spark (8× GB10) — v18 Gilded Gnosis

**1,329 t/s prefill · 66 t/s peak decode · +9.5% prefill over v16**

Serves [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) (in-checkpoint MTP, 256 experts) on an **8-node DGX Spark GB10 cluster** via TP8+PP1 with MTP k=4 speculative decoding.

| Version | Stack | Status |
|---------|-------|--------|
| **v18** | `gilded-gnosis-v18` | **Current production** 🚀 |
| **v18-vision** | v18-prod + vision + adaptive MTP 2/4 | [Vision + adaptive MTP](v18-vision/) |
| v16 | `fathomless-firmament-v16-unified` | [Legacy fallback](v16/) |

**Forum post:** [GLM-5.2 Int4-Int8 on 8× GB10 — 1,329 t/s prefill, 66 t/s peak decode](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831?u=ciprianveg)

---

## Quick Start (v18)

```bash
# 1. Clone dependencies
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git ../blackwell-llm-docker
cd ../blackwell-llm-docker && git checkout 7f3cbc6

# 2. Build v18 image
cd ../gb10-glm-5.2
./v18/build.sh                # builds + copy to all 7 workers (~30-40 min cached)

# 3. Deploy (from spark-vllm-docker)
cd ../spark-vllm-docker
./run-recipe.sh ../gb10-glm-5.2/v18/recipes/glm52-int4int8-v18.yaml --setup
```

> **First boot:** CuTe DSL + Triton kernels JIT-compile for each unique batch shape. Expect latency spikes (1-2 s) during the first ~10-20 requests. After warmup, decode speed matches and surpasses v16. For persistent caches, mount the directories listed in the [warmup section](#warmup--cache-requirements).

### Prebuilt Images (GHCR)

Skip the build entirely — pull a prebuilt image from GitHub Container Registry:

| Tag | Contents | Use when |
|-----|----------|----------|
| `ghcr.io/ciprianveg/gb10-glm-5.2:v18-base` | Compiled stack + source patches, **no runtime mods** | You want to select/tune mods yourself via `v18/mods/*/run.sh` |
| `ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod` | Base + all 7 production runtime mods baked in | Zero-config deploy matching the v18 recipes |

```bash
# Zero-config production image
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod

# Or the mod-free base (apply mods yourself at deploy)
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-base
```

The base image is the canonical artifact; `v18-prod` is a convenience layer with the [production mod set](v18/recipes/) pre-applied. Both are single-arch `linux/arm64` built for GB10 / sm_121 — they will not run on x86_64 or non-Blackwell GPUs. See [ATTRIBUTION.md](ATTRIBUTION.md) for upstream credits.

---

## v18-vision (vision + adaptive MTP 2/4)

A thin 9-file overlay on top of `v18-prod` that adds **GLM-5.2 vision** (MoonViT-3d + PatchMerger, weights from [baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)) and **adaptive MTP 2/4** (runtime speculative-depth ratcheting, from [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10)). Serves a composite checkpoint that symlinks the QuantTrio Int4-Int8 text weights and the baseten NVFP4 vision snapshot into one tree (~0.87 GiB of new vision weights; zero-copy).

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-vision
```

See [`v18-vision/README.md`](v18-vision/README.md) for the vision-tower download, composite-model assembly, build, and launch instructions. Recipe: [`v18-vision/recipes/glm52-int4int8-v18-vision.yaml`](v18-vision/recipes/glm52-int4int8-v18-vision.yaml).

---

## v18 Benchmarks (llama-benchy, coherent corpus, TP8/DCP1, MTP k=4)

> **Important:** For accurate results, first run a warmup pass (~20 requests at varying context depths) to trigger CuTe DSL + Triton kernel compilation. Without warmup, first-batch JIT overhead inflates TTFR and depresses decode t/s. With persistent cache mounts (see below), warmup is only needed once.

| Test | t/s | Peak t/s | TTFR (ms) | Est PPT (ms) | E2E TTFT (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 | 1,329.72 ± 0.00 | — | 1,542.44 ± 0.00 | 1,540.17 ± 0.00 | 1,542.44 ± 0.00 |
| tg1500 | 35.21 ± 0.00 | **66.00** | — | — | — |
| pp2048 @ d16000 | 1,319.37 ± 0.00 | — | 13,681.53 ± 0.00 | 13,679.26 ± 0.00 | 13,683.24 ± 0.00 |
| tg1500 @ d16000 | 41.80 ± 0.00 | 66.00 | — | — | — |
| pp2048 @ d100000 | 1,202.08 ± 0.00 | — | 84,895.04 ± 0.00 | 84,892.77 ± 0.00 | 84,901.05 ± 0.00 |
| tg1500 @ d100000 | 34.99 ± 0.00 | 45.00 | — | — | — |

- `d<N>` = prompt tokens already in KV cache before the test
- Peak t/s = best single-step throughput
- Prefill: **+9.5%** vs v16 (~1,215 → ~1,330)

### Warmup & Cache Requirements

The following cache directories must be **persistent** (mounted from host) to avoid re-JIT on every restart:

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

If caches are **not** persistent (e.g., ephemeral container filesystem), the warmup period repeats on every boot.

---

## What Changed: v16 → v18

v18 rebases the entire stack from `local-inference-lab`'s **Gilded Gnosis** v18 branch, bringing ~30 upstream PRs and component bumps.

### Upstream stack

| Component | v16 | v18 |
|-----------|-----|-----|
| vLLM | `fathomless-firmament-v16-unified` @ `5dffea8` | `gilded-gnosis-v18-final` @ `264bce1d` |
| B12X | `lukealonso/b12x` @ `97b3d64` | `voipmonitor/b12x` @ `bc85ef3` |
| FlashInfer | Prebuilt sm_121 wheels | `voipmonitor/flashinfer` @ `801d57a` |
| DeepGEMM | — | `a6b593d` |
| InstantTensor | — | `85e7c5f` |
| NCCL | `nvidia-nccl-cu12==2.30.4` | `local-inference-lab/nccl-canonical` 2.30.4 |
| PyTorch | 2.11.0 | 2.12.0 |
| CUDA | 13.2.0 | 13.2.1 |
| Build system | `eugr/spark-vllm-docker` | `local-inference-lab/blackwell-llm-docker` |

### New upstream PRs in v18

| PR / Feature | Description |
|---|---|
| **PR #109** | DSpark hardening |
| **PR #111** | TP8 full-CKV DCP prefill (DCP >1 only) |
| **PR #113** | NF3 Grid188 integration |
| **PR #115** | NVFP4 MLA KV cache support |
| **PR #116** | B12X scratch-format guard |
| **PR #117** | DCP A2A CUDA graph buffer lifetime |
| **PR #118** | MTP target-revision inheritance |
| **Upstream #47979** | SM120 PCIe serving stack |
| **B12X PR #41** | Deterministic CuTe cache keys (cherry-picked as mod) |
| **DeepGEMM** | SM120 GEMM kernels for MoE |
| **InstantTensor** | Fast model loader (not used on GB10/NFS — falls back to safetensors) |
| **Unified memory** | DGX Spark UMA handling in `mem_utils` |
| **Embed-tokens PP** | `embed_tokens` skip for PP built in |
| **GC after load** | `gc.collect()` + `empty_cache()` built in |

### What stayed the same

- Model weights: `QuantTrio/GLM-5.2-Int4-Int8Mix` (unchanged)
- Topology: TP8+PP1, DCP=1 across 8× GB10
- Attention backend: `B12X_MLA_SPARSE`
- MTP k=4 speculative decoding
- Decode-aware scheduler (penguinchang mod, now applies to both v16 and v18)

---

## v18 Runtime Mods

Seven runtime mods are applied by the v18 recipe at container startup. Each lives in [`v18/mods/`](v18/mods/) with its own README and apply script.

| # | Mod | Purpose | Required? |
|---|-----|---------|:--------:|
| 1 | `fix-v18-venv-paths` | Symlink `/opt/venv` → `/usr/local/lib/python3.12/dist-packages` | ✅ |
| 2 | `fix-mtp-quant-packed-mapping` | `packed_modules_mapping` for `fused_qkv_a_proj` + `gate_up_proj` in MTP draft quant config | ✅ |
| 3 | `fix-dsa-block-table-dim` | Off-by-one in `expanded_block_table_buffer` — concurrent MTP decode crash | ✅ |
| 4 | `fix-fsm-toolcall-v18` | PR #44993: `should_advance` with `new_token_ids` — FSM tool-call fix under MTP | ✅ |
| 5 | `fix-b12x-cute-cache-key` | Deterministic CuTe DSL cache keys (eliminates warmup JIT across restarts) | ✅ |
| 6 | `decode-aware-scheduler` | Custom decode-aware prefill scheduler (penguinchang) | optional |
| 7 | `fix-v16-b12x-stale-topk` | Stale `topk_indices_buffer` fix (may be in v18 source; included defensively) | ✅ |

---

## Directory Structure

```
├── v16/                       v16 build (fathomless-firmament, fallback)
│   ├── Dockerfile             reference build
│   ├── build.sh               build script
│   ├── patches/               v16-final patches (6 files)
│   ├── mods/                  v16 + shared mods
│   │   ├── decode-aware-scheduler/
│   │   ├── fix-fsm-toolcall/
│   │   └── fix-v16-b12x-stale-topk/
│   └── recipes/               v16 recipes
│       ├── glm52-int4int8-v16.yaml
│       └── glm52-int4int8-v16-pp2.yaml
├── v18/                       ← self-contained v18 build (current prod)
│   ├── Dockerfile             adapted aarch64/SM121 reference
│   ├── build.sh               v18 build script
│   ├── mods/                  all 7 v18 runtime mods
│   └── recipes/               v18 recipes (production + DSpark variants)
├── v18-vision/                v18-prod + vision + adaptive MTP 2/4 overlay
│   ├── Dockerfile             9-file overlay on v18-prod
│   ├── build.sh               overlay build script
│   ├── overlay/vllm/...       the 9 overlaid vLLM .py files
│   ├── scripts/               composite assembler + registry overlay
│   └── recipes/               v18-vision recipe
├── README.md
└── ATTRIBUTION.md
```

---

## Requirements

- 8× DGX Spark GB10 (SM121, aarch64)
- Node-to-node RoCE v2 (ConnectX-7, subnet 192.168.177.0/24)
- ~410 GB weights per node (NFS-mounted)
- `local-inference-lab/blackwell-llm-docker` for v18 build
- `eugr/spark-vllm-docker` for v18 deploy (recipe runner)

## License

Apache-2.0 (this repo). Serves MIT weights (GLM-5.2 by Z.ai → QuantTrio quants).

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits.
