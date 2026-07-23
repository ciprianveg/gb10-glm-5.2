# GLM-5.2 on DGX Spark (8× GB10)

GLM-5.2-Int4-Int8Mix serving on 8× NVIDIA DGX Spark GB10 (SM121 / Grace-Blackwell SoC) with MTP k=4 speculative decoding.

| Version | Stack | Status |
|---------|-------|--------|
| **v16** | `fathomless-firmament-v16-unified` | Production (fallback) |
| **v18** | `gilded-gnosis-v18` | **Current production** |

**Forum post:** [GLM-5.2 Int4-Int8 on 8× GB10 — 1,200 t/s prefill, 33-54 t/s avg decode](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831?u=ciprianveg)

---

## Quick Start

```bash
# 1. Clone dependencies
git clone https://github.com/eugr/spark-vllm-docker ../spark-vllm-docker
cd ../spark-vllm-docker && ./run-recipe.sh --discover  # generates .env with cluster IPs

# 2. Build image (v16)
cd ../gb10-glm-5.2
./build.sh                # builds + copies to all 7 workers (~13 min)
# ./build.sh --solo       # build only

# 3. Deploy & run (from spark-vllm-docker)
cd ../spark-vllm-docker
./run-recipe.sh ../gb10-glm-5.2/recipes/glm52-int4int8-v16.yaml --setup
```

---

# v16 — Original Production Build

## Overview

Serves the [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) (in-checkpoint MTP) on an **8-node GB10 cluster** via TP8 + PP1 with MTP k=4.

**Current production config:** TP8 + PP1 (1,211 t/s prefill, 35 t/s decode coherent corpus, 54 t/s game bench, 91/100 tool eval — relies on the `fix-fsm-toolcall` mod for stable MTP tool calling)
**Experimental:** TP4 + PP2 1800tps prefill but blocked on MTP acceptance ~8% vs expected ~85%

| Config | Prefill (t/s) | Decode (t/s) | MTP Acceptance |
|--------|--------------|--------------|----------------|
| **TP8+PP1 (prod)** | **~1,211** | **~35** | ~85% |
| TP4+PP2 (exp) | ~1,800 | ~12 | ~8% |

## Build Stack

Starting from CosmicRaisins' DCP1 solution (TP8+PP1, MTP k=4, B12X_MLA_SPARSE), this image upgrades to the codex/fathomless-firmament-v16-unified-20260712 unified branch and adds targeted patches + a runtime mod.

### Base versions

| Component | Version | Why |
|-----------|---------|-----|
| vLLM fork | `local-inference-lab/vllm` @ `5dffea8` (branch `codex/fathomless-firmament-v16-unified-20260712`) | DSpark support, SM120 PCIe serving, GLM-5.2 MTP kernels, MRv2 model runner, B12X MoE integration |
| b12x | `lukealonso/b12x` @ `97b3d64` (master) | W4A8 MoE, unified SM120 sparse MLA, PCIe DCP collectives, decode optimization (~28-49 t/s with MTP k=4 on old branch -> 33-55 t/s range) |
| CUDA | 13.2.0 | GB10 / sm_121 support |
| PyTorch | 2.11.0 | Pinned by v16 branch |
| FlashInfer | Prebuilt sm_121 wheels | Sparse MLA attention kernels |
| NCCL | 2.30.4 (custom aarch64) | 3-node mesh ring support |
| transformers | >=5.0 (`--tf5` build flag) | Required for GLM-5.2 model definitions |

### Patches (patches/v16-final/)

| Patch | Purpose | Production? |
|-------|---------|:-----------:|
| `01-pr72-1-draft-dcp-config-propagation.patch` | DCP config -> draft model (prevents MTP collapse under DCP>1). From CosmicRaisins' PR #72. | Yes |
| `03-draft-quant-packed-mapping.patch` | Quantized NextN draft token mapping (without this, quantized drafts silently build unquantized and MTP acceptance collapses). From CosmicRaisins. | Yes |
| `04-v16-essential.patch` | Three fixes: (1) DeepSeekMTP `SupportsPP` interface, (2) stale `topk_indices_buffer` in flashinfer_sm120 sparse MLA (from PR #46994), (3) MTP `embed_tokens` loading under PP. | Yes |
| `06-b12x-stale-topk-buffer.patch` | Same stale `topk_indices_buffer` fix applied to `b12x_mla_sparse.py` (Fix #4 from PR #46994). Without this, `_maybe_share_lm_head` replaces the indexer's buffer but the backend holds a stale reference -> garbage DSA attention and ~30% acceptance instead of ~85%. | Yes |
| `05-pp-mtp-broadcast-and-draft-relay.patch` | PR #46994 Fix #2 (broadcast padding to `max_sample_len`) + Fix #3 (draft token relay to non-last PP ranks). | PP2 only |
| `07-draft-pp-size-fix.patch` | Sets draft `pipeline_parallel_size=1` instead of copying target's. | PP2 only |

**Production image (TP8+PP1) uses patches 01, 03, 04, 06 only.**

### Runtime mods

**`fix-fsm-toolcall`** (PR #44993) -- Fixes `"Failed to advance FSM"` errors during
tool calling + MTP. The v16 fork already includes PR #44297 (`trim_reasoning_for_advance`)
and #46149 (`reasoning=reasoning_enabled` in structural tags), but `should_advance()`
still uses `num_computed_tokens - num_output_placeholders` to derive the delta window --
which breaks under MTP rejection (placeholder count stays >0, window starts past the
reasoning-end marker, grammar never enforced -> HTTP 500). This mod passes `new_token_ids`
directly to `should_advance()`, bypassing the broken placeholder math, and extends
same-step advance to all backend types.

**`decode-aware-scheduler`** (penguinchang, NVIDIA Developer Forums 2026-07-15) --
Prevents long-prefill requests from starving decode streams under concurrent load.
When decode is active, prefill is limited to a shared token budget (256 tokens/step
in production); when idle, prefill gets the full batched token budget. At most 1
long-prefill per step with round-robin rotation. Reduces decode stalls from
multi-second blocks to ~0.5s. See [mods/decode-aware-scheduler/README.md](mods/decode-aware-scheduler/README.md)
for tuning guide.

## v16 Recipe Files

- `recipes/glm52-int4int8-v16.yaml` -- **Production** (TP8+PP1, DCP=1, MTP k=4)
- `recipes/glm52-int4int8-v16-pp2.yaml` -- **Experimental** (TP4+PP2, MTP k=4, low mtp acceptance)

Both recipes reference the same image tag: `vllm-node-tf5-glm52-v16:latest`

## v16 Requirements

- 8x GB10 / DGX Spark (sm_121, aarch64)
- Node-to-node RoCE v2 (ConnectX-7, subnet 192.168.177.0/24)
- ~410 GB weights per node (or NFS-mounted)
- eugr/spark-vllm-docker for build + deploy

## v16 Performance (llama-benchy, coherent corpus, tg=1500)

| Depth | Prefill (t/s) | Avg decode (t/s) | Peak decode (t/s) | TTFR (ms) |
|-------|--------------:|-----------------:|------------------:|----------:|
| 0 | 1,211 +/- 0.9 | 34.9 +/- 2.8 | 53.5 +/- 3.5 | 1,693 |
| 4k | 1,117 +/- 100.7 | 38.3 +/- 0.5 | 58.0 +/- 0.0 | 5,461 |
| 16k | 1,215 +/- 23.8 | 37.7 +/- 0.0 | 58.0 +/- 0.0 | 14,867 |
| 32k | 1,176 +/- 4.7 | 33.3 +/- 2.7 | 54.5 +/- 2.5 | 28,963 |
| 100k | 1,128 +/- 0.9 | 34.8 +/- 3.8 | 51.5 +/- 1.5 | 90,448 |
| 200k | 1,019 +/- 0.0 | 37.8 +/- 0.0 | 50.0 +/- 0.0 | 198,327 |

**Game bench (Snake, 1500 tokens, temp=0, thinking=disabled):** 54.16 tok/s sustained

**Coding context** (vs coherent corpus above): avg gen 40-55 t/s single-stream, 60-70 t/s with 2 concurrent requests.

### Tool evaluation (tool-eval-bench v2.0.0)

| Metric | Score |
|--------|------:|
| **Overall quality** | 91 / 100 (Excellent) |
| Responsiveness | 43 / 100 (median turn: 3.6s) |
| Deployability | 77 / 100 (alpha=0.7) |
| Pass rate | 59 passed, 8 partial, 2 failed (126/138 pts) |
| Token efficiency | 0.6 pts/1K tokens (210K tokens total) |
| Weakest category | Toolset Scale (62%) |

---

# v18 -- Gilded Gnosis Build (Current Production)

## Overview

This repo adapts the [local-inference-lab](https://github.com/local-inference-lab)
Gilded Gnosis v18 image (originally built for RTX PRO 6000 / SM120 x86_64) for
the DGX Spark GB10 cluster (SM121 / aarch64).

The v18 source stack is used as-is -- only the Dockerfile is modified for
aarch64/SM121, and runtime mods are applied for GB10-specific needs.

## v18 Results

### v18 vs v16 (MTP k=4, TP8/DCP1)

| Metric | v16 (fathomless-firmament) | v18 (gilded-gnosis) | Delta |
|---|---|---|---|
| Prefill | ~1200 t/s | ~1300 t/s | **+8%** |
| Decode | baseline | same (after warmup) | -- |

### v18 Detailed Benchmarks (llama-benchy, coherent corpus, TP8/DCP1, MTP k=4)

| Test | t/s | Peak t/s | TTFR (ms) | Est PPT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| pp2048 | 1329.72 | -- | 1542.44 | 1540.17 | 1542.44 |
| tg1500 | 35.21 | 57.00 | -- | -- | -- |
| pp2048 @ d16000 | 1319.37 | -- | 13681.53 | 13679.26 | 13683.24 |
| tg1500 @ d16000 | 41.80 | 66.00 | -- | -- | -- |
| pp2048 @ d100000 | 1202.08 | -- | 84895.04 | 84892.77 | 84901.05 |
| tg1500 @ d100000 | 34.99 | 45.00 | -- | -- | -- |

Context depth (d) = prompt tokens already in KV cache before the test.
Peak t/s = best single-step throughput. TTFR = time to first response token.

## Warmup & Cache Requirements (v18)

**First boot:** CuTe DSL and Triton kernels JIT-compile for each unique batch
shape. Expect latency spikes (1-2s) during the first ~10-20 requests as new
shapes are encountered. After that, decode speed matches v16.

**Subsequent boots:** Kernels are loaded from persistent disk cache -- no
warmup needed. The following cache directories must be persistent (mounted
from host) for this to work:

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

**If caches are NOT persistent** (e.g., inside ephemeral container filesystem),
every restart will re-JIT all kernels -- the warmup period repeats on every boot.

## Source Stack (v18 Gilded Gnosis)

Based on [glm5.2_v18.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v18.md).

| Component | Ref / Commit |
|---|---|
| vLLM | [`local-inference-lab/vllm` branch `build/gilded-gnosis-v18-final-20260718`](https://github.com/local-inference-lab/vllm/tree/build/gilded-gnosis-v18-final-20260718) @ `264bce1d` |
| B12X | `voipmonitor/b12x` @ `bc85ef3` (branch `codex/nf3-grid188-decode-20260717`) |
| FlashInfer | `voipmonitor/flashinfer` @ `801d57a` |
| DeepGEMM | `a6b593d` |
| InstantTensor | `85e7c5f` |
| NCCL | `local-inference-lab/nccl-canonical` 2.30.4 |
| PyTorch | 2.12.0+cu132 |
| CUDA | 13.2.1 |
| Build repo | `local-inference-lab/blackwell-llm-docker` @ `7f3cbc6` |

### v18 PRs Included (from upstream)

| PR | Description |
|---|---|
| #109 | DSpark hardening |
| #111 | TP8 full-CKV DCP prefill (DCP>1 only) |
| #113 | NF3 Grid188 integration |
| #114 | Environment-only DS4 helper |
| #115 | NVFP4 MLA KV cache support |
| #116 | B12X scratch-format guard |
| #117 | DCP A2A CUDA graph buffer lifetime |
| #118 | MTP target-revision inheritance |
| upstream #47979 | SM120 PCIe serving stack |

### What v19/v20 add (NOT included -- not worth rebuilding for DCP1)

- **v19**: B12X PR #41 (deterministic CuTe cache keys) -- **we cherry-pick this as
  a runtime mod** (`fix-b12x-cute-cache-key`). CUTLASS DSL 4.5.3 pin. DCP A2A
  buffer lifetime fix (DCP>1 only).
- **v20**: DCP head-major layout (DCP>1 only). TP6 workspace. SparkInfer rename.
  NVFP4 MLA KV outer-scale. Graph resource isolation.

For our setup (TP8/DCP1/MTP k=4/INT4-Int8), none of the v19/v20 DCP/TP6
improvements are relevant. The only useful v19 fix (CuTe cache keys) is
applied as a runtime mod.

## Critical GB10/SM121 Architecture Lessons (v18)

These are the **non-obvious** issues that will cause build failures or runtime
crashes. Read carefully before modifying any arch settings.

### 1. sm_120 vs sm_120a (BUILD TIME)

| Target | Loads on SM121? | Supports NVFP4? |
|---|---|---|
| `sm_120` (no suffix) | Yes (forward-compatible) | No |
| `sm_120a` (with suffix) | No (`cudaErrorNoKernelImageForDevice`) | Yes |

**Rule:** Always build with `sm_120` (NOT `sm_120a`). The `a` suffix produces
arch-specific cubins that do NOT load on SM121 hardware despite being "close".

### 2. CUTE_DSL_ARCH = sm_121a (RUNTIME, not build time)

CuTeDSL JIT-compiles kernels at runtime for the actual hardware. The Dockerfile
uses `sm_120a` in the serve script (for RTX PRO 6000), but GB10 requires:

```yaml
CUTE_DSL_ARCH: "sm_121a"   # In recipe env, NOT Dockerfile
```

### 3. NVFP4 SM120 Must Be Disabled (BUILD TIME)

The v18 vLLM CMakeLists.txt compiles NVFP4 kernels for SM120 when CUDA >=13.0
and arch `12.0` is in the arch list. But `cvt.e2m1x2.f32` (NVFP4 PTX instruction)
requires `sm_120a` in ptxas -- which we can't use.

The Dockerfile patches `CMakeLists.txt` to disable NVFP4 SM120:
```dockerfile
RUN sed -i 's/cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")/cuda_archs_loose_intersection(FP4_SM120_ARCHS "99.0f" "${CUDA_ARCHS}")/' /opt/vllm/CMakeLists.txt
```

### 4. MAX_JOBS=12 (not 64) to avoid earlyoom

GB10 has 121 GB RAM. The default `MAX_JOBS=64` causes OOM kills from earlyoom
during vLLM compilation. Use `MAX_JOBS=12` and `VLLM_MAX_JOBS=12`.

### 5. B12X serve script flags are for NVFP4, not INT4-Int8

The v18 image's `serve-glm52-v18.sh` sets dozens of B12X optimization env vars.
These are designed for NVFP4 quantization. For INT4-Int8 (compressed-tensors),
the recipe uses `raw_entrypoint: true` and does NOT set these flags.

### 6. FlashInfer arch -- do NOT override at runtime

The image pre-builds FlashInfer kernels for `sm_120` (forward-compatible with
SM121). Do NOT set `FLASHINFER_CUDA_ARCH_LIST=12.1f` at runtime.

### 7. InstantTensor does NOT work on GB10/NFS

`--load-format instanttensor` crashes with `RuntimeError: std::exception` on
the GB10 aarch64 + NFS setup. Use default `--load-format auto` (safetensors).

## Dockerfile Changes (v18)

1. **Architecture paths**: `x86_64-linux-gnu` -> `$(dpkg-architecture -qDEB_HOST_MULTIARCH)`
2. **CUDA arch**: `TORCH_CUDA_ARCH_LIST=12.0a` -> `12.0`, `CMAKE_CUDA_ARCHITECTURES=120a` -> `120`
3. **NVFP4 SM120 disabled**: sed patch to `CMakeLists.txt`
4. **cmake cache clean**: `rm -rf /opt/vllm/build /opt/vllm/CMakeCache.txt /opt/vllm/CMakeFiles`
5. **pip check non-fatal**: `nvidia-cusparselt-cu13` not supported on aarch64
6. **NCCL path assertion non-fatal**: aarch64 loads backup copy differently
7. **Base images**: `BUILD_BASE_IMAGE=1` to build from source on aarch64

## Build Command (v18)

### Option A: Using the build script (recommended)

```bash
./build-v18-gb10.sh
```

### Option B: Manual build

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 7f3cbc6

# Copy pre-patched Dockerfile
cp /path/to/gb10-glm-5.2/Dockerfile.vllm-b12x-cu132.aarch64-sm121 Dockerfile.vllm-b12x-cu132

# Build
IMAGE=vllm-node-tf5-glm52-v18:latest \
BUILD_BASE_IMAGE=1 \
PUSH_BASE_IMAGE=0 \
MAX_JOBS=12 \
VLLM_MAX_JOBS=12 \
NVCC_THREADS=1 \
  ./build-gilded-gnosis-v18-cu132.sh
```

Build time on GB10: ~30-40 minutes (cached base stages) or ~80-90 min (full rebuild).

### Post-build verification

```bash
# Verify sm_120 cubins (NOT sm_120a)
docker run --rm vllm-node-tf5-glm52-v18:latest bash -c '
  so=/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
  echo "sm_120a (must be 0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120a)"
  echo "sm_120 (should be >0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120)"
'
```

### Distributing to worker nodes

```bash
# Save image
docker save vllm-node-tf5-glm52-v18:latest | gzip > /tmp/v18.tar.gz

# Copy to each worker SEQUENTIALLY
for host in 192.168.177.12 ... 192.168.177.18; do
    ssh "$host" "docker rmi vllm-node-tf5-glm52-v18:latest 2>/dev/null; docker image prune -f"
    ssh "$host" "docker load" < /tmp/v18.tar.gz
done
```

**Before loading a new image on workers, remove old images first.** Failing to
remove old images can cause layer conflicts, different image IDs, and bloated
disk usage.

## v18 Runtime Mods (7 total -- all auto-detect v18/v16 paths)

| # | Mod | Purpose | In v18 source? |
|---|---|---|---|
| 1 | `fix-v18-venv-paths` | Symlink `/opt/venv` -> `/usr/local/lib/python3.12/dist-packages` for mod compat | N/A (path compat) |
| 2 | `fix-mtp-quant-packed-mapping` | `packed_modules_mapping` for `fused_qkv_a_proj` + `gate_up_proj` in MTP draft quant config | **Required!** |
| 3 | `fix-dsa-block-table-dim` | Off-by-one in `expanded_block_table_buffer` -- crashes with concurrent MTP decode | **Required!** |
| 4 | `fix-fsm-toolcall-v18` | PR #44993: `should_advance` with `new_token_ids` -- fixes FSM tool-call failures under MTP | NOT in v18 |
| 5 | `fix-b12x-cute-cache-key` | B12X PR #41: deterministic CuTe DSL cache keys -- eliminates decode warmup JIT recompilation across restarts | NOT in v18 |
| 6 | `decode-aware-scheduler` | Custom decode-aware prefill scheduler (penguinchang) -- prevents long prefills from starving decode | NOT in v18 |
| 7 | `fix-v16-b12x-stale-topk` | Stale `topk_indices_buffer` fix for B12X_MLA_SPARSE under spec decode | Possibly in v18 (defensive) |

### Mod Details

#### fix-mtp-quant-packed-mapping (REQUIRED)
Without this, loading fails with:
```
KeyError: 'model.layers.78.mtp_block.self_attn.kv_a_proj_with_mqa.weight_packed'
```
v18 has `stacked_params_mapping` for `fused_qkv_a_proj` in `load_weights` but
is missing `packed_modules_mapping` in the draft quant config.

#### fix-dsa-block-table-dim (REQUIRED)
Without this, concurrent MTP requests crash with:
```
RuntimeError: The expanded size of the tensor (7969) must match the existing size (7970)
```
The scheduler's `block_table_tensor` has `max_num_blocks_per_req+1` columns but
the indexer's buffer only has `max_num_blocks_per_req`.

#### fix-b12x-cute-cache-key (REQUIRED for persistent cache)
Without this, `repr(OptLevel(2))` produces a memory-address-dependent string
-- cache keys never match across restarts.
The fix replaces `repr()` with deterministic structural serialization: `OptLevel(_value=2)`.

#### fix-fsm-toolcall-v18 (REQUIRED for concurrent tool-call requests)
Without this, concurrent requests with tool calls crash with:
```
NameError: name 'new_token_ids' is not defined
```
The mod updates `should_advance` to accept optional `new_token_ids` (PR #44993).

### Mods NOT needed (already in v18 source)

| Mod | Reason |
|---|---|
| `embed-tokens-pp-fix` | `skip_substrs.append("embed_tokens")` built in |
| `fix-trim-after-load` | `gc.collect()` + `torch.accelerator.empty_cache()` built in |
| `fix-pcp-dead-import` | v18 doesn't have LiteTopK |
| `fix-v1-pp-drafter-guard` | `getattr(self, "drafter", None)` safe guards built in |
| `fix-glm5-deepgemm` | UMA handling in `mem_utils.py` built in |

## v18 Recipe

See `recipes/glm52-int4int8-v18.yaml` for the production vLLM serve command.

Key parameters:
- TP=8, PP=1, DCP=1
- MTP k=4 with `B12X_MLA_SPARSE` attention backend
- `--kv-cache-dtype fp8_ds_mla`
- `--gpu-memory-utilization 0.726`
- `--enable-decode-aware-prefill` (custom mod)
- `--compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":16}'`

### Recipe env: v16 vs v18 differences

| Env var | Value | Why |
|---|---|---|
| `LD_PRELOAD` | `/opt/libnccl-local-inference.so.2.30.4` | v18 has NCCL built in |
| `VLLM_DCP_QUERY_SPLIT` | `0` | DCP=1, no query split |
| `VLLM_B12X_MLA_CKV_GATHER` | `1` | Enable CKV gather for decode |
| `B12X_CUTE_COMPILE_CACHE_DIR` | `/root/.cache/huggingface/b12x/cute_compile` | Persistent CuTe cache |
| `B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START` | `0` | Suppress noisy cache logs |

---

# Shared

## Cluster Topology

8x DGX Spark GB10 (SM121 / Grace-Blackwell):
- Head: `192.168.178.11` (also runs opencode control plane)
- Workers: `.178.12`-`.178.18`
- RoCE fabric: `192.168.177.0/24` (used for cluster management + model distribution)
- Model: NFS-mounted at `/home/ciprian/models/models14/GLM-5.2-Int4-Int8`

## Docker Tag Strategy

Only two tags maintained:
- `vllm-node-tf5-glm52-v16:latest` -- fallback (v16 fathomless-firmament)
- `vllm-node-tf5-glm52-v18:latest` -- current prod (v18 gilded-gnosis)

## Files in This Repo

| File | Description |
|---|---|
| `Dockerfile` | v16 multi-stage build (eugr/spark-vllm-docker based) |
| `build.sh` | v16 build + deploy script |
| `patches/v16-final/` | 6 production patches + 2 PP2-only patches |
| `recipes/glm52-int4int8-v16.yaml` | v16 production recipe |
| `recipes/glm52-int4int8-v16-pp2.yaml` | v16 experimental PP2 recipe |
| `recipes/glm52-int4int8-v18.yaml` | v18 production recipe |
| `mods/decode-aware-scheduler/` | Decode-aware prefill scheduler (v16+v18) |
| `mods/fix-fsm-toolcall/` | PR #44993 FSM fix for v16 (diff-based) |
| `mods/fix-fsm-toolcall-v18/` | PR #44993 FSM fix for v18 (inline Python) |
| `mods/fix-mtp-quant-packed-mapping/` | MTP draft quant packed mapping (v18) |
| `mods/fix-dsa-block-table-dim/` | DSA indexer buffer off-by-one (v18) |
| `mods/fix-b12x-cute-cache-key/` | Deterministic CuTe cache keys (v18) |
| `mods/fix-v16-b12x-stale-topk/` | Stale topk buffer fix (v16+v18) |
| `mods/fix-v18-venv-paths/` | Venv symlink for mod compat (v18) |

## Attribution & Credits

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits.

This work stands on the shoulders of:

| Contribution | Source |
|--------------|--------|
| **Foundational GLM-5.2-on-GB10 stack** -- DCP patches (PR #72), `index_topk_pattern` override, B12X config, `draft-quant-packed-mapping` fix, eugr/spark-vllm-docker build system | [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) |
| **vLLM v16 branch** (DSpark, SM120 PCIe serving, GLM MTP fixes, B12X MoE kernels, MRv2 default) | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) @ `codex/fathomless-firmament-v16-unified-20260712` |
| **vLLM v18 branch** (Gilded Gnosis, DCP fast-path, NVFP4 MLA, MTP target-revision) | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) @ `build/gilded-gnosis-v18-final-20260718` |
| **b12x** (W4A8 MoE, unified SM120 sparse MLA, PCIe DCP collectives) | [lukealonso/b12x](https://github.com/lukealonso/b12x) |
| **PR #72** (DCP draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting`) | m9e / voipmonitor |
| **PR #46994** (V2+MTP+PP: SupportsPP, broadcast padding, draft relay, embed_tokens, stale topk fix) | eastwood-c / vllm-project |
| **PR #44993** (FSM `should_advance` new_token_ids fix) | vllm-project |
| **Decode-Aware Custom Scheduler** (dynamic prefill budgets, round-robin long-prefill selection) | [penguinchang](https://forums.developer.nvidia.com/u/penguinchang) / [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/376831) |
| **FlashInfer SM120 kernels** | FlashInfer team |
| **DeepGEMM SM120 support** | DeepSeek AI |
| **QuantTrio GLM-5.2-Int4-Int8Mix** (256-expert, in-checkpoint MTP) | QuantTrio / cyankiwi |
| **NCCL 2.30.4 aarch64 wheel** | NVIDIA |
| **eugr/spark-vllm-docker** build system (multi-stage Docker, wheel caching, SCP deploy) | ciprian / eugr |
| **B12X PR #41** (deterministic CuTe cache keys) | voipmonitor/b12x |

## License

Apache-2.0 (this repo). Serves MIT weights (GLM-5.2 by Z.ai -> QuantTrio quants).
