# GLM-5.2 on DGX Spark (8× GB10) — v18 Gilded Gnosis Build

Production vLLM image for 8× NVIDIA DGX Spark GB10 (SM121 / Grace-Blackwell SoC)
running GLM-5.2-Int4-Int8 with MTP k=4 speculative decoding.

## Results

### v18 vs v16 (MTP k=4, TP8/DCP1)

| Metric | v16 (fathomless-firmament) | v18 (gilded-gnosis) | Delta |
|---|---|---|---|
| Prefill | ~1200 t/s | ~1300 t/s | **+8%** |
| Decode | baseline | same (after warmup) | — |

### v18 Detailed Benchmarks (llama-benchy, coherent corpus, TP8/DCP1, MTP k=4)

| Test | t/s | Peak t/s | TTFR (ms) | Est PPT (ms) | E2E TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| pp2048 | 1329.72 | — | 1542.44 | 1540.17 | 1542.44 |
| tg1500 | 35.21 | 57.00 | — | — | — |
| pp2048 @ d16000 | 1319.37 | — | 13681.53 | 13679.26 | 13683.24 |
| tg1500 @ d16000 | 41.80 | 66.00 | — | — | — |
| pp2048 @ d100000 | 1202.08 | — | 84895.04 | 84892.77 | 84901.05 |
| tg1500 @ d100000 | 34.99 | 45.00 | — | — | — |

Context depth (d) = prompt tokens already in KV cache before the test.
Peak t/s = best single-step throughput. TTFR = time to first response token.

## ⚠ Warmup & Cache Requirements

**First boot:** CuTe DSL and Triton kernels JIT-compile for each unique batch
shape. Expect latency spikes (1-2s) during the first ~10-20 requests as new
shapes are encountered. After that, decode speed matches v16.

**Subsequent boots:** Kernels are loaded from persistent disk cache — no
warmup needed. The following cache directories must be persistent (mounted
from host) for this to work:

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

**If caches are NOT persistent** (e.g., inside ephemeral container filesystem),
every restart will re-JIT all kernels — the warmup period repeats on every boot.

The recipe sets `B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=0` to suppress
noisy disk-hit/miss logs during inference. Remove this env var if you need
to debug cache behavior.

## Overview

This repo adapts the [local-inference-lab](https://github.com/local-inference-lab)
Gilded Gnosis v18 image (originally built for RTX PRO 6000 / SM120 x86_64) for
the DGX Spark GB10 cluster (SM121 / aarch64).

The v18 source stack is used as-is — only the Dockerfile is modified for
aarch64/SM121, and runtime mods are applied for GB10-specific needs.

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

### What v19/v20 add (NOT included — not worth rebuilding for DCP1)

- **v19**: B12X PR #41 (deterministic CuTe cache keys) — **we cherry-pick this as
  a runtime mod** (`fix-b12x-cute-cache-key`). CUTLASS DSL 4.5.3 pin. DCP A2A
  buffer lifetime fix (DCP>1 only).
- **v20**: DCP head-major layout (DCP>1 only). TP6 workspace. SparkInfer rename.
  NVFP4 MLA KV outer-scale. Graph resource isolation.

For our setup (TP8/DCP1/MTP k=4/INT4-Int8), none of the v19/v20 DCP/TP6
improvements are relevant. The only useful v19 fix (CuTe cache keys) is
applied as a runtime mod.

## ⚠ Critical GB10/SM121 Architecture Lessons

These are the **non-obvious** issues that will cause build failures or runtime
crashes. Read carefully before modifying any arch settings.

### 1. sm_120 vs sm_120a (BUILD TIME)

| Target | Loads on SM121? | Supports NVFP4? |
|---|---|---|
| `sm_120` (no suffix) | ✅ Yes (forward-compatible) | ❌ No |
| `sm_120a` (with suffix) | ❌ No (`cudaErrorNoKernelImageForDevice`) | ✅ Yes |

**Rule:** Always build with `sm_120` (NOT `sm_120a`). The `a` suffix produces
arch-specific cubins that do NOT load on SM121 hardware despite being "close".

Dockerfile settings:
```
TORCH_CUDA_ARCH_LIST=12.0
CMAKE_CUDA_ARCHITECTURES=120
FLASHINFER_CUDA_ARCH_LIST=12.0f
CUDA_ARCH_FLAGS="-gencode arch=compute_120,code=sm_120"
```

Do NOT use `12.1`, `121`, or any `a`-suffixed variant in the Dockerfile.

### 2. CUTE_DSL_ARCH = sm_121a (RUNTIME, not build time)

CuTeDSL JIT-compiles kernels at runtime for the actual hardware. The Dockerfile
uses `sm_120a` in the serve script (for RTX PRO 6000), but GB10 requires:

```yaml
CUTE_DSL_ARCH: "sm_121a"   # In recipe env, NOT Dockerfile
```

Setting `CUTE_DSL_ARCH=sm_120a` at runtime causes `cudaErrorNoKernelImageForDevice`
because CuTeDSL produces sm_120a cubins that SM121 rejects.

### 3. NVFP4 SM120 Must Be Disabled (BUILD TIME)

The v18 vLLM CMakeLists.txt compiles NVFP4 kernels for SM120 when CUDA ≥13.0
and arch `12.0` is in the arch list. But `cvt.e2m1x2.f32` (NVFP4 PTX instruction)
requires `sm_120a` in ptxas — which we can't use (see #1 above).

The Dockerfile patches `CMakeLists.txt` to disable NVFP4 SM120:
```dockerfile
RUN sed -i 's/cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")/cuda_archs_loose_intersection(FP4_SM120_ARCHS "99.0f" "${CUDA_ARCHS}")/' /opt/vllm/CMakeLists.txt
```

This is safe — GLM-5.2 INT4-Int8 uses AWQ/marlin, not NVFP4.

Without this fix, the vLLM build fails with:
```
ptxas error: Feature 'cvt.e2m1x2.f32' not supported on .target 'sm_120'
```

### 4. MAX_JOBS=12 (not 64) to avoid earlyoom

GB10 has 121 GB RAM. The default `MAX_JOBS=64` causes OOM kills from earlyoom
during vLLM compilation. Use `MAX_JOBS=12` and `VLLM_MAX_JOBS=12`.

### 5. B12X serve script flags are for NVFP4, not INT4-Int8

The v18 image's `serve-glm52-v18.sh` sets dozens of B12X optimization env vars
(`VLLM_USE_B12X_WO_PROJECTION`, `B12X_W4A16_TC_DECODE`, `VLLM_NF3_GRID188_DECODE`,
etc.). These are designed for NVFP4 quantization. For INT4-Int8 (compressed-tensors),
the recipe uses `raw_entrypoint: true` and does NOT set these flags — matching
the v16 recipe that works without them.

Do NOT blindly copy serve script env vars into the recipe. Most are NVFP4-specific
and some (like `CUTE_DSL_ARCH=sm_120a`) will crash on GB10.

### 6. FlashInfer arch — do NOT override at runtime

The image pre-builds FlashInfer kernels for `sm_120` (forward-compatible with
SM121). Do NOT set `FLASHINFER_CUDA_ARCH_LIST=12.1f` at runtime — this forces
JIT recompilation and can produce suboptimal kernels. Just let it use the
pre-built sm_120 kernels.

### 7. InstantTensor does NOT work on GB10/NFS

`--load-format instanttensor` crashes with `RuntimeError: std::exception` in
`instanttensor._C.open()` on the GB10 aarch64 + NFS setup. Use default
`--load-format auto` (safetensors).

## Dockerfile Changes (`Dockerfile.vllm-b12x-cu132`)

1. **Architecture paths**: `x86_64-linux-gnu` → `$(dpkg-architecture -qDEB_HOST_MULTIARCH)`
   and `x86_64-linux` → `$(uname -m)-linux` for cuBLAS/CUDA lib paths.
2. **CUDA arch**: `TORCH_CUDA_ARCH_LIST=12.0a` → `12.0`,
   `CMAKE_CUDA_ARCHITECTURES=120a` → `120`,
   `FLASHINFER_CUDA_ARCH_LIST=12.0f` → `12.0f` (unchanged),
   `CUDA_ARCH_FLAGS="-gencode arch=compute_120,code=sm_120"` added.
3. **NVFP4 SM120 disabled**: sed patch to `CMakeLists.txt` (see #3 above).
4. **cmake cache clean**: `rm -rf /opt/vllm/build /opt/vllm/CMakeCache.txt /opt/vllm/CMakeFiles`
   before vLLM build to force reconfigure with correct arch.
5. **pip check non-fatal**: `nvidia-cusparselt-cu13` not supported on aarch64.
6. **NCCL path assertion non-fatal**: aarch64 loads backup copy differently.
7. **Base images**: `BUILD_BASE_IMAGE=1` to build from source on aarch64.

A pre-patched Dockerfile is included: `Dockerfile.vllm-b12x-cu132.aarch64-sm121`.

## Build Command

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

⚠ **Before loading a new image on workers, remove old images first.** Failing to
remove old images can cause layer conflicts, different image IDs, and bloated
disk usage.

## Runtime Mods (7 total — all auto-detect v18/v16 paths)

| # | Mod | Purpose | In v18 source? |
|---|---|---|---|
| 1 | `fix-v18-venv-paths` | Symlink `/opt/venv` → `/usr/local/lib/python3.12/dist-packages` for mod compat | N/A (path compat) |
| 2 | `fix-mtp-quant-packed-mapping` | `packed_modules_mapping` for `fused_qkv_a_proj` + `gate_up_proj` in MTP draft quant config | ❌ **Required!** |
| 3 | `fix-dsa-block-table-dim` | Off-by-one in `expanded_block_table_buffer` — crashes with concurrent MTP decode | ❌ **Required!** |
| 4 | `fix-fsm-toolcall-v18` | PR #44993: `should_advance` with `new_token_ids` — fixes FSM tool-call failures under MTP | ❌ NOT in v18 |
| 5 | `fix-b12x-cute-cache-key` | B12X PR #41: deterministic CuTe DSL cache keys — eliminates decode warmup JIT recompilation across restarts | ❌ NOT in v18 |
| 6 | `decode-aware-scheduler` | Custom decode-aware prefill scheduler (penguinchang) — prevents long prefills from starving decode | ❌ NOT in v18 |
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
Without this, `repr(OptLevel(2))` produces `<cutlass.base_dsl.compiler.OptLevel object at 0xe64d36ebf6e0>`
— the memory address differs per process, so cache keys never match across restarts.
The fix replaces `repr()` with deterministic structural serialization: `OptLevel(_value=2)`.

**Also requires** `B12X_CUTE_COMPILE_CACHE_DIR` to be set to a persistent
(host-mounted) path, otherwise the cache is lost on container restart.

#### fix-fsm-toolcall-v18 (REQUIRED for concurrent tool-call requests)
Without this, concurrent requests with tool calls crash with:
```
NameError: name 'new_token_ids' is not defined
```
The mod updates `should_advance` to accept optional `new_token_ids` (PR #44993).
Only call site 1 (`_update_request_with_output`) passes `new_token_ids`; call
sites 2 and 3 fall back to the original delta calculation.

### Mods NOT needed (already in v18 source)

| Mod | Reason |
|---|---|
| `embed-tokens-pp-fix` | `skip_substrs.append("embed_tokens")` built in |
| `fix-trim-after-load` | `gc.collect()` + `torch.accelerator.empty_cache()` built in |
| `fix-pcp-dead-import` | v18 doesn't have LiteTopK |
| `fix-v1-pp-drafter-guard` | `getattr(self, "drafter", None)` safe guards built in |
| `fix-glm5-deepgemm` | UMA handling in `mem_utils.py` built in |

## Recipe

See `recipes/glm52-int4int8-v18.yaml` for the production vLLM serve command.

Key parameters:
- TP=8, PP=1, DCP=1
- MTP k=4 with `B12X_MLA_SPARSE` attention backend
- `--kv-cache-dtype fp8_ds_mla`
- `--gpu-memory-utilization 0.726`
- `--enable-decode-aware-prefill` (custom mod)
- `--compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":16}'`
- `-cc.pass_config.fuse_allreduce_rms=True`
- `--enable-flashinfer-autotune`

### Recipe env: v16 vs v18 differences

The v18 recipe env matches v16 exactly, with v18-specific additions only:

| Env var | Value | Why |
|---|---|---|
| `LD_PRELOAD` | `/opt/libnccl-local-inference.so.2.30.4` | v18 has NCCL built in |
| `VLLM_DCP_QUERY_SPLIT` | `0` | DCP=1, no query split |
| `VLLM_B12X_MLA_CKV_GATHER` | `1` | Enable CKV gather for decode |
| `B12X_CUTE_COMPILE_CACHE_DIR` | `/root/.cache/huggingface/b12x/cute_compile` | Persistent CuTe cache |
| `B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START` | `0` | Suppress noisy cache logs |

### DSpark recipe (experimental)

`recipes/glm52-int4int8-dspark.yaml` uses the
[b1rd/GLM-5.2-speculator.dspark-quanttrio-int4-ft](https://huggingface.co/b1rd/GLM-5.2-speculator.dspark-quanttrio-int4-ft)
draft model, finetuned specifically for QuantTrio Int4-Int8Mix.

⚠ **v18 lacks `VLLM_DSPARK_DRAFT_RING`** — the speculator README says ring-buffer
windowed drafting is "required" to avoid acceptance depth-collapse. Without it,
acceptance will be lower (~1.4 vs ~2.1 tokens/step). Still worth testing.

## Cluster Topology

8× DGX Spark GB10 (SM121 / Grace-Blackwell):
- Head: `192.168.178.11` (also runs opencode control plane)
- Workers: `.178.12`–`.178.18`
- RoCE fabric: `192.168.177.0/24` (used for cluster management + model distribution)
- Model: NFS-mounted at `/home/ciprian/models/models14/GLM-5.2-Int4-Int8`

## Docker Tag Strategy

Only two tags maintained:
- `vllm-node-tf5-glm52-v16:latest` — fallback (v16 fathomless-firmament)
- `vllm-node-tf5-glm52-v18:latest` — current prod (v18 gilded-gnosis)

## Files in This Repo

| File | Description |
|---|---|
| `build-v18-gb10.sh` | One-command build script (clone + patch + build + verify) |
| `Dockerfile.vllm-b12x-cu132.aarch64-sm121` | Pre-patched Dockerfile (all GB10 fixes) |
| `recipes/glm52-int4int8-v18.yaml` | Production MTP recipe |
| `recipes/glm52-int4int8-dspark.yaml` | Experimental DSpark recipe |
| `mods/` | 7 runtime mods (applied inside container before `vllm serve`) |
