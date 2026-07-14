# gb10-glm-5.2 — GLM-5.2-Int4-Int8Mix on 8× DGX Spark (GB10, sm_121)

## Overview

Serves the [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) (in-checkpoint MTP) on an **8-node GB10 cluster** via TP8 + PP1 with MTP k=4.

**Current production config:** TP8 + PP1 (1,211 t/s prefill, 35 t/s decode coherent corpus, 54 t/s game bench, 91/100 tool eval — relies on the `fix-fsm-toolcall` mod for stable MTP tool calling)
**Experimental:** TP4 + PP2 1800tps prefill but blocked on MTP acceptance ~8% vs expected ~85%

| Config | Prefill (t/s) | Decode (t/s) | MTP Acceptance |
|--------|--------------|--------------|----------------|
| **TP8+PP1 (prod)** | **~1,211** | **~35** | ~85% |
| TP4+PP2 (exp) | ~1,800 | ~12 | ~8% |

## Quick Start

```bash
# 1. Clone dependencies
git clone https://github.com/eugr/spark-vllm-docker ../spark-vllm-docker
cd ../spark-vllm-docker && ./run-recipe.sh --discover  # generates .env with cluster IPs

# 2. Build image
cd ../gb10-glm-5.2
./build.sh                # builds + copies to all 7 workers (~13 min)
# ./build.sh --solo       # build only

# 3. Deploy & run (from spark-vllm-docker)
cd ../spark-vllm-docker
./run-recipe.sh ../gb10-glm-5.2/recipes/glm52-int4int8-v16.yaml --setup
```

## Build Stack

Starting from CosmicRaisins' DCP1 solution (TP8+PP1, MTP k=4, B12X_MLA_SPARSE), this image upgrades to the codex/fathomless-firmament-v16-unified-20260712 unified branch and adds targeted patches + a runtime mod.

### Base versions

| Component | Version | Why |
|-----------|---------|-----|
| vLLM fork | `local-inference-lab/vllm` @ `5dffea8` (branch `codex/fathomless-firmament-v16-unified-20260712`) | DSpark support, SM120 PCIe serving, GLM-5.2 MTP kernels, MRv2 model runner, B12X MoE integration |
| b12x | `lukealonso/b12x` @ `97b3d64` (master) | W4A8 MoE, unified SM120 sparse MLA, PCIe DCP collectives, decode optimization (~28–49 t/s with MTP k=4 on old branch → 33–55 t/s range) |
| CUDA | 13.2.0 | GB10 / sm_121 support |
| PyTorch | 2.11.0 | Pinned by v16 branch |
| FlashInfer | Prebuilt sm_121 wheels | Sparse MLA attention kernels |
| NCCL | 2.30.4 (custom aarch64) | 3-node mesh ring support |
| transformers | ≥5.0 (`--tf5` build flag) | Required for GLM-5.2 model definitions |

### Patches (patches/v16-final/)

| Patch | Purpose | Production? |
|-------|---------|:-----------:|
| `01-pr72-1-draft-dcp-config-propagation.patch` | DCP config → draft model (prevents MTP collapse under DCP>1). From CosmicRaisins' PR #72. | ✅ |
| `03-draft-quant-packed-mapping.patch` | Quantized NextN draft token mapping (without this, quantized drafts silently build unquantized and MTP acceptance collapses). From CosmicRaisins. | ✅ |
| `04-v16-essential.patch` | Three fixes: (1) DeepSeekMTP `SupportsPP` interface, (2) stale `topk_indices_buffer` in flashinfer_sm120 sparse MLA (from PR #46994), (3) MTP `embed_tokens` loading under PP. | ✅ |
| `06-b12x-stale-topk-buffer.patch` | Same stale `topk_indices_buffer` fix applied to `b12x_mla_sparse.py` (Fix #4 from PR #46994). Without this, `_maybe_share_lm_head` replaces the indexer's buffer but the backend holds a stale reference → garbage DSA attention and ~30% acceptance instead of ~85%. | ✅ |
| `05-pp-mtp-broadcast-and-draft-relay.patch` | PR #46994 Fix #2 (broadcast padding to `max_sample_len`) + Fix #3 (draft token relay to non-last PP ranks). | PP2 only |
| `07-draft-pp-size-fix.patch` | Sets draft `pipeline_parallel_size=1` instead of copying target's. | PP2 only |

**Production image (TP8+PP1) uses patches 01, 03, 04, 06 only.**

### Runtime mod (mods/fix-fsm-toolcall/)

**`fix-fsm-toolcall`** (PR #44993) — Fixes `"Failed to advance FSM"` errors during
tool calling + MTP. The v16 fork already includes PR #44297 (`trim_reasoning_for_advance`)
and #46149 (`reasoning=reasoning_enabled` in structural tags), but `should_advance()`
still uses `num_computed_tokens - num_output_placeholders` to derive the delta window —
which breaks under MTP rejection (placeholder count stays >0, window starts past the
reasoning-end marker, grammar never enforced → HTTP 500). This mod passes `new_token_ids`
directly to `should_advance()`, bypassing the broken placeholder math, and extends
same-step advance to all backend types.

## Recipe Files

- `recipes/glm52-int4int8-v16.yaml` — **Production** (TP8+PP1, DCP=1, MTP k=4)
- `recipes/glm52-int4int8-v16-pp2.yaml` — **Experimental** (TP4+PP2, MTP k=4, low mtp acceptance)

Both recipes reference the same image tag: `vllm-node-tf5-glm52-v16:latest`

## Requirements

- 8× GB10 / DGX Spark (sm_121, aarch64)
- Node-to-node RoCE v2 (ConnectX-7, subnet 192.168.177.0/24)
- ~410 GB weights per node (or NFS-mounted)
- eugr/spark-vllm-docker for build + deploy

## Performance (llama-benchy, coherent corpus, tg=1500)

| Depth | Prefill (t/s) | Avg decode (t/s) | Peak decode (t/s) | TTFR (ms) |
|-------|--------------:|-----------------:|------------------:|----------:|
| 0 | 1,211 ± 0.9 | 34.9 ± 2.8 | 53.5 ± 3.5 | 1,693 |
| 4k | 1,117 ± 100.7 | 38.3 ± 0.5 | 58.0 ± 0.0 | 5,461 |
| 16k | 1,215 ± 23.8 | 37.7 ± 0.0 | 58.0 ± 0.0 | 14,867 |
| 32k | 1,176 ± 4.7 | 33.3 ± 2.7 | 54.5 ± 2.5 | 28,963 |
| 100k | 1,128 ± 0.9 | 34.8 ± 3.8 | 51.5 ± 1.5 | 90,448 |
| 200k | 1,019 ± 0.0 | 37.8 ± 0.0 | 50.0 ± 0.0 | 198,327 |

**Game bench (Snake, 1500 tokens, temp=0, thinking=disabled):** 54.16 tok/s sustained

```
=== Game Benchmark (Single-Stream, temp=0, thinking=disabled) ===
Waiting for server to be ready...
Server ready after 1s
Running game benchmark (Snake game generation)...
Completion tokens: 1500
Prompt tokens: 43
Total tokens: 1543
Wall time: 27.69s
Average tok/s: 54.16
```

**Coding context** (vs coherent corpus above): avg gen 40–55 t/s single-stream, 60–70 t/s with 2 concurrent requests.

**Long-context coding benchmark** (4000-token input, 1500-token output, single-stream):

```
=== Long-Context Benchmark ===
Type:           coding
Target input:   4000 tokens
Output:         1500 tokens

Actual tokens:  3999 tokens (confirmed by server)

Sending request (streaming via httpx)...

============ Result ============
Input tokens:      4012
Output tokens:     1500
Wall time:         34.33s

TTFT:              2996.0 ms
Prefill tok/s:     1339.1
Gen tok/s:         47.8
Mean ITL:          20.9 ms
```

### Tool evaluation (tool-eval-bench v2.0.0)

| Metric | Score |
|--------|------:|
| **Overall quality** | 91 / 100 (★★★★★ Excellent) |
| Responsiveness | 43 / 100 (median turn: 3.6s) |
| Deployability | 77 / 100 (α=0.7) |
| Pass rate | 59 passed, 8 partial, 2 failed (126/138 pts) |
| Token efficiency | 0.6 pts/1K tokens (210K tokens total) |
| Weakest category | Toolset Scale (62%) |

## Attribution & Credits

This work stands on the shoulders of:

| Contribution | Source |
|--------------|--------|
| **Foundational GLM-5.2-on-GB10 stack** — DCP patches (PR #72), `index_topk_pattern` override, B12X config, `draft-quant-packed-mapping` fix, eugr/spark-vllm-docker build system | [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) |
| **vLLM v16 branch** (DSpark, SM120 PCIe serving, GLM MTP fixes, B12X MoE kernels, MRv2 default) | `local-inference-lab/vllm` @ `codex/fathomless-firmament-v16-unified-20260712` |
| **b12x** (W4A8 MoE, unified SM120 sparse MLA, PCIe DCP collectives, 80eb49b decode optimization) | `lukealonso/b12x` @ `97b3d64` |
| **PR #72** (DCP draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting`) | m9e / voipmonitor |
| **PR #46994** (V2+MTP+PP: SupportsPP, broadcast padding, draft relay, embed_tokens, stale topk fix) | eastwood-c / vllm-project |
| **FlashInfer SM120 kernels** | FlashInfer team |
| **DeepGEMM SM120 support** | DeepSeek AI |
| **QuantTrio GLM-5.2-Int4-Int8Mix** (256-expert, in-checkpoint MTP) | QuantTrio / cyankiwi |
| **NCCL 2.30.4 aarch64 wheel** | NVIDIA |
| **eugr/spark-vllm-docker** build system (multi-stage Docker, wheel caching, SCP deploy) | ciprian / eugr |

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits.

## License

Apache-2.0 (this repo). Serves MIT weights (GLM-5.2 by Z.ai → QuantTrio quants).
