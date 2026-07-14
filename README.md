# gb10-glm-5.2 — GLM-5.2-Int4-Int8Mix on 8× DGX Spark (GB10, sm_121)

## Overview

Serves the [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) (256 experts, 378 GB, in-checkpoint MTP) on an **8-node GB10 cluster** via TP8 + PP1 with MTP k=4.

**Current production config:** TP8 + PP1 (8% faster decode, 15% faster prefill )  
**Experimental:** TP4 + PP2 (blocked on MTP acceptance ~8% vs expected ~85%)

| Config | Prefill (t/s) | Decode (t/s) | MTP Acceptance |
|--------|--------------|--------------|----------------|
| **TP8+PP1 (prod)** | **~1,200** | **~35** | ~85% |
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
# or use the manage script:
# ../gb10-glm-5.2/manage-glm52-int4int8.sh start
```

## What's New vs Old Image

| Component | Old (`vllm-node-tf5-glm52-dcp`) | New (`vllm-node-tf5-glm52-v16`) |
|-----------|--------------------------------|--------------------------------|
| vLLM fork | `local-inference-lab/vllm` @ `e232d26` (branch `codex/dcp-globaltopk...`) | `local-inference-lab/vllm` @ `5dffea8` (branch `codex/fathomless-firmament-v16-unified`) |
| b12x | `9cd63a7` | `master` @ `97b3d64` |
| DSpark | No | Yes |
| SM120 PCIe | No | Yes |
| GLM-5.2 MTP kernels | Partial | Full (fused_indexer_q_rope_quant 1.9-3.3%, reduce-scatter MoE 3.1-3.2%) |
| B12X MoE | No | Yes (W4A8, unified SM120 sparse MLA) |
| PCIe DCP collectives | No | Yes |
| 80eb49b decode opt | No | Yes (24 t/s vs 9 t/s) |
| MRv2 default | No | Yes |

## Patches (patches/v16-final/)

| Patch | Purpose |
|-------|---------|
| `01-pr72-1-draft-dcp-config-propagation.patch` | DCP config → draft (prevents MTP collapse under DCP>1) |
| `03-draft-quant-packed-mapping.patch` | Quantized NextN drafts (prevents silent unquantized build) |
| `04-v16-essential.patch` | SupportsPP + stale topk (flashinfer_sm120) + embed_tokens loading |
| `06-b12x-stale-topk-buffer.patch` | **Critical:** B12X_MLA_SPARSE stale `topk_indices_buffer` fix (Fix #4 from PR #46994 applied to b12x_mla_sparse.py) |
| `05-pp-mtp-broadcast-and-draft-relay.patch` | PR #46994 Fix #2 (broadcast padding) + Fix #3 (draft relay) — **PP2 only** |
| `07-draft-pp-size-fix.patch` | Draft `pipeline_parallel_size=1` — **PP2 only** |

**Production image (TP8+PP1) uses patches 01, 03, 04, 06 only.**  
PP2 patches (05, 07) are included in the repo for experimentation but not needed for TP8+PP1.

## Recipe Files

- `recipes/glm52-int4int8-v16.yaml` — **Production** (TP8+PP1, DCP=1, MTP k=4)
- `recipes/glm52-int4int8-v16-pp2.yaml` — **Experimental** (TP4+PP2, MTP k=4, low mtp acceptance)

Both recipes reference the same image tag: `vllm-node-tf5-glm52-v16:latest`

## Requirements

- 8× GB10 / DGX Spark (sm_121, aarch64)
- Node-to-node RoCE v2 (ConnectX-7, subnet 192.168.177.0/24)
- ~410 GB weights per node (or NFS-mounted at `/home/ciprian/models/models14`)
- eugr/spark-vllm-docker for build + deploy

## Performance (llama-benchy, coherent corpus, tg=1500)

| Depth | Prefill (t/s) | Decode (t/s) |
|-------|--------------|--------------|
| 0 | 1,211 | 35 |
| 4k | 1,118 | 33 |
| 16k | 1,210 | 42 |
| 100k | 1,123 | 29 |
| 200k | 1,019 | 38 |

**Game bench (Snake, 1500 tokens):** 54 tok/s sustained

## License

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
| **QuantTrio GLM-5.2-Int4-Int8Mix** (unpruned 256-expert, in-checkpoint MTP) | QuantTrio / cyankiwi |
| **NCCL 2.30.4 aarch64 wheel** | NVIDIA |
| **eugr/spark-vllm-docker** build system (multi-stage Docker, wheel caching, SCP deploy) | ciprian / eugr |

See [ATTRIBUTION.md](ATTRIBUTION.md) for full credits.

## License

Apache-2.0 (this repo). Serves MIT weights (GLM-5.2 by Z.ai → QuantTrio quants).
