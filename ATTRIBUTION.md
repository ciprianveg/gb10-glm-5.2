# Attribution & Credits

This project builds on the following open-source works. Full credit to their authors.

## Foundational Work

**CosmicRaisins/glm-5.2-gb10** pioneered GLM-5.2 serving on DGX Spark (GB10, sm_121). This project builds directly on their work:

- Identified the `--hf-overrides '{"index_topk_pattern":"FFFSSS..."}'` requirement (78-char pattern derived from `indexer_types`; without it, 56/78 layers top-k through uninitialized weights — coherent under ~2k tokens, garbage beyond)
- Created the DCP stack patches (PR #72: draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting`)
- Established the `VLLM_USE_V2_MODEL_RUNNER=1` + `VLLM_USE_B12X_SPARSE_INDEXER=1` + `--attention-backend B12X_MLA_SPARSE` serving configuration
- Identified the `draft-quant-packed-mapping` fix (without it, quantized NextN drafts silently build unquantized and MTP acceptance collapses)
- Created the `eugr/spark-vllm-docker` fork (multi-stage Docker build, wheel caching, SCP deploy, recipe runner, autodiscovery)

Full credit to **CosmicRaisins** for the foundational GLM-5.2-on-GB10 serving stack. See [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) and their [ATTRIBUTION.md](https://github.com/CosmicRaisins/glm-5.2-gb10/blob/master/ATTRIBUTION.md) for the complete lineage.

## Upstream Sources

| Project | Repo | Commit/Branch | License | Used For |
|---------|------|---------------|---------|----------|
| **vLLM** | `local-inference-lab/vllm` | `codex/fathomless-firmament-v16-unified-20260712` @ `5dffea8` | Apache-2.0 | Core inference engine, V2 runner, MTP, DCP, B12X integration |
| **b12x** | `lukealonso/b12x` | `master` @ `97b3d64` | Apache-2.0 | MoE kernels, W4A8 quantization, unified SM120 sparse MLA, PCIe DCP collectives |
| **FlashInfer** | `flashinfer-ai/flashinfer` | Prebuilt wheels (sm_121) | Apache-2.0 | Sparse MLA attention kernels, page attention |
| **DeepGEMM** | `deepseek-ai/DeepGEMM` | `nv_dev` branch | Apache-2.0 | SM120 GEMM kernels for MoE |
| **NCCL** | `zyang-dev/nccl` | `dgxspark-3node-ring` | BSD-3-Clause | 3-node ring collectives over RoCE |
| **QuantTrio GLM-5.2-Int4-Int8Mix** | `QuantTrio/GLM-5.2-Int4-Int8Mix` | Latest | MIT | Model weights (256-expert, in-checkpoint MTP) |
| **eugr/spark-vllm-docker** | `eugr/spark-vllm-docker` | Latest | Apache-2.0 | Multi-stage Docker build, wheel caching, SCP deploy, recipe runner |
| **CosmicRaisins/glm-5.2-gb10** | `CosmicRaisins/glm-5.2-gb10` | Latest | Apache-2.0 | Foundational GLM-5.2-on-GB10 stack: DCP patches, `index_topk_pattern` override, B12X config, `draft-quant-packed-mapping` fix |

## Key Upstream PRs Incorporated

| PR | Author | Repo | Status | What It Fixes |
|----|--------|------|--------|---------------|
| **#72** | m9e | `local-inference-lab/vllm` | Merged (part) | DCP draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting` |
| **#46994** | eastwood-c | `vllm-project/vllm` | Open | V2+MTP+PP: SupportsPP, broadcast padding (Fix #2), draft relay (Fix #3), embed_tokens loading (6f54d3c), stale topk_indices_buffer (Fix #4) |

Our patches `01`, `03`, `04`, `06` backport the above fixes to the v16 branch.

## Patches in This Repo (patches/v16-final/)

| File | Origin | Description |
|------|--------|-------------|
| `01-pr72-1-draft-dcp-config-propagation.patch` | PR #72 part 1 | Propagates `decode_context_parallel_size` to draft config |
| `03-draft-quant-packed-mapping.patch` | PR #72 related | Maps quantized NextN draft tokens correctly |
| `04-v16-essential.patch` | PR #46994 Fix #1, #4 (flashinfer), 6f54d3c | DeepSeekMTP `SupportsPP` + `make_empty_intermediate_tensors`; stale `topk_indices_buffer` in `flashinfer_mla_sparse_sm120`; MTP `embed_tokens` loading under PP |
| `06-b12x-stale-topk-buffer.patch` | PR #46994 Fix #4 adapted | **Critical:** B12X_MLA_SPARSE stale `topk_indices_buffer` — store `self._indexer = indexer`, read dynamically in `forward_mqa` |
| `05-pp-mtp-broadcast-and-draft-relay.patch` | PR #46994 Fix #2 + #3 | PPHandler broadcast padding + draft token relay (PP2 only) |
| `07-draft-pp-size-fix.patch` | New (same class as PR #72) | `create_draft_parallel_config()` sets `pipeline_parallel_size=1` for draft (PP2 only) |

## Community Contributions

| Contribution | Author | Source | What It Does |
|-------------|--------|--------|--------------|
| **Decode-Aware Custom Scheduler** | [penguinchang](https://forums.developer.nvidia.com/u/penguinchang) | [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831) (2026-07-15) | Scheduler patch that prevents long-prefill requests from starving decode streams. Adds dynamic prefill budgets (`--decode-prefill-token-budget`, `--idle-prefill-token-budget`), round-robin long-prefill selection, and runtime enable/disable. See `mods/decode-aware-scheduler/README.md`. |

## Model Provenance

```
GLM-5.2 (744B/40B MoE, GlmMoeDsa)
  └─ Z.ai (original)
      └─ QuantTrio/GLM-5.2-Int4-Int8Mix (w4a16/w8a16, 256 experts, in-checkpoint MTP layer 78)
          └─ cyankiwi (quantization)
```

## Build System

The multi-stage Dockerfile, wheel caching, SCP parallel deploy, and recipe runner are from **eugr/spark-vllm-docker** (Apache-2.0). See that repo for build infrastructure credits.

## License

This repository: Apache-2.0  
Served weights: MIT (GLM-5.2 by Z.ai → QuantTrio quants)