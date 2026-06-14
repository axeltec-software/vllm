# PoC Test Suite

## Configuration & Shared Infrastructure

**[conftest.py](conftest.py)** — Session-scoped fixtures for all integration tests.
- `--poc-model`: HuggingFace model to serve (default: `RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16`)
- `--poc-port`: Port of an already-running vLLM server (skips auto-launch)

**[utils.py](utils.py)** — Shared helpers: `poc_request_body()`, `decode_artifact_vector()`, `check_artifact()`

**[_server.py](_server.py)** — `PoCTestServer` context manager: launches `vllm serve` subprocess, waits for `/health`, captures logs.

---

## Unit Tests

### [unit/test_layer_hook_math.py](unit/test_layer_hook_math.py)
Tests Householder reflection math and layer hook behavior.

| Test | Description |
|------|-------------|
| `TestHouseholderMath::test_self_inverse` | H(H(x)) == x |
| `TestHouseholderMath::test_norm_preservation` | Householder is an isometry |
| `TestHouseholderMath::test_reflection_of_v` | H(v) == -v |
| `TestHouseholderMath::test_orthogonal_to_v_unchanged` | Orthogonal vectors unchanged |
| `TestHouseholderMath::test_deterministic_from_seed` | Same seed → same vector |
| `TestHouseholderMath::test_different_seeds_differ` | Different seeds → different vectors |
| `TestHouseholderMath::test_unit_norm` | Generated Householder vector has unit norm |
| `TestHookContext::test_no_context_flag_false` | `is_poc_forward_active()` is False outside context |
| `TestHookContext::test_binary_context_sets_flag` | `poc_forward_context()` sets/clears flag |
| `TestHookContext::test_mask_context_sets_mask` | `poc_forward_context_with_mask()` sets global mask |
| `TestHookContext::test_context_restores_on_exception` | Context managers restore state on exception |
| `TestHookTransformation::test_hook_count_matches_layers` | One hook per transformer layer |
| `TestHookTransformation::test_detach_removes_all_hooks` | After `detach()`, num_layers == 0 |
| `TestHookTransformation::test_no_transform_without_context` | Layer output unchanged without PoC context |
| `TestHookTransformation::test_binary_context_transforms_all` | All positions differ with `poc_forward_context` |
| `TestHookTransformation::test_mask_mode_transforms_only_poc_positions` | Only selected positions differ with mask |
| `TestHookTransformation::test_multiple_attach_detach_cycles` | Repeated attach/detach leaves no dangling hooks |
| `TestHookTransformation::test_tuple_output_both_components_transformed` | Both hidden and residual components transformed |

**Run:** `pytest tests/poc/unit/test_layer_hook_math.py`

---

### [unit/test_check_params_match.py](unit/test_check_params_match.py)
Tests `routes.check_params_match` validation logic.

| Test | Description |
|------|-------------|
| `test_no_deployed_config_is_noop` | No deployment config → no validation |
| `test_matching_params_ok` | Matching params pass without error |
| `test_max_tokens_mismatch_raises_409` | max_tokens mismatch → HTTP 409 |
| `test_seq_len_mismatch_raises_409` | seq_len mismatch → HTTP 409 |
| `test_k_dim_mismatch_raises_409` | k_dim mismatch → HTTP 409 |
| `test_max_tokens_zero_is_a_real_configured_value` | max_tokens=0 (prefill-only) is valid |
| `test_requested_detail_includes_max_tokens` | Error detail includes requested max_tokens |

**Run:** `pytest tests/poc/unit/test_check_params_match.py`

---

### [unit/test_layer_hooks.py](unit/test_layer_hooks.py)
Tests `LayerHouseholderHook` attachment and selective transformation.

| Test | Description |
|------|-------------|
| `TestLayerHouseholderHookAttachment::test_hooks_attach_to_all_layers` | Hooks attach to all transformer layers |
| `TestLayerHouseholderHookAttachment::test_hooks_detach_cleanly` | Hooks detach without errors |
| `TestLayerHouseholderHookAttachment::test_multiple_attach_detach_cycles` | Multiple attach/detach cycles work |
| `TestHooksSelectiveTransformation::test_mixed_batch_transforms_only_poc_positions` | Only PoC positions transformed in mixed batches |
| `TestHooksSelectiveTransformation::test_pure_poc_batch_transforms_all_positions` | All positions transformed in pure PoC batch |
| `TestHooksSelectiveTransformation::test_no_context_no_transformation` | No transformation without PoC context |
| `TestMaskPaddingAndAlignment::test_mask_smaller_than_hidden_states` | Handles mask with fewer elements than hidden states |
| `TestMaskPaddingAndAlignment::test_mask_larger_than_hidden_states` | Handles mask with more elements than hidden states |
| `TestMaskPaddingAndAlignment::test_empty_mask` | Empty mask behaves same as no context |

**Run:** `pytest tests/poc/unit/test_layer_hooks.py`

---

### [unit/test_distributed_poc.py](unit/test_distributed_poc.py)
Tests distributed TP/PP logic, sphere projection, and codebook.

| Test | Description |
|------|-------------|
| `TestSphereProjection::test_unit_norm_single` | Single vector normalized to unit sphere |
| `TestSphereProjection::test_unit_norm_batch` | All batch rows normalized |
| `TestSphereProjection::test_zero_vector_safe` | Zero vector → no NaN |
| `TestSphereProjection::test_already_unit_unchanged` | Unit-norm vector passes through unchanged |
| `TestSphereCodebook::test_shape` | Codebook shape is `[SPHERE_POINTS, SPHERE_DIM]` |
| `TestSphereCodebook::test_all_unit_norm` | Every codebook point is unit vector |
| `TestSphereCodebook::test_no_nan_or_inf` | Codebook contains only finite values |
| `TestSphereCodebook::test_minimum_pairwise_distance` | All codebook points are distinct |
| `TestSphereCodebook::test_deterministic` | `build_equidistant_codebook` is deterministic |
| `TestNearestSphereIndex::test_values_in_range` | Indices in `[0, SPHERE_POINTS)` |
| `TestNearestSphereIndex::test_exact_codebook_point` | Query == codebook point → that point's index |
| `TestNearestSphereIndex::test_batch_shape` | Output shape matches batch size |
| `TestNearestSphereIndex::test_deterministic` | Same query → same index |
| `TestTPPPContract::test_tp_broadcast_required_keys` | TP driver broadcasts required keys |
| `TestTPPPContract::test_tp_decode_step_broadcast_key` | TP driver broadcasts prev_k during decode |
| `TestTPPPContract::test_pp_non_last_rank_must_return_none` | Non-last PP rank returns None |
| `TestTPPPContract::test_pp_decode_disabled_when_world_size_gt_1` | Decode skipped when pp_group.world_size > 1 |
| `TestTPPPContract::test_pp_first_rank_generates_inputs_embeds` | Only first PP rank generates inputs_embeds |

**Run:** `pytest tests/poc/unit/test_distributed_poc.py`

---

### [unit/test_kv_reservation.py](unit/test_kv_reservation.py)
Tests KV-cache block reservation math and BlockPool carving.

| Test | Description |
|------|-------------|
| `TestReservationMath::test_formula_matches_physical_layout` | Reservation formula matches physical memory layout |
| `TestReservationMath::test_prefill_only_max_tokens_zero` | max_tokens=0 (prefill-only) handled correctly |
| `TestReservationMath::test_partial_block_rounds_up` | Partial blocks round up |
| `TestReservationMath::test_scales_linearly_with_batch` | Scales linearly with batch size |
| `TestReservationMath::test_reserved_blocks_reads_config` | `poc_reserved_blocks()` reads CacheConfig correctly |
| `TestBlockPoolCarve::test_reserves_low_block_ids` | Reserved block IDs are exactly `[0, reserved)` |
| `TestBlockPoolCarve::test_chat_free_pool_excludes_reserved` | Chat never uses reserved or null blocks |
| `TestBlockPoolCarve::test_zero_reservation_only_removes_null` | Zero reservation only holds null block |
| `TestBlockPoolCarve::test_default_is_backward_compatible` | Default `poc_reserved_blocks=0` path intact |

**Run:** `pytest tests/poc/unit/test_kv_reservation.py`

---

### [unit/test_mixed_decode.py](unit/test_mixed_decode.py)
Tests step-driven mixed decode-PoC bookkeeping.

| Test | Description |
|------|-------------|
| `test_per_slot_blocks_matches_reservation_per_seq` | Per-slot sizing matches per-seq |
| `test_slots_tile_the_reservation_exactly` | Slots tile reservation without gaps/overlap |
| `test_slot_block_ids_contiguous_and_offset` | Slot block IDs are contiguous and correctly offset |
| `test_manager_allocate_free_reuse` | Manager allocates/frees/reuses slots correctly |
| `test_decode_state_defaults` | Decode state initializes with correct defaults |
| `test_apply_poc_kv_skip_pads_prefill_only_keeps_decode` | Prefill-only PoC is PADded; decode-PoC kept |
| `test_apply_poc_kv_skip_noop_when_no_prefill_only` | No-op when no prefill-only positions |

**Run:** `pytest tests/poc/unit/test_mixed_decode.py`

---

## Integration Tests

All integration tests require a running vLLM server. They are marked `@pytest.mark.integration` and start one automatically unless `--poc-port` is passed.

**Common run pattern:**
```bash
pytest tests/poc/integration/ -m integration [--poc-model MODEL] [--poc-port PORT]
```

---

### [integration/test_all_modes.py](integration/test_all_modes.py)
Smoke tests covering all PoC request modes.

| Test | Description |
|------|-------------|
| `TestPureChat::test_simple_completion` | Chat endpoint returns non-empty response |
| `TestPurePoC::test_generate_returns_valid_artifacts` | Generate endpoint returns completed status with valid artifacts |
| `TestPurePoC::test_artifact_vector_shape` | Artifact vector has expected k_dim length |
| `TestMixedBatch::test_concurrent_chat_and_poc` | Simultaneous chat and PoC requests complete successfully |
| `TestDifferentBlockHash::test_distinct_hashes_produce_distinct_vectors` | Different block_hashes → distinct vectors |
| `TestHighConcurrency::test_concurrent_requests_all_succeed` | 10 chat + 10 PoC concurrent requests all succeed |
| `TestChatNotTruncatedByPoC::test_single_chat_not_truncated` | Long chat not truncated by concurrent PoC |
| `TestChatNotTruncatedByPoC::test_multiple_chats_not_truncated` | 3 concurrent chats not truncated by mid-stream PoC |

**Run:** `pytest tests/poc/integration/test_all_modes.py -m integration`

---

### [integration/test_poc_decode.py](integration/test_poc_decode.py)
Tests decode-PoC structure, entropy, and determinism.

| Test | Description |
|------|-------------|
| `TestDecodeStepStructure::test_kpoints_steps_length` | With max_tokens=N, k_points_steps has N+1 entries |
| `TestDecodeStepStructure::test_kpoints_steps_all_in_range` | All k_points_steps values in `[0, SPHERE_POINTS)` |
| `TestDecodeStepStructure::test_sphere_k_present_with_decode` | Prefill sphere_k present and equals k_points_steps[0] |
| `TestDecodeEntropy::test_kpoints_steps_not_constant` | 20 nonces × 10 decode steps → k values not all same |
| `TestDecodeEntropy::test_kpoints_steps_entropy_above_threshold` | Shannon entropy > 1.0 bit |
| `TestDecodeEntropy::test_distinct_nonces_distinct_kpoints` | Different nonces → different k_points_steps |
| `TestDecodeDeterminism::test_same_request_same_kpoints` | Identical request → identical k_points_steps |
| `TestDecodeDeterminism::test_same_request_same_final_vector` | Decode produces same final vector on repeat |

**Run:** `pytest tests/poc/integration/test_poc_decode.py -m integration`

---

### [integration/test_cudagraph_equivalence.py](integration/test_cudagraph_equivalence.py)
Tests that CUDA graph is a pure speed optimization (no behavioral change).

| Test | Description |
|------|-------------|
| `test_eager_vs_cudagraph_byte_identical` | **SKIPPED** — Pending byte-identical validation |
| `test_decode_trajectory_length_matches_max_tokens` | Both eager and cudagraph produce full prefill+decode trajectory |

**Run:** `pytest tests/poc/integration/test_cudagraph_equivalence.py -m integration`

---

### [integration/test_chat_quality.py](integration/test_chat_quality.py)
Tests chat outputs are not corrupted when mixed with concurrent PoC requests.

| Test | Description |
|------|-------------|
| `TestChatQualityUnderLoad::test_no_corrupted_outputs` | 20 chat responses valid under concurrent PoC load |
| `TestChatQualityUnderLoad::test_math_answers_are_correct` | Arithmetic answers correct under concurrent PoC load |

**Run:** `pytest tests/poc/integration/test_chat_quality.py -m integration`

---

### [integration/test_artifact_validity.py](integration/test_artifact_validity.py)
Tests artifact vector validity and distinctness.

| Test | Description |
|------|-------------|
| `TestVectorContent::test_no_nan_or_inf` | All vector components are finite |
| `TestVectorContent::test_unit_norm` | Each artifact vector has L2 norm ≈ 1.0 |
| `TestVectorContent::test_artifact_count_matches_nonce_count` | Artifact count equals requested nonce count |
| `TestVectorContent::test_artifact_nonce_values_match_input` | Artifact nonces match input nonces |
| `TestDeterminism::test_same_request_same_vectors` | Identical request → identical vectors (run twice) |
| `TestNonceIndependence::test_all_nonce_vectors_distinct` | 10 nonces → 10 distinct vectors |
| `TestNonceIndependence::test_different_nonces_different_from_each_other` | Nonce vectors far apart (L2 > 0.1) |
| `TestBlockHashEffect::test_different_hashes_different_vectors` | Different block_hashes → distinct vectors for same nonce |

**Run:** `pytest tests/poc/integration/test_artifact_validity.py -m integration`

---

### [integration/test_kv_cache_integrity.py](integration/test_kv_cache_integrity.py)
Tests that KV cache is not corrupted by PoC.

| Test | Description |
|------|-------------|
| `TestChatReproducibility::test_chat_unchanged_after_single_poc_round` | Chat response identical before/after one PoC round |
| `TestChatReproducibility::test_chat_unchanged_after_multiple_poc_rounds` | Chat response identical across 5 PoC rounds |
| `TestChatReproducibility::test_chat_works_immediately_after_poc` | Chat succeeds immediately after PoC (no lock held) |
| `TestChatQualityAfterPoC::test_factual_answer_correct_after_poc` | Model gives correct factual answer after PoC round |
| `TestChatQualityAfterPoC::test_arithmetic_correct_after_poc` | Simple arithmetic correct after PoC round |

**Run:** `pytest tests/poc/integration/test_kv_cache_integrity.py -m integration`

---

### [integration/test_mixed_decode_concurrent_chat.py](integration/test_mixed_decode_concurrent_chat.py)
Tests decode-PoC running concurrently with live chat in the same forward pass.

**Requires:** `VLLM_POC_MIXED_DECODE=1`, CUDA graphs ON

| Test | Description |
|------|-------------|
| `test_poc_decode_reproducible_under_concurrent_chat` | PoC artifacts byte-identical whether or not chat runs concurrently; chat not empty; mixed batch confirmed |

**Run:**
```bash
VLLM_POC_MIXED_DECODE=1 pytest tests/poc/integration/test_mixed_decode_concurrent_chat.py -m integration
```

---

### [integration/test_mixed_decode_parity.py](integration/test_mixed_decode_parity.py)
Acceptance gates for Phase-2 mixed decode-PoC. Uses L2 tolerance and statistical comparison (not byte-identical).

Tolerance: `MAX_STEP_MISMATCH_RATE = 0.15` (15% per-step sphere_k mismatch allowed)

| Test | Description |
|------|-------------|
| `test_mixed_decode_matches_pure` | Full k_points trajectory matches pure path within tolerance |
| `test_mixed_decode_is_kv_bound` | Changing block_hash changes trajectory (verifies KV dependency) |
| `test_mixed_decode_deterministic_enough` | Mixed decode reproducible run-to-run within tolerance |
| `test_mixed_decode_multi_nonce_no_crash` | Multiple concurrent nonces (sizes 1–8) don't crash engine |
| `test_mixed_decode_trajectory_length` | Mixed decode produces full prefill + max_tokens trajectory |

**Run:**
```bash
VLLM_POC_MIXED_DECODE=1 pytest tests/poc/integration/test_mixed_decode_parity.py -m integration
```

---

## Quick Reference

| Scope | Command |
|-------|---------|
| All unit tests | `pytest tests/poc/unit/` |
| All integration tests | `pytest tests/poc/integration/ -m integration` |
| Mixed decode tests only | `VLLM_POC_MIXED_DECODE=1 pytest tests/poc/integration/test_mixed_decode_*.py -m integration` |
| Against existing server | `pytest tests/poc/ -m integration --poc-port 8000` |
| Different model | `pytest tests/poc/ --poc-model <HF_MODEL_ID>` |
