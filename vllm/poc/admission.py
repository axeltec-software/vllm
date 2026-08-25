# SPDX-License-Identifier: Apache-2.0
"""Per-step PoC admission policy for the V1 scheduler.

All chat<->PoC mixing policy lives here so ``schedule()`` carries only a handful
of delegating calls. Construction is cheap and ``active`` is False whenever the
step holds no PoC request, in which case every method is a no-op and the
pure-chat path is untouched.

Policy (ported from the 0.20 in-tree branch, ``vllm/poc/mixed_decode.py``):
  * chat and PoC share a forward only while BOTH are decoding; either side's
    prefill runs isolated, so a mixed step stays uniform-decode shaped and lands
    on a captured cudagraph rung;
  * ``poc_share`` splits the step's token budget so PoC cannot starve chat;
  * ``poc_max_batch_size`` caps PoC rows per step;
  * a defer valve bounds consecutive chat-prefill defers so chat churn cannot
    starve a decoding PoC.
"""

from typing import TYPE_CHECKING

from vllm.poc.mixed_decode import poc_cfg
from vllm.poc.mixed_decode import (
    POC_DEFER_LIMIT,
    decode_only_mixing_gate,
    poc_alloc_footprint,
    poc_kv_capacity,
    poc_share_budget,
    poc_step_num_tokens,
    resolve_poc_max_batch_size,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class PoCAdmission:
    """Decides which PoC/chat requests may enter the current forward."""

    __slots__ = ("active", "_defer_chat", "_defer_poc", "_max_batch",
                 "_token_budget", "_scheduled", "_tokens", "_poc_prefill")

    def __init__(self, scheduler, token_budget: int) -> None:
        queues = (scheduler.running, scheduler.waiting)
        # ONE pass for all four flags. As four separate any() calls, the ones whose
        # answer is "no" walk the whole of running+waiting every step — and on a
        # PoC-only node both chat questions are exactly that, so the queues were
        # scanned twice per step for nothing.
        poc_present = chat_present = False
        poc_will_prefill = chat_will_prefill = False
        for q in queues:
            for r in q:
                if r.poc_params is not None:
                    poc_present = True
                    if r.num_computed_tokens == 0:
                        poc_will_prefill = True
                else:
                    chat_present = True
                    if r.num_computed_tokens < r.num_prompt_tokens:
                        chat_will_prefill = True
                if (poc_present and chat_present
                        and poc_will_prefill and chat_will_prefill):
                    break
            else:
                continue
            break
        self.active = poc_present
        if not self.active:
            return

        cache_config = scheduler.cache_config
        # Resolve here, not at config init: num_gpu_blocks is only known once
        # the engine has profiled free memory and built the KV pool.
        self._max_batch = resolve_poc_max_batch_size(
            poc_cfg(cache_config, "poc_max_batch_size"),
            scheduler.scheduler_config.max_num_seqs,
            poc_kv_capacity(
                getattr(cache_config, "num_gpu_blocks", 0),
                getattr(cache_config, "block_size", 0),
                poc_cfg(cache_config, "poc_seq_len"),
                poc_cfg(cache_config, "poc_max_tokens"),
            ),
        )
        self._token_budget = poc_share_budget(
            poc_cfg(cache_config, "poc_share"), token_budget, chat_present)
        self._scheduled = 0
        self._tokens = 0

        # Vestigial in the 0.20 branch (hardcoded False): PoC never demands an
        # exclusive pure-decode step, so chat+PoC decode freely share a forward.
        poc_decode_pending = False
        self._poc_prefill = poc_will_prefill
        self._defer_chat, self._defer_poc, scheduler._poc_defers = (
            decode_only_mixing_gate(
                mixed_cudagraph=True,
                poc_decode_pending=poc_decode_pending,
                poc_will_prefill=poc_will_prefill,
                chat_will_prefill=chat_will_prefill,
                consecutive_defers=getattr(scheduler, "_poc_defers", 0),
                defer_limit=POC_DEFER_LIMIT,
            )
        )

    def skip(self, request: "Request") -> bool:
        """True if this request must not be scheduled into this step."""
        if not self.active:
            return False
        if request.poc_params is None:
            return self._defer_chat
        if self._defer_poc or self._scheduled >= self._max_batch:
            return True
        # Keep a step uniform: never mix a PoC prefill with PoC decode rows.
        return self._poc_prefill and request.num_computed_tokens > 0

    def num_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """Token count for a PoC row; chat rows pass through unchanged."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_step_num_tokens(request.poc_params, request.num_computed_tokens)

    def over_budget(self, request: "Request", num_new_tokens: int) -> bool:
        """True once PoC has consumed its slice of this step's token budget."""
        if not self.active or request.poc_params is None:
            return False
        return self._tokens + num_new_tokens > self._token_budget

    def alloc_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """KV footprint to reserve via the shared KVCacheManager."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_alloc_footprint(request.poc_params, num_new_tokens)

    def note_scheduled(self, request: "Request", num_new_tokens: int) -> None:
        if not self.active or request.poc_params is None:
            return
        self._scheduled += 1
        self._tokens += num_new_tokens
