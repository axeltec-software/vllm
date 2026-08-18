# SPDX-License-Identifier: Apache-2.0
"""Model-runner side of mixed chat+PoC execution (counterpart of admission.py).

One bridge object per GPUModelRunner. Every hook is a no-op when the step holds
no PoC rows, so the pure-chat path is untouched. Embeds/routing/reflection are
applied IN-MODEL via native.py state (never through ``inputs_embeds`` — on 0.25
external embeds force the engine off the compiled path).

Runner anchors (5):
  load()        - load_model: attach native PoC state to the model
  pre_step()    - execute_model entry: decode-state slots for PoC rows
  pre_forward() - after _preprocess: row mask + per-row routing into native
  filter_rows() - logits: PoC rows excluded from sampling
  extract()     - post-forward: k-snap/chain from hidden states -> poc_outputs
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.poc import mixed_decode

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import PoCOutput

logger = init_logger(__name__)


class PoCRunnerBridge:
    def __init__(self, runner) -> None:
        self.runner = runner
        self.native = None
        self._step: dict[str, Any] | None = None  # per-step mixed-batch info

    # ------------------------------------------------------------- load_model
    def load(self, model: torch.nn.Module) -> None:
        from vllm.poc.native import attach_native_poc

        cfg = self.runner.vllm_config.cache_config
        try:
            self.native = attach_native_poc(
                model,
                layers=list(getattr(model.model, "layers", [])),
                embed_owner=model.model,
                max_tokens=cfg.poc_max_tokens,
                route_window=cfg.poc_route_window,
            )
        except Exception:
            logger.exception("PoC native attach failed; PoC disabled")
            self.native = None

    # --------------------------------------------------------- per-step hooks
    def pre_step(self, scheduler_output: "SchedulerOutput") -> None:
        poc_req_ids = getattr(scheduler_output, "poc_req_ids", None)
        if not poc_req_ids:
            self._step = None
            return
        # Deterministic order: nonce, not set-iteration (PYTHONHASHSEED).
        poc_requests = sorted(
            (self.runner.requests[rid] for rid in poc_req_ids
             if rid in self.runner.requests),
            key=lambda req: req.poc_params.nonce,
        )
        mixed_decode.setup_decode_poc(self.runner, poc_requests)
        self._step = {
            "poc_req_ids": poc_req_ids,
            "poc_requests": poc_requests,
            "poc_metadata": None,
            "poc_position_mask": None,
        }

    def pre_forward(self, scheduler_output: "SchedulerOutput",
                    positions: torch.Tensor, num_total_tokens: int) -> None:
        if self._step is None:
            if self.native is not None:
                self.native.set_mask(None)
            return
        embeds, _positions, mask, metadata = (
            mixed_decode.build_unified_mixed_batch_inputs(
                self.runner, scheduler_output, None, None, positions,
                self._step["poc_req_ids"], num_total_tokens,
            )
        )
        self._step["poc_metadata"] = metadata
        self._step["poc_position_mask"] = mask
        if self.native is not None and embeds is not None:
            self.native.set_embeds(embeds)
            self.native.set_mask(mask)

    def extract(self, hidden_states: torch.Tensor
                ) -> "dict[str, PoCOutput] | None":
        if self._step is None or self._step.get("poc_metadata") is None:
            return None
        out = mixed_decode.process_poc_outputs_from_hidden(
            self.runner, hidden_states, self._step["poc_metadata"])
        if out:
            mgr = mixed_decode.get_decode_manager(self.runner)
            for rid in out:
                mgr.free(rid)
        return out

    # ------------------------------------------------------ sampling exclusion
    def mixed_active(self) -> bool:
        return self._step is not None

    def _chat_rows(self) -> list[int]:
        rows = self._step.get("chat_rows")
        if rows is None:
            poc_ids = self._step["poc_req_ids"]
            rows = [
                i for i, rid in enumerate(
                    self.runner.input_batch.req_ids[
                        : self.runner.input_batch.num_reqs])
                if rid not in poc_ids
            ]
            self._step["chat_rows"] = rows
            self._step["num_reqs_snapshot"] = self.runner.input_batch.num_reqs
        return rows

    def compute_logits(self, sample_hidden_states: torch.Tensor
                       ) -> torch.Tensor | None:
        # PoC scores hidden states, not logits — never run the LM head for
        # PoC rows (0.20: _compute_logits_with_poc_filter).
        chat_rows = self._chat_rows()
        if not chat_rows:
            return None
        idx = torch.tensor(chat_rows, device=sample_hidden_states.device,
                           dtype=torch.long)
        return self.runner.model.compute_logits(sample_hidden_states[idx])

    def sample_chat_rows(self, logits, sampling_metadata):
        # Sample CHAT ROWS ONLY, scatter back into a full natural-order tensor
        # (PoC slots = 0, never read). 0.20: _sample mixed branch, verbatim.
        from vllm.v1.outputs import SamplerOutput

        runner = self.runner
        chat_rows = self._chat_rows()
        n_full = self._step["num_reqs_snapshot"]
        if not chat_rows:
            return SamplerOutput(
                sampled_token_ids=torch.zeros(
                    (n_full, 1), dtype=torch.int32, device=runner.device),
                logprobs_tensors=None,
            )
        chat_sm = mixed_decode.slice_sampling_metadata(
            sampling_metadata, chat_rows, runner.device)
        chat_out = runner.sampler(logits=logits, sampling_metadata=chat_sm)
        idx = torch.tensor(chat_rows, device=runner.device, dtype=torch.long)
        full = torch.zeros(
            (n_full, chat_out.sampled_token_ids.shape[1]),
            dtype=chat_out.sampled_token_ids.dtype, device=runner.device)
        full[idx] = chat_out.sampled_token_ids
        return SamplerOutput(
            sampled_token_ids=full,
            logprobs_tensors=chat_out.logprobs_tensors,
        )
