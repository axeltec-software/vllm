# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config, set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.v1.sample.metadata import SamplingMetadata

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mlp_speculator_nebius import MLPSpeculatorHeads
from vllm.compilation.backends import set_model_tag


class MLPProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        prefix: str = "",
    ):
        self.vllm_config = vllm_config
        self._speculative_config = vllm_config.speculative_config
        self._model_config = self._speculative_config.draft_model_config
        self._load_config = vllm_config.load_config

        hidden_size = self._model_config.get_hidden_size()

        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        with set_default_torch_dtype(
                self._model_config.dtype), set_current_vllm_config(
                    vllm_config):
            
            with set_model_tag("mlp_head"):
                self.speculator = MLPSpeculatorHeads(vllm_config=vllm_config,
                                                    prefix=prefix).to(device)
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, hidden_size),
                dtype=self._model_config.dtype,
                device=device,
            )
            self.input_ids = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device=device,
            )

    def propose(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(dim=0)

        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            num_input_tokens = batch_size

        self.hidden_states[:batch_size, :] = hidden_states
        self.input_ids[:batch_size] = input_ids
        with set_forward_context(None,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            draft_token_ids = self.speculator(
                self.input_ids[:num_input_tokens],
                self.hidden_states[:num_input_tokens])
            draft_token_ids = torch.stack(draft_token_ids, dim=-1)

        # [batch_size, num_speculative_tokens]
        return draft_token_ids[:batch_size, :]

    def load_model(self, target_model: nn.Module) -> None:
        loader = get_model_loader(self._load_config)
        weights = loader.get_all_weights(self._model_config, self.speculator)

        # Load embedding and unembedding
        for block in self.speculator.blocks:
            block.embed_tokens = target_model.model.embed_tokens
        self.speculator.lm_head = target_model.lm_head

        self.attn_layer_name = list(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())[0]

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.speculator.named_parameters())
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            self.speculator(
                input_ids=self.input_ids[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )
