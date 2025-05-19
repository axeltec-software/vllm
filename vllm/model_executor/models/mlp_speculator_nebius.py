import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import LlamaMLP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.layers.activation import NewGELU


@support_torch_compile
class MLPProjection(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        self.dense = nn.Linear(self.config.hidden_size * 2,
                               self.config.hidden_size,
                               bias=False)
        self.act = NewGELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        return hidden_states


@support_torch_compile
class MLPSpeculatorHead(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        
        self.projection = MLPProjection(vllm_config=vllm_config, prefix=prefix)

        self.ffn_layers = nn.ModuleList([
            LlamaMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act='silu',
                bias=getattr(self.config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
            for _ in range(self.config.num_hidden_layers)
        ])

        eps = getattr(self.config, "rms_norm_eps", 1e-5)
        self.embeddings_norm = RMSNorm(self.config.hidden_size, eps=eps)
        self.outputs_norm = RMSNorm(self.config.hidden_size, eps=eps)
        self.features_norm = RMSNorm(self.config.hidden_size, eps=eps)
        
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        input_embeds = self.embeddings_norm(input_embeds)
        hidden_states = self.features_norm(hidden_states)

        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)
        hidden_states = self.projection(hidden_states)

        for ffn_layer in self.ffn_layers:
            hidden_states = hidden_states + ffn_layer(hidden_states)
        
        return self.outputs_norm(hidden_states), hidden_states


@support_torch_compile
class MLPSpeculatorHeads(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config

        num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens  # noqa E501
        dtype = vllm_config. \
            speculative_config.draft_model_config.dtype

        with set_default_torch_dtype(dtype), set_current_vllm_config(
                vllm_config):
            self.blocks = nn.ModuleList([
                MLPSpeculatorHead(vllm_config=vllm_config, prefix=prefix)
                for _ in range(num_speculative_tokens)
            ])
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            )
        self.logits_processor = LogitsProcessor(
            vocab_size=self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        result = []
        hidden_states_prenorm = hidden_states
        for layer in self.blocks:
            hidden_states, hidden_states_prenorm = layer(
                input_ids,
                hidden_states_prenorm,
            )
            logits = self.logits_processor(self.lm_head, hidden_states)
            input_ids = logits.argmax(dim=-1)
            result.append(input_ids)
        return result
