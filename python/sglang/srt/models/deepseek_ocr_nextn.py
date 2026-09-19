# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only DeepSeek-OCR NextN (FastMTP) speculative decoding.

Jina OCR (jinaai/jina-ocr-v1) ships a FastMTP draft head next to its
DeepSeek-OCR backbone under ``mtp_module.heads.0.*``: ``enorm`` / ``hnorm`` /
``eh_proj`` plus one dense ``DeepseekV2DecoderLayer`` (``mtp_moe=false``). The
head shares the backbone's embedding, final norm and lm_head
(``mtp_share_*=true``) and was trained recursively (``mtp_recursive=true``), so
each draft step feeds its own post-norm hidden state into the next one, which
is exactly SGLang's NextN/EAGLE draft loop.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekDecoderLayer
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class DeepseekOCRModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # layer_id 0 sits below first_k_dense_replace, so this builds the dense
        # MLP variant the FastMTP head was trained with (mtp_moe=false).
        self.decoder = DeepseekDecoderLayer(
            config,
            0,
            quant_config=quant_config,
            prefix=add_prefix("decoder", prefix),
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            # The target's prefill leaves its multimodal input embeddings on the
            # batch so the draft extend sees image features at image positions;
            # the last (rotated-in) token per request is embedded here.
            input_embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                assert input_embeds is not None
                last_indices = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                input_embeds[last_indices] = self.embed_tokens(input_ids[last_indices])
            if input_embeds is None:
                input_embeds = self.embed_tokens(input_ids)
        hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        hidden_states, residual = self.decoder(
            positions, hidden_states, forward_batch, residual
        )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states


class DeepseekOCRForCausalLMNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # `config` is the multimodal DeepseekVLV2Config; the head lives on the
        # language backbone.
        self.config = config
        self.text_config = config.text_config
        self.quant_config = quant_config
        self.model = DeepseekOCRModelNextN(
            self.text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            self.text_config.vocab_size,
            self.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
        )
        self.logits_processor = LogitsProcessor(self.text_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _map_checkpoint_name(self, name: str) -> Optional[str]:
        """Map a checkpoint tensor name onto this module, or None to skip it."""
        head_prefix = "mtp_module.heads.0."
        if name.startswith(head_prefix):
            name = name[len(head_prefix) :]
            if name.startswith("mtp_block."):
                return "model.decoder." + name[len("mtp_block.") :]
            # Self-contained heads (mtp_share_*=false) carry their own norm
            # and lm_head under shared_head.{_norm,norm,_head,local_head}.
            if name.startswith(("shared_head._norm.", "shared_head.norm.")):
                return "model.shared_head.norm." + name.split(".", 2)[2]
            if name.startswith(("shared_head._head.", "shared_head.local_head.")):
                return "lm_head." + name.split(".", 2)[2]
            return "model." + name  # enorm / hnorm / eh_proj
        if name == "mtp_embed_tokens.weight":
            return "model.embed_tokens.weight"
        if name == "model.embed_tokens.weight" and getattr(
            self.config, "mtp_share_embedding_weights", True
        ):
            return "model.embed_tokens.weight"
        if name == "model.norm.weight" and getattr(self.config, "mtp_share_norm", True):
            return "model.shared_head.norm.weight"
        if name == "lm_head.weight" and getattr(self.config, "mtp_share_lm_head", True):
            return "lm_head.weight"
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = self._map_checkpoint_name(name)
            if name is None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                f"Some FastMTP draft weights are not initialized from the checkpoint: "
                f"{unloaded_params}"
            )


EntryClass = [DeepseekOCRForCausalLMNextN]
