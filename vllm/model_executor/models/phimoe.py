# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Microsoft team. All rights reserved.
#
# This code is modified based on mixtral.py
#
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

"""Inference-only PhiMoE model."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.utils import print_warning_once
import os

logger = logging.get_logger(__name__)

def is_sm80(device_id=0):
    if not torch.cuda.is_available():
        return False
    device_properties = torch.cuda.get_device_properties(device_id)
    return (device_properties.major == 8 and device_properties.minor == 0)


class PhiMoEConfig(PretrainedConfig):

    model_type = "phimoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        attention_bias=False,
        lm_head_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

def sparsemixer(scores, top_k, jitter_eps=0.1):
    assert top_k == 2

    ################ routing ################

    mask_logits_threshold, selected_experts = torch.topk(scores, 2)

    ################ first expert gating ################

    mask_logits_threshold_1 = mask_logits_threshold[:, 0].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_1)
    logits_mask = (
        (mask_logits_threshold_1 - scores) / factor
    ) > (2 * jitter_eps)

    multiplier_1 = torch.softmax(
      scores.masked_fill(logits_mask, float('-inf')), dim=-1
    ).gather(dim=-1, index=selected_experts[:, 0].unsqueeze(-1))

    ################ second expert gating ################

    mask_logits_threshold_2 = mask_logits_threshold[:, 1].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_2)
    logits_mask = (
        (mask_logits_threshold_2 - scores) / factor
    ) > (2 * jitter_eps)

    multiplier_2 = torch.softmax(
      torch.scatter(
        scores, -1, selected_experts[:, 0].unsqueeze(-1), float('-inf')
      ).masked_fill(
        logits_mask, float('-inf')
      ), 
      dim=-1
    ).gather(dim=-1, index=selected_experts[:, 1].unsqueeze(-1))

    multiplier = torch.concat((multiplier_1, multiplier_2), dim=-1)
    return (
        multiplier, 
        selected_experts,
    )


class PhiMoE(nn.Module):
    """A tensor-parallel MoE implementation for PhiMoE that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        remove_fp8_subnormal = False,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        # FIXME(pcmoritz): Make this more general to support different
        # quantization 
        self.use_fp8 = isinstance(quant_config, Fp8Config)
        self.apply_a100_fp8 = is_sm80() and self.use_fp8
        self.remove_fp8_subnormal = remove_fp8_subnormal
        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     params_dtype=self.params_dtype,
                                     quant_config=None)
        if self.use_fp8:
            cpu_or_gpu = 'cpu'
        else:
            cpu_or_gpu = 'cuda'
        self.ws = nn.Parameter(
            torch.empty(self.num_total_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device=cpu_or_gpu,
                        dtype=self.params_dtype))
        self.w2s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device=cpu_or_gpu,
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
        })

        # Scaling factors for FP8 weights
        self.ws_scale = nn.Parameter(
            torch.ones(
                self.num_total_experts, device="cuda", dtype=torch.float32),
            requires_grad=False) if self.use_fp8 else None
        self.w2s_scale = nn.Parameter(
            torch.ones(
                self.num_total_experts, device="cuda", dtype=torch.float32),
            requires_grad=False) if self.use_fp8 else None

        # Scaling factors for FP8 activations
        need_act_scales = (self.use_fp8
                           and False)
        self.as_scale = nn.Parameter(
            torch.zeros(1, device="cuda", dtype=torch.float32),
            requires_grad=False) if need_act_scales else None
        self.a2s_scale = nn.Parameter(
            torch.zeros(1, device="cuda", dtype=torch.float32),
            requires_grad=False) if need_act_scales else None

        if need_act_scales:
            set_weight_attrs(self.as_scale, {
                "weight_loader": self.weight_loader,
            })
            set_weight_attrs(self.a2s_scale, {
                "weight_loader": self.weight_loader,
            })
        
        if self.apply_a100_fp8:
            import vllm.model_executor.layers.phi_ops.moe.tensorrt_llm_moe.ampere_fp8_fused_moe as ampere_fp8_fused_moe
            self.fused_moe_forward = ampere_fp8_fused_moe.fused_moe
        else:
            self.fused_moe_forward = fused_moe

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]
        if "act_scale" in weight_name:
            param_data[:] = param_data[:].max(loaded_weight)

    def process_weights_after_loading(self):
        if self.use_fp8:
            ws = torch.empty_like(self.ws.data, dtype=torch.float8_e4m3fn)
            w2s = torch.empty_like(self.w2s.data, dtype=torch.float8_e4m3fn)
            for expert in range(self.num_total_experts):
                ws[expert, :, :], self.ws_scale[expert] = ops.scaled_fp8_quant(
                    self.ws.data[expert, :, :])
                w2s[expert, :, :], self.w2s_scale[
                    expert] = ops.scaled_fp8_quant(self.w2s.data[expert, :, :])

            def remove_subnormal_fp8(tensor):
                assert tensor.dtype == torch.uint8, "Tensor must be a byte tensor representing fp8 values"
                exponent_mask = 0b11111000
                mantissa_mask = 0b00000111
                exponents = (tensor & exponent_mask) >> 3
                mantissas = tensor & mantissa_mask
                subnormal_mask = (exponents == 0) & (mantissas != 0)
                if subnormal_mask.any():
                    print(subnormal_mask.sum().item() / subnormal_mask.numel() * 100, "% of values are subnormal")
                tensor[subnormal_mask] = 0
                return subnormal_mask.any()

            if self.remove_fp8_subnormal:
                remove_subnormal_fp8(ws.view(torch.uint8))
                remove_subnormal_fp8(w2s.view(torch.uint8))

            self.ws = nn.Parameter(ws.to("cuda"), requires_grad=False)
            self.w2s = nn.Parameter(w2s.to("cuda"), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert False == self.training
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.fused_moe_forward(hidden_states,
                                                    self.ws,
                                                    self.w2s,
                                                    router_logits,
                                                    self.top_k,
                                                    renormalize=False,
                                                    inplace=True,
                                                    use_fp8=self.use_fp8,
                                                    w1_scale=self.ws_scale,
                                                    w2_scale=self.w2s_scale,
                                                    a1_scale=self.as_scale,
                                                    a2_scale=self.a2s_scale,
                                                    routing_func=sparsemixer)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


class PhiMoEAttention(nn.Module):

    def __init__(self,
                 config: PhiMoEConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 sliding_window: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        # self.sliding_window = sliding_window

        if isinstance(quant_config, Fp8Config):
            print_warning_once(
                "For PhiMoE FP8 quantization, we currently do not quantize "
                "the attention layers until their FP8 performance is improved."
            )
            quant_config = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        # print ("checking configs", config.attention_bias)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
            rope_scaling=getattr(config, 'rope_scaling', None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            # sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class PhiMoEDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PhiMoEConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = PhiMoEAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            # sliding_window=config.sliding_window,
            quant_config=quant_config)

        self.block_sparse_moe = PhiMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       elementwise_affine=True)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                elementwise_affine=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states, residual


class PhiMoEModel(nn.Module):

    def __init__(
        self,
        config: PhiMoEConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            PhiMoEDecoderLayer(config, cache_config, quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], attn_metadata,
                                            residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class PhiMoEForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: PhiMoEConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = PhiMoEModel(config,
                                 cache_config,
                                 quant_config,
                                 lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            bias=True
        )
        
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # These are the weights for the experts
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["w1", "w3"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ] + [
            # These are the activation scales for the experts
            # (param_name, weight_name, expert_id)
            ("as_scale" if weight_name in ["w1", "w3"] else "a2s_scale",
             f"experts.{expert_id}.{weight_name}.act_scale", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)