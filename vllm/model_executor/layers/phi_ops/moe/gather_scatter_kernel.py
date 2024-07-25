import torch
import triton
import triton.language as tl

import pytest

import vllm
from vllm import _custom_ops as ops

from typing import Tuple
from functools import wraps

import torch
import functools

def timeit_decorator(times=100):
    def decorator(function_call):
        @functools.wraps(function_call)
        def wrapper(*args, **kwargs):

            # cuda graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3):
                    function_call(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                function_call(*args, **kwargs)

            all_time = 0.0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for j in range(times):
                #function_call(*args, **kwargs)
                g.replay()

            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            all_time = elapsed_time_ms
            
            avg_time = all_time / times
            print(f"{function_call.__name__} average time: {avg_time} ms")
            return function_call(*args, **kwargs)
        
        return wrapper
    return decorator


@triton.jit
def moe_gather(
    a_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    M,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_cm,
    stride_ck,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    topk: tl.constexpr,
    splitk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // splitk
    pid_n = pid % splitk

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // topk * stride_am + offs_k[None, :] * stride_ak
    )

    c_ptrs = c_ptr + (offs_token_id[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    w_token_mask = offs_token_id < num_tokens_post_padded

    SPLITED_K = tl.cdiv(K, BLOCK_SIZE_K) // splitk

    a_ptrs = a_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ak
    c_ptrs = c_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ck

    for k in range(pid_n * SPLITED_K, (pid_n + 1) * SPLITED_K):
        a_mask = token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        c_mask = w_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        tl.store(c_ptrs, a, mask=c_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        c_ptrs += BLOCK_SIZE_K * stride_ck


@triton.jit
def moe_scatter(
    a_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    topk_weights_ptr,
    M,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_cm,
    stride_ck,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    topk: tl.constexpr,
    splitk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // splitk
    pid_n = pid % splitk

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + (offs_token_id[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_token_mask = offs_token_id < num_tokens_post_padded

    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    w_token_mask = offs_token < num_valid_tokens

    c_ptrs = c_ptr + (offs_token[:, None] * stride_cm + offs_k[None, :] * stride_ck)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=w_token_mask, other=0)

    SPLITED_K = tl.cdiv(K, BLOCK_SIZE_K) // splitk
    a_ptrs = a_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ak
    c_ptrs = c_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ck

    for k in range(pid_n * SPLITED_K, (pid_n + 1) * SPLITED_K):
        a_mask = a_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        c_mask = w_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        if MUL_ROUTED_WEIGHT:
            a = a * moe_weight[:, None]
        tl.store(c_ptrs, a, mask=c_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        c_ptrs += BLOCK_SIZE_K * stride_ck


def sparsemixer(scores, top_k, jitter_eps=0.01):
    assert top_k == 2

    ################ first expert ################

    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts = max_ind

    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    multiplier = multiplier_o

    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2 = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_moe_gather(
    inp,
    outp,
    sorted_token_ids,
    num_tokens_post_padded,
    topk_ids,
    block_m,
    block_k,
    topk,
    splitk=1,
):
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], block_m) * splitk,)

    moe_gather[grid](
        inp,
        outp,
        sorted_token_ids,
        num_tokens_post_padded,
        inp.size(0),
        inp.size(1),
        sorted_token_ids.size(0),
        topk_ids.numel(),
        inp.stride(0),
        inp.stride(1),
        outp.stride(0),
        outp.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        topk=topk,
        splitk=splitk,
    )


def invoke_moe_scatter(
    inp,
    outp,
    sorted_token_ids,
    num_tokens_post_padded,
    topk_ids,
    block_m,
    block_k,
    topk,
    splitk=1,
    topk_weights=None,
):
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], block_m) * splitk,)

    moe_scatter[grid](
        inp,
        outp,
        sorted_token_ids,
        num_tokens_post_padded,
        topk_weights,
        inp.size(0),
        inp.size(1),
        sorted_token_ids.size(0),
        topk_ids.numel(),
        inp.stride(0),
        inp.stride(1),
        outp.stride(0),
        outp.stride(1),
        topk_weights is not None,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        topk=topk,
        splitk=splitk,
    )
