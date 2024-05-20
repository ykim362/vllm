import vllm
import torch
from vllm import _custom_ops as ops
import argparse
import json
import sys
import os
import shutil

# pip install cupy-cuda12x

import cupy
import fused_moe
import ampere_fp8_fused_moe


def moe_perf(
    tokens=1024,
    experts=8,
    topk=2,
    intermediate_size=14336,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)

    hidden_state = torch.randn(tokens, hidden_size).cuda().half()


    if use_fp8:
        w1, ws_scale = ops.scaled_fp8_quant(
            torch.randn(experts, intermediate_size * topk, hidden_size).cuda().half()
        )
        w2, w2s_scale = ops.scaled_fp8_quant(
            torch.randn(experts, hidden_size, intermediate_size).cuda().half()
        )
        _, h_scale = ops.scaled_fp8_quant(hidden_state)
        ws_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device)
        w2s_scale = torch.ones(experts, dtype=ws_scale.dtype, device=ws_scale.device)
    else:
        w1 = torch.randn(experts, intermediate_size * topk, hidden_size).cuda().half()
        w2 = torch.randn(experts, hidden_size, intermediate_size).cuda().half()
        h_scale = None
        ws_scale = None
        w2s_scale = None


    gatew = torch.randn(hidden_size, experts).cuda().half()
    gating_output = torch.matmul(hidden_state, gatew).float()

    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        ampere_fp8_fused_moe.fused_moe(
            hidden_states=hidden_state,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
            topk=topk,
            override_config=config,
            renormalize=True,
            inplace=True,
            use_fp8=use_fp8,
            w1_scale=ws_scale,
            w2_scale=w2s_scale,
            a1_scale=h_scale,
            a2_scale=h_scale,
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)

        if j >= 10:
            all_time += elapsed_time_ms

    return all_time / times


searchspace = [1] + list(range(0, 256, 32))[1:] + list(range(256, 4097, 256))
#searchspace = [1, 32]
intermediate_size = 6400
expert_num = 16

for tk in searchspace:
    print(
        tk,
        ",",
        moe_perf(tokens=tk, experts=expert_num, intermediate_size=intermediate_size, use_fp8=True),
    )
