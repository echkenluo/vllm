# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 indexer head gate for decode-sized batches.

The indexer computes its per-head gate as ``hidden_states.float() @ W_fp32``
([M, 4096] x [4096, 32]) in FP32 on purpose (bf16 error can change near-tie
pool rankings). For a handful of rows cuBLAS picks a small-N kernel that runs
~30-45 us on L20 for 0.5 MB of weights. This computes the same FP32 product
with a two-stage split-K: stage 1 gives every (N block, K chunk) its own
program and writes FP32 partial sums, stage 2 adds the partials in a fixed
order. Inputs are read as BF16 and widened in-kernel (exact), products and
sums are FP32, so the result differs from torch.mm only in summation order,
and it is deterministic. Only for M <= MAX_ROWS; larger batches keep torch.mm.
"""

import torch

from vllm.triton_utils import tl, triton

MAX_ROWS = 16
_SPLIT_K = 128
_BN = 16


@triton.jit
def _gate_partial_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    p_ptr,
    M,
    N,
    K,
    K_PER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = pid_k * K_PER + tl.arange(0, K_PER)
    x = tl.load(
        x_ptr + rm[:, None] * x_stride + rk[None, :],
        mask=(rm[:, None] < M) & (rk[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        w_ptr + rn[:, None] * K + rk[None, :],
        mask=(rn[:, None] < N) & (rk[None, :] < K),
        other=0.0,
    )
    acc = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    tl.store(
        p_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _gate_reduce_kernel(p_ptr, o_ptr, MN, S: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(S):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, acc, mask=mask)


def indexer_gate_fp32(x: torch.Tensor, w_nk: torch.Tensor) -> torch.Tensor:
    """FP32 ``x.float() @ w_nk.t()`` for x [M, K] (M <= MAX_ROWS), w_nk [N, K] FP32."""
    M, K = x.shape
    N = w_nk.shape[0]
    assert M <= MAX_ROWS and w_nk.dtype == torch.float32 and w_nk.shape[1] == K
    assert K % _SPLIT_K == 0 and x.stride(1) == 1 and w_nk.is_contiguous()
    k_per = K // _SPLIT_K
    partial = torch.empty((_SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _gate_partial_kernel[(triton.cdiv(N, _BN), _SPLIT_K)](
        x,
        x.stride(0),
        w_nk,
        partial,
        M,
        N,
        K,
        K_PER=k_per,
        BM=max(triton.next_power_of_2(M), 1),
        BN=_BN,
    )
    _gate_reduce_kernel[(triton.cdiv(M * N, 256),)](
        partial, out, M * N, S=_SPLIT_K, BLOCK=256
    )
    return out
