# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-tiled FP32 mHC pre-norm GEMM for prefill-sized batches.

mhc_pre needs, per token, ``residual[4*H] @ fn[24, 4*H]^T`` and the squared
sum of the residual, both in FP32. The TileLang kernels give every CTA one or
two tokens, so each token re-reads the whole 24 x 16384 FP32 ``fn`` (1.5 MB)
from L2; on L20 that makes the op L2-bound, 5-10x slower than reading the
residual once from DRAM. Here each program takes BM tokens and one K range,
so ``fn`` is reused across BM tokens; K is split into N_SPLITS ranges for
parallelism and the partial sums are left in ``out[split]``, which the
existing big-fuse kernels already add up (the DeepGEMM path uses the same
layout). The products run on the BF16 tensor cores without losing FP32
accuracy: ``fn`` is split in-kernel into three BF16 parts that together hold
all 24 mantissa bits, the BF16 residual times each part is exact, and the
sums are FP32. On one L20 this takes 44 us for 864 tokens (TileLang 355 us,
DRAM read bound 34 us) with a max error of 5e-6 against FP64.
"""

import torch

from vllm.triton_utils import tl, triton

MIN_TOKENS = 128
N_SPLITS = 16
_BM = 64
_BN = 32
_BK = 64


@triton.jit
def _prenorm_tiled_kernel(
    x_ptr,
    x_stride,
    fn_ptr,
    out_ptr,
    sq_ptr,
    M,
    K,
    K_PER: tl.constexpr,
    N_OUT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    sq = tl.zeros((BM,), dtype=tl.float32)
    for kk in range(0, K_PER, BK):
        rk = pid_s * K_PER + kk + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * x_stride + rk[None, :],
            mask=rm[:, None] < M,
            other=0.0,
        )
        f = tl.load(fn_ptr + rn[:, None] * K + rk[None, :], mask=rn[:, None] < N_OUT, other=0.0)
        f1 = f.to(tl.bfloat16)
        r1 = f - f1.to(tl.float32)
        f2 = r1.to(tl.bfloat16)
        f3 = (r1 - f2.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(f1), acc)
        acc = tl.dot(x, tl.trans(f2), acc)
        acc = tl.dot(x, tl.trans(f3), acc)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
    tl.store(
        out_ptr + pid_s * M * N_OUT + rm[:, None] * N_OUT + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N_OUT),
    )
    tl.store(sq_ptr + pid_s * M + rm, sq, mask=rm < M)


def hc_prenorm_gemm_tiled(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    """x [M, K] BF16, fn [N_OUT, K] FP32; writes out [N_SPLITS, M, N_OUT] and
    sqrsum [N_SPLITS, M] (FP32 partial sums over N_SPLITS K ranges)."""
    M, K = x.shape
    n_out = fn.shape[0]
    assert out.shape == (N_SPLITS, M, n_out) and sqrsum.shape == (N_SPLITS, M)
    assert n_out <= _BN and K % (N_SPLITS * _BK) == 0
    assert x.stride(1) == 1 and fn.is_contiguous() and fn.dtype == torch.float32
    _prenorm_tiled_kernel[(triton.cdiv(M, _BM), N_SPLITS)](
        x,
        x.stride(0),
        fn,
        out,
        sqrsum,
        M,
        K,
        K_PER=K // N_SPLITS,
        N_OUT=n_out,
        BM=_BM,
        BN=_BN,
        BK=_BK,
        num_warps=4,
        num_stages=3,
    )
