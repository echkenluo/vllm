# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill sparse MLA for GLM-5.3 on SM89: TileLang, reading the FP8 NoPE cache.

The SM89 path runs every query token through FlashInfer's sparse decode kernel
(one query token per "request", its own top-k list), about 2.1 us per token
per layer on one L20 under TP8 (8 heads per rank). Upstream SGLang's TileLang
sparse attention v1 kernel (python/sglang/kernels/ops/attention/dsa/
tilelang_kernel.py at 172b1b4825) does the same work in about 1.5 us per token
on a BF16 cache. This is that kernel with two changes:

* the KV rows are read from the GLM NoPE cache that the FlashInfer fork writes
  (528 bytes per token: 512 FP8 e4m3 values, then 4 FP32 scales, one per 128
  values) and dequantized to BF16 while loading, so there is no BF16 copy of
  the cache and each key reads 528 bytes instead of 1024;
* keys are masked by the per-token valid count (the converted top-k list is a
  packed prefix), padding indices are clamped to row 0 before the load, and
  the loop stops after the last block that holds a valid key (the FlashInfer
  kernel's cost also scales with the valid count; prefill rows before position
  2048 have fewer than 2048 keys).

One pipeline stage: two stages need 160 KB of shared memory, SM89 has ~100 KB.
Rows whose valid count is 0 come out as 0, as in the FlashInfer path.
"""

import functools

import tilelang
import tilelang.language as T
import torch

D_NOPE = 512
ROW_BYTES = 528
MIN_TOKENS = 128


@functools.cache
def _kernel(num_heads: int, topk: int, sm_scale: float, block_i: int = 64, threads: int = 256):
    sm_scale_log2 = sm_scale * 1.44269504  # exp2 instead of exp
    assert topk % block_i == 0
    padded_h = max(tilelang.math.next_power_of_2(num_heads), 16)
    seq_len = T.symbolic("seq_len")
    num_rows = T.symbolic("num_rows")
    bf16, fp8, f32 = "bfloat16", "float8_e4m3fn", "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build():
        @T.prim_func
        def main(
            Q: T.Tensor([seq_len, num_heads, D_NOPE], bf16),  # type: ignore
            KVq: T.Tensor([num_rows, ROW_BYTES], fp8),  # type: ignore
            KVs: T.Tensor([num_rows, ROW_BYTES // 4], f32),  # type: ignore
            Indices: T.Tensor([seq_len, topk], "int32"),  # type: ignore
            Lens: T.Tensor([seq_len], "int32"),  # type: ignore
            Output: T.Tensor([seq_len, num_heads, D_NOPE], bf16),  # type: ignore
        ):
            with T.Kernel(seq_len, threads=threads) as bx:
                Q_shared = T.alloc_shared([padded_h, D_NOPE], bf16)
                KV_shared = T.alloc_shared([block_i, D_NOPE], bf16)
                mask = T.alloc_fragment([block_i], "bool")
                acc_o = T.alloc_fragment([padded_h, D_NOPE], f32)
                acc_s = T.alloc_fragment([padded_h, block_i], f32)
                S_shared = T.alloc_shared([padded_h, block_i], bf16)
                sumexp = T.alloc_fragment([padded_h], f32)
                sumexp_i = T.alloc_fragment([padded_h], f32)
                alpha = T.alloc_fragment([padded_h], f32)
                m_i = T.alloc_fragment([padded_h], f32)
                m_i_prev = T.alloc_fragment([padded_h], f32)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf = nan
                valid = Lens[bx]
                T.copy(Q[bx, 0:padded_h, :], Q_shared)

                for i_i in T.Pipelined(T.ceildiv(valid, block_i), num_stages=1):
                    for bi_i in T.Parallel(block_i):
                        mask[bi_i] = i_i * block_i + bi_i < valid
                    for bi_i, d_i in T.Parallel(block_i, D_NOPE):
                        row = T.max(Indices[bx, i_i * block_i + bi_i], 0)
                        KV_shared[bi_i, d_i] = (
                            KVq[row, d_i].astype(f32)
                            * KVs[row, D_NOPE // 4 + d_i // 128]
                        ).astype(bf16)
                    for h_i, bi_i in T.Parallel(padded_h, block_i):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(
                        Q_shared, KV_shared, acc_s, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(padded_h):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2)
                    for h_i, bi_i in T.Parallel(padded_h, block_i):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale_log2 - m_i[h_i] * sm_scale_log2
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(padded_h):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(padded_h, D_NOPE):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]
                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                for h_i, d_i in T.Parallel(padded_h, D_NOPE):
                    acc_o[h_i, d_i] = T.if_then_else(
                        valid > 0, acc_o[h_i, d_i] / sumexp[h_i], 0
                    )
                T.copy(acc_o, Output[bx, 0:padded_h, :])

        return main

    return build()


def sparse_prefill_fp8_nope(
    q: torch.Tensor,
    kv_rows: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """q [T, H, 512] BF16; kv_rows [R, 528] uint8 or FP8 view of the paged cache
    (flat physical rows); indices [T, topk] int32 flat row ids (packed valid
    prefix, topk % 64 == 0); lens [T] int32 valid counts. Returns [T, H, 512]."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == D_NOPE
    assert kv_rows.shape[-1] == ROW_BYTES and kv_rows.stride(-1) == 1
    rows_u8 = kv_rows.view(torch.uint8)
    kvq = rows_u8.view(torch.float8_e4m3fn)
    kvs = rows_u8.view(torch.float32)
    kern = _kernel(q.shape[1], indices.shape[1], float(sm_scale))
    return kern(q.contiguous(), kvq, kvs, indices.contiguous(), lens.to(torch.int32).contiguous())
