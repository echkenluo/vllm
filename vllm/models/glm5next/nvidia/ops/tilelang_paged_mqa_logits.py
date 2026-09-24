# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang FP8 paged MQA logits for the GLM-5.3 kpool indexer decode path.

Opt-in replacement (``GLM53_TILELANG_INDEXER=1``) for the SM89 fallback that
``vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits`` dispatches to
(``sm12x_mqa.fp8_paged_mqa_logits_rowwise_triton``), which widens FP8 to FP32
and uses TF32 dot products. This kernel feeds the FP8 values straight into
FP8 tensor-core MMAs (``mma.sync`` e4m3, FP32 accumulate).

Ported from SGLang's ``fp8_paged_mqa_logits_kernel``
(python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py), with these
changes:

- page size is a parameter (SGLang asserts 64);
- K values and K scales are read through two typed views of the same page
  bytes instead of ``T.view`` on a shared byte buffer, which lets TileLang
  software-pipeline the page loads (cp.async double buffering);
- the MMA is partitioned with all warps along the page dimension, so the sum
  over heads stays inside a warp (no shared-memory cross-warp reduction);
- one block-table row may serve ``next_n`` Q rows (native MTP layout);
- positions at or past each row's context length are not written.

Cache layout (as written by ``kpool_ops.kpool_compress_and_write_cache``):
each page holds ``page_size`` pooled entries as ``page_size * head_dim`` FP8
bytes followed by ``page_size`` FP32 scales, i.e. one page is
``page_size * (head_dim + 4)`` bytes. ``block_table[r, i]`` is the physical
page of logical page ``i``; logical entry ``n`` lives at offset
``n % page_size`` of logical page ``n // page_size``.

Output contract: ``logits[r, n] = sum_h relu(q[r, h] . k[n]) * k_scale[n] *
weights[r, h]`` for ``n < context_lens[r]``. Entries at or past
``context_lens[r]`` are left unwritten (DeepGEMM's ``clean_logits=False``
contract); the Triton fallback writes ``-inf`` there. The consumers on this
path (``persistent_topk`` / ``top_k_per_row_decode``) only read each row up
to its length.
"""

from __future__ import annotations

import torch

from vllm.tilelang_utils import T, tilelang

# CTAs per row are chosen so that the grid has about this many CTAs per SM.
# A CTA keeps the row's Q (num_heads x head_dim FP8, 4 KB for 32 heads) and
# two pipeline stages of one K page (2 x 8.25 KB for 64-entry pages) in shared
# memory, about 21 KB, so four fit in an SM89 SM's 100 KB and the whole grid
# stays within one wave. SGLang used a fixed 256 CTAs on the same L20
# (92 SMs, about 2.8 per SM). Tune with the benchmark's --split sweep.
CTAS_PER_SM = 4
# Upper bound on CTAs per row; together with the power-of-two rounding this
# keeps the number of compiled variants at four for 1-24 rows (8/16/32/64),
# so an unseen eager batch size does not trigger a JIT compile mid-serving.
MAX_SPLIT_KV = 64
# Software pipeline depth over K pages (cp.async double buffering on SM89).
NUM_STAGES = 2
NUM_THREADS = 128

_num_sms_cache: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    num_sms = _num_sms_cache.get(index)
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(index).multi_processor_count
        _num_sms_cache[index] = num_sms
    return num_sms


def default_split_kv(num_rows: int, table_width: int, device: torch.device) -> int:
    """CTAs per row: largest power of two with at most ``CTAS_PER_SM`` CTAs
    per SM in total, capped at ``MAX_SPLIT_KV`` and the block-table width.

    Depends only on host-static shapes, so it is fixed inside a CUDA graph.
    """
    split = max(1, (CTAS_PER_SM * _num_sms(device)) // max(1, num_rows))
    split = 1 << (split.bit_length() - 1)
    return max(1, min(split, MAX_SPLIT_KV, max(1, table_width)))


def _pass_configs() -> dict:
    # As in SGLang's L20 build: no warp specialization / TMA (SM89 has
    # neither), and no bounds-check predicates on the paged loads. Pages are
    # read only below the row's length, and the block table is trusted, as in
    # the Triton fallback.
    return {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }


@tilelang.jit(pass_configs=_pass_configs())
def glm_fp8_paged_mqa_logits_tilelang(
    q,
    kv_values,
    kv_scales,
    weights,
    context_lens,
    block_table,
    logits,
    head_dim: int,
    num_heads: int,
    page_size: int,
    next_n: int,
    split_kv: int,
):
    num_rows = T.dynamic("num_rows")
    num_table_rows = T.dynamic("num_table_rows")
    table_width = T.dynamic("table_width")
    logits_width = T.dynamic("logits_width")
    num_pages = T.dynamic("num_pages")
    # One page viewed as FP8 rows of head_dim bytes: page_size value rows,
    # then page_size * 4 / head_dim rows holding the FP32 scales.
    page_rows = page_size * (head_dim + 4) // head_dim
    # The same page viewed as FP32 words; the scales start after the values.
    page_words = page_size * (head_dim + 4) // 4
    scale_word = page_size * head_dim // 4

    q: T.Tensor[[num_rows, num_heads, head_dim], T.float8_e4m3fn]  # type: ignore[no-redef, valid-type]
    kv_values: T.Tensor[[num_pages, page_rows, head_dim], T.float8_e4m3fn]  # type: ignore[no-redef, valid-type]
    kv_scales: T.Tensor[[num_pages, page_words], T.float32]  # type: ignore[no-redef, valid-type]
    weights: T.Tensor[[num_rows, num_heads], T.float32]  # type: ignore[no-redef, valid-type]
    context_lens: T.Tensor[[num_rows], T.int32]  # type: ignore[no-redef, valid-type]
    block_table: T.Tensor[[num_table_rows, table_width], T.int32]  # type: ignore[no-redef, valid-type]
    logits: T.Tensor[[num_rows, logits_width], T.float32]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_rows * split_kv, threads=NUM_THREADS) as bid:
        row = bid % num_rows
        split = bid // num_rows
        table_row = row // next_n
        seq_len = context_lens[row]
        row_pages = T.ceildiv(seq_len, page_size)
        pages_per_split = T.ceildiv(row_pages, split_kv)
        first_page = split * pages_per_split
        n_iters = T.max(0, T.min(pages_per_split, row_pages - first_page))

        q_smem = T.alloc_shared((num_heads, head_dim), T.float8_e4m3fn)
        w_frag = T.alloc_fragment((num_heads,), T.float32)
        k_smem = T.alloc_shared((page_size, head_dim), T.float8_e4m3fn)
        ks_smem = T.alloc_shared((page_size,), T.float32)
        ks_frag = T.alloc_fragment((page_size,), T.float32)
        scores = T.alloc_fragment((page_size, num_heads), T.float32)
        page_logits = T.alloc_fragment((page_size,), T.float32)

        T.copy(q[row, 0, 0], q_smem)
        T.copy(weights[row, 0], w_frag)

        for j in T.Pipelined(n_iters, num_stages=NUM_STAGES):
            logical_page = first_page + j
            page = block_table[table_row, logical_page]
            T.copy(kv_values[page, 0:page_size, 0:head_dim], k_smem)
            T.copy(kv_scales[page, scale_word : scale_word + page_size], ks_smem)
            T.copy(ks_smem, ks_frag)
            T.gemm(
                k_smem,
                q_smem,
                scores,
                transpose_A=False,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            # relu(q . k) * k_scale == relu(q . k * k_scale) since scales > 0.
            for h, p in T.Parallel(num_heads, page_size):
                scores[p, h] = T.max(scores[p, h], 0.0) * w_frag[h]
            T.reduce_sum(scores, page_logits, dim=1)
            for p in T.Parallel(page_size):
                if logical_page * page_size + p < seq_len:
                    logits[row, logical_page * page_size + p] = (
                        page_logits[p] * ks_frag[p]
                    )


def tilelang_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    *,
    split_kv: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Paged FP8 MQA logits with the ``fp8_fp4_paged_mqa_logits`` FP8 contract.

    Args:
        q: ``[B, next_n, H, D]`` float8_e4m3fn.
        kv_cache: ``[num_pages, page_size, 1, D + 4]`` (or 3-D without the
            head axis) uint8 page view of the indexer cache.
        weights: ``[B * next_n, H]`` float32 (Q scale already folded in).
        context_lens: ``[B, next_n]`` or ``[B, 1]`` int32 valid entries per row.
        block_table: ``[B, max_pages]`` or ``[B * next_n, max_pages]`` int32
            physical page per logical page.
        max_model_len: logits width.
        split_kv: CTAs per row; ``None`` picks :func:`default_split_kv`.
        out: optional preallocated ``[B * next_n, max_model_len]`` float32.

    Returns:
        ``[B * next_n, max_model_len]`` float32 logits, or ``None`` when the
        inputs do not match the layout this kernel handles (the caller then
        keeps the existing fallback). Every check reads host-side metadata
        only, so the call is CUDA-graph safe.
    """
    if (
        q.dim() != 4
        or q.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or context_lens.dtype != torch.int32
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
    ):
        return None
    batch_size, next_n, num_heads, head_dim = q.shape
    num_rows = batch_size * next_n
    if kv_cache.dim() == 4:
        if kv_cache.shape[2] != 1:
            return None
        kv_cache = kv_cache.squeeze(2)
    if kv_cache.dim() != 3:
        return None
    num_pages, page_size, entry_bytes = kv_cache.shape
    page_bytes = page_size * entry_bytes
    if (
        head_dim != 128
        or num_heads % 16 != 0
        or page_size % 32 != 0
        or entry_bytes != head_dim + 4
        or kv_cache.stride(0) != page_bytes
        or kv_cache.stride(1) != entry_bytes
        or kv_cache.stride(2) != 1
        # cp.async moves 16-byte K chunks and 4-byte scales.
        or kv_cache.data_ptr() % 16 != 0
        or kv_cache.storage_offset() % 4 != 0
        or weights.shape != (num_rows, num_heads)
        or block_table.shape[0] not in (batch_size, num_rows)
    ):
        return None
    if context_lens.numel() == batch_size and next_n != 1:
        context_lens = context_lens.reshape(batch_size, 1).expand(batch_size, next_n)
    if context_lens.numel() != num_rows:
        return None
    # A block table with one row per Q row (the flattened MTP layout) is
    # addressed per row; one row per request is shared by its next_n rows.
    table_next_n = 1 if block_table.shape[0] == num_rows else next_n
    # .contiguous() is a no-op for the layouts vLLM builds; it only copies a
    # strided slice a caller hands in (still graph-safe).
    q_rows = q.reshape(num_rows, num_heads, head_dim).contiguous()
    if q_rows.data_ptr() % 16 != 0:
        return None

    if out is None:
        logits = torch.empty(
            (num_rows, max_model_len), dtype=torch.float32, device=q.device
        )
    else:
        assert out.shape == (num_rows, max_model_len)
        assert out.dtype == torch.float32 and out.is_contiguous()
        logits = out
    if num_rows == 0:
        return logits
    if split_kv is None:
        split_kv = default_split_kv(num_rows, block_table.shape[1], q.device)

    # Zero-copy views of the page bytes: FP8 rows for K, FP32 words for scales.
    kv_bytes = kv_cache.view(num_pages, page_bytes)
    kv_values = kv_bytes.view(torch.float8_e4m3fn).view(
        num_pages, page_bytes // head_dim, head_dim
    )
    kv_scales = kv_bytes.view(torch.float32)
    glm_fp8_paged_mqa_logits_tilelang(
        q_rows,
        kv_values,
        kv_scales,
        weights.contiguous(),
        context_lens.reshape(num_rows).contiguous(),
        block_table.contiguous(),
        logits,
        head_dim,
        num_heads,
        page_size,
        table_next_n,
        split_kv,
    )
    return logits
