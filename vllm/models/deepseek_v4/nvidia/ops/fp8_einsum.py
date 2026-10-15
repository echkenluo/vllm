# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 Triton FP8 einsum kernels for DeepSeek V4."""

import os

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.deep_gemm import fp8_einsum

# DSV4_WO_A_SPLITK=1 (off by default): split the hidden (K) loop across CTAs when
# the token x output tiles alone cannot fill the GPU. wo_a in decode has 1024
# output columns per rank, so at most 16 rows there are only 8 output tiles and
# each CTA walks all 32 K blocks of 128 one after another; the call then costs
# about 36 us on L20 for a 4 MB weight that streams in 5 us. With the split, the
# partial sums go to an FP32 buffer and a second kernel adds the splits in a
# fixed order, so the result does not depend on scheduling.
_SPLITK = os.getenv("DSV4_WO_A_SPLITK", "0") == "1"
_SPLITK_TARGET_CTAS = int(os.getenv("DSV4_WO_A_SPLITK_CTAS", "184"))


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "a_stride_group",
        "a_scale_stride_group",
        "a_scale_stride_hidden",
    ]
)
def _deepseek_v4_sm89_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    partial_ptr,
    num_tokens,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group,
    a_scale_stride_hidden,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    SPLIT: tl.constexpr,
) -> None:
    a_stride_group = tl.multiple_of(a_stride_group, 16)
    a_scale_stride_group = tl.multiple_of(a_scale_stride_group, 16)
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2) // SPLIT
    split = tl.program_id(2) % SPLIT
    hidden_per_split = hidden_size // SPLIT
    hidden_begin = split * hidden_per_split

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_step in range(0, hidden_per_split, BLOCK_HIDDEN):
        hidden_start = hidden_begin + hidden_step
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        ).to(tl.bfloat16)
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank) & (hidden[:, None] < hidden_size),
            other=0.0,
        ).to(tl.bfloat16)
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    mask = (token_offsets[:, None] < num_tokens) & (out_offsets[None, :] < out_rank)
    if SPLIT == 1:
        tl.store(
            out_ptr
            + token_offsets[:, None] * out_stride_token
            + group * out_stride_group
            + out_offsets[None, :] * out_stride_rank,
            accum,
            mask=mask,
        )
    else:
        # partial is [SPLIT, num_tokens, num_groups, out_rank], contiguous
        tl.store(
            partial_ptr
            + split * num_tokens * num_groups * out_rank
            + token_offsets[:, None] * num_groups * out_rank
            + group * out_rank
            + out_offsets[None, :],
            accum,
            mask=mask,
        )


@triton.jit(do_not_specialize=["numel"])
def _deepseek_v4_sm89_fp8_einsum_reduce_kernel(
    partial_ptr,
    out_ptr,
    numel,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(partial_ptr + s * numel + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, acc, mask=mask)


def _choose_split(num_tiles: int, k_blocks: int) -> int:
    """Largest power of two that divides the K blocks and keeps the grid within
    the target CTA count; 1 when the tiles already fill the GPU."""
    split = 1
    while (
        split * 2 <= k_blocks
        and k_blocks % (split * 2) == 0
        and num_tiles * split * 2 <= _SPLITK_TARGET_CTAS
    ):
        split *= 2
    return split


def _deepseek_v4_sm89_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    split_override: int | None = None,
) -> None:
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if a_scale.dtype == e8m0_dtype:
        a_scale = _upcast_e8m0_to_fp32(a_scale)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    num_tiles = (
        triton.cdiv(num_tokens, block_tokens)
        * triton.cdiv(out_rank, block_out)
        * num_groups
    )
    split = (
        _choose_split(num_tiles, hidden_size // block_hidden)
        if _SPLITK and out.is_contiguous()
        else 1
    )
    if split_override is not None:
        split = split_override
    # the partial buffer and the reduce kernel index out as [tokens, groups, rank]
    assert split == 1 or out.is_contiguous()
    partial = (
        torch.empty(
            (split, num_tokens, num_groups, out_rank),
            device=out.device,
            dtype=torch.float32,
        )
        if split > 1
        else out
    )
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups * split,
    )
    _deepseek_v4_sm89_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        partial,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        SPLIT=split,
        num_warps=4,
        num_stages=3,
    )
    if split > 1:
        numel = num_tokens * num_groups * out_rank
        reduce_block = 1024
        _deepseek_v4_sm89_fp8_einsum_reduce_kernel[
            (triton.cdiv(numel, reduce_block),)
        ](partial, out, numel, SPLIT=split, BLOCK=reduce_block, num_warps=4)


def _use_deepseek_v4_sm89_triton_fp8_einsum(
    equation: str,
    recipe: tuple[int, int, int],
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return (
        HAS_TRITON
        and capability == DeviceCapability(8, 9)
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and b_scale.dtype in (torch.float32, e8m0_dtype)
    )


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: tuple[int, int, int],
) -> None:
    if _use_deepseek_v4_sm89_triton_fp8_einsum(equation, recipe, b_scale):
        if b.dim() == 2:
            num_groups = out.shape[1]
            out_rank = out.shape[2]
            hidden_size = a.shape[2]
            if b.shape[0] % out_rank != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                    f"out_rank={out_rank}, got {b.shape[0]}"
                )
            b_groups = b.shape[0] // out_rank
            group_start = 0
            if b_groups != num_groups:
                if b_groups % num_groups != 0:
                    raise RuntimeError(
                        "DeepSeek V4 fp8 einsum weight groups must match the "
                        "TP-local output groups or be an integer multiple of "
                        f"them, got weight_groups={b_groups}, "
                        f"output_groups={num_groups}"
                    )
                group_partitions = b_groups // num_groups
                group_start = (
                    get_tensor_model_parallel_rank() % group_partitions
                ) * num_groups
            b = b.view(b_groups, out_rank, hidden_size)
            if group_start != 0 or b_groups != num_groups:
                b = b.narrow(0, group_start, num_groups)

            if b_scale.dim() == 2:
                scale_mn = recipe[1]
                scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
                scale_k = recipe[2] * scale_k_pack
                scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
                scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
                if b_scale.shape[0] % scale_out_blocks != 0:
                    raise RuntimeError(
                        "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                        f"scale_out_blocks={scale_out_blocks}, "
                        f"got {b_scale.shape[0]}"
                    )
                scale_groups = b_scale.shape[0] // scale_out_blocks
                if scale_groups not in (num_groups, b_groups):
                    raise RuntimeError(
                        "DeepSeek V4 fp8 einsum scale groups must match the "
                        "TP-local output groups or weight groups, got "
                        f"scale_groups={scale_groups}, output_groups={num_groups}, "
                        f"weight_groups={b_groups}"
                    )
                b_scale = b_scale.view(
                    scale_groups,
                    scale_out_blocks,
                    scale_hidden_blocks,
                )
                if scale_groups == b_groups and scale_groups != num_groups:
                    b_scale = b_scale.narrow(0, group_start, num_groups)
            elif b_scale.dim() == 3 and b_scale.shape[0] == b_groups:
                if b_groups != num_groups:
                    b_scale = b_scale.narrow(0, group_start, num_groups)

        _deepseek_v4_sm89_fp8_einsum(a, a_scale, b, b_scale, out)
        return

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
