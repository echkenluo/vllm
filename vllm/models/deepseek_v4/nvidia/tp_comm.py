# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in token-scattered TP for eager DSv4 prefill, retaining TP weights.

FP8 wire format follows SGLang's DSv4 port: E4M3 payload with one FP32 scale
per 128 values. Auxiliary draft states and the final mHC state stay BF16.
"""

import os
from collections import Counter

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.models.common.ops.sequence_parallel import sp_shard
from vllm.triton_utils import tl, triton

MODES = ("off", "preserve_ar", "bf16", "fp8_ag", "fp8_rs", "fp8_both")
GROUP = 128


def enabled() -> bool:
    return os.getenv("DSV4_TP_COMM", "0") == "1"


@triton.jit
def _unpack_kernel(
    q, scales, out, stride: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    col = group * GROUP + tl.arange(0, GROUP)
    value = tl.load(q + row * stride + col).to(tl.float32)
    scale = tl.load(scales + row * (H // GROUP) + group)
    tl.store(out + row * H + col, value * scale)


def _pack(x: torch.Tensor) -> torch.Tensor:
    rows, hidden = x.shape
    q, scales = per_token_group_quant_fp8(
        x.contiguous(), GROUP, dtype=torch.float8_e4m3fn, use_ue8m0=False
    )
    scale_bytes = hidden // GROUP * 4
    packed = torch.empty(
        (rows, hidden + scale_bytes), dtype=torch.uint8, device=x.device
    )
    packed[:, :hidden] = q.view(torch.uint8)
    packed[:, hidden:] = scales.view(torch.uint8).reshape(rows, scale_bytes)
    return packed


def _unpack(packed: torch.Tensor, hidden: int, rows: int, dtype) -> torch.Tensor:
    q = packed.view(torch.float8_e4m3fn)
    scales = packed[:, hidden:].contiguous().view(torch.float32)
    out = torch.empty((rows, hidden), dtype=dtype, device=packed.device)
    _unpack_kernel[(rows, hidden // GROUP)](
        q, scales, out, packed.stride(0), hidden, GROUP
    )
    return out


def _pad(x: torch.Tensor) -> torch.Tensor:
    padding = (-x.shape[0]) % get_tp_group().world_size
    if padding:
        x = torch.cat((x, x.new_zeros((padding, *x.shape[1:]))))
    return x.contiguous()


class TPComm:
    """Forward-local layout decisions; no changes to attention or MoE weights."""

    def __init__(self, config):
        self.enabled = enabled()
        self.mode = os.getenv("DSV4_TP_COMM_MODE", "bf16")
        self.min_tokens = int(os.getenv("DSV4_TP_COMM_MIN_TOKENS", "512"))
        self.embedding_rs = os.getenv("DSV4_TP_COMM_EMBED_RS", "1") == "1"
        self.stats: Counter[str] = Counter()
        if self.mode not in MODES or self.min_tokens < 1:
            raise ValueError("Invalid DSV4_TP_COMM mode or token threshold")
        if self.enabled:
            pc = config.parallel_config
            if not (
                pc.tensor_parallel_size > 1
                and pc.pipeline_parallel_size == 1
                and pc.data_parallel_size == 1
                and not pc.enable_expert_parallel
                and not pc.enable_dbo
                and config.model_config.dtype == torch.bfloat16
                and config.model_config.hf_config.hidden_size % GROUP == 0
            ):
                raise ValueError("DSV4_TP_COMM requires BF16 TP-only PP1/DP1, no DBO")

    def begin(self, rows: int, metadata_key: str) -> str | None:
        if not self.enabled:
            return None
        if self.mode == "off" or rows < self.min_tokens:
            self.stats["fallback_off_or_short"] += 1
            return None
        if not is_forward_context_available():
            self.stats["fallback_no_context"] += 1
            return None
        ctx = get_forward_context()
        if (
            ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or ctx.ubatch_slices is not None
            or not isinstance(ctx.attn_metadata, dict)
        ):
            self.stats["fallback_graph_or_dummy"] += 1
            return None
        meta = ctx.attn_metadata.get(metadata_key)
        if meta is None or getattr(meta, "num_prefills", 0) == 0:
            self.stats["fallback_no_prefill"] += 1
            return None
        self.stats["forwards"] += 1
        self.stats["rows"] += rows
        self.stats["mixed_forwards"] += int(getattr(meta, "num_decodes", 0) > 0)
        return self.mode

    def embed(self, embedding, input_ids: torch.Tensor, mode: str):
        if not self.embedding_rs or mode == "preserve_ar":
            self.stats["embedding_ar_shard"] += 1
            return sp_shard(embedding(input_ids)).contiguous()
        partial = embedding.forward_unreduced(input_ids)
        if partial.dtype != torch.bfloat16:
            raise ValueError("DSV4 TP entry reduction requires BF16 embeddings")
        self.stats["embedding_bf16_rs"] += 1
        return get_tp_group().reduce_scatter(_pad(partial), dim=0)

    def gather(self, x: torch.Tensor, rows: int, mode: str, site: str):
        group = get_tp_group()
        total = x.shape[0] * group.world_size
        if not (0 < rows <= total and total - rows < group.world_size):
            raise ValueError("TP gather row count does not match token padding")
        fp8 = site in ("attn", "moe") and mode in ("fp8_ag", "fp8_both")
        self.stats[("fp8_ag_" if fp8 else "bf16_ag_") + site] += 1
        if fp8:
            packed = _pack(x)
            full = torch.empty(
                (total, packed.shape[1]), dtype=torch.uint8, device=x.device
            )
            dist.all_gather_into_tensor(full, packed, group=group.device_group)
            return _unpack(full, x.shape[-1], rows, x.dtype)
        return group.all_gather(x.contiguous(), dim=0)[:rows]

    def reduce(self, x: torch.Tensor, mode: str, site: str):
        group = get_tp_group()
        self.stats[mode + "_rs_" + site] += 1
        if mode == "preserve_ar":
            return sp_shard(group.all_reduce(x.contiguous())).contiguous()
        x = _pad(x)
        if mode not in ("fp8_rs", "fp8_both"):
            return group.reduce_scatter(x, dim=0)
        packed = _pack(x)
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=group.device_group)
        rows, hidden = x.shape
        values = _unpack(received, hidden, rows, torch.float32)
        return (
            values.view(group.world_size, rows // group.world_size, hidden)
            .sum(dim=0)
            .to(x.dtype)
        )
