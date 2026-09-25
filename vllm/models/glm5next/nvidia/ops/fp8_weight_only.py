# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 (Marlin W8A16) copies of BF16 weights for decode-sized GEMMs.

Decode on L20 is bound by reading weights: a [1..24, K] x [K, N] GEMM costs
about bytes / 830 GB/s. Marlin reads an FP8 copy and dequantizes in-kernel,
activations stay BF16, so the same GEMM reads half the bytes (measured on one
L20: the 4096 x 19360 lm_head shard 190 -> 98 us, the 8192 x 4096 MTP eh_proj
82 -> 44 us). Used for the MTP draft only, where the target model's
verification keeps the output unchanged and the precision loss can only
change the acceptance rate.
"""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class Fp8MarlinWeight(nn.Module):
    """Marlin-packed FP8 copy of a BF16 [N, K] weight, per-output-channel scale."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        n, k = weight.shape
        w = weight.detach().float()
        scale = (w.abs().amax(dim=1).clamp(min=1e-12) / _FP8_MAX).float()
        q = (w / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        del w
        self.weight = nn.Parameter(q, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.output_size_per_partition = n
        self.input_size_per_partition = k
        self.orig_dtype = weight.dtype
        self.logical_widths = [n]
        prepare_fp8_layer_for_marlin(self, size_k_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=self.weight,
            weight_scale=self.weight_scale,
            workspace=self.workspace,
            size_n=self.output_size_per_partition,
            size_k=self.input_size_per_partition,
            bias=None,
        )


class Fp8LMHeadProxy:
    """Stands in for a ParallelLMHead in LogitsProcessor calls: same vocab
    shard metadata, but the projection runs on an FP8 Marlin copy."""

    def __init__(self, head: nn.Module):
        self.tp_size = head.tp_size
        self.shard_indices = head.shard_indices
        self.quant_method = self
        self.fp8 = Fp8MarlinWeight(head.weight.data)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        assert bias is None, "FP8 draft lm_head has no bias support"
        return self.fp8(x)
