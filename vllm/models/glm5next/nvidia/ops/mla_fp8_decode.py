# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA projections on the checkpoint's own FP8 weights.

The GLM-5.3 checkpoint stores MLA q_a / kv_a / q_b / o_proj as block-FP8
(128 x 128 scales); the model dequantizes them to BF16 on load, so decode
reads twice the bytes it needs. With GLM53_MLA_FP8_DECODE=1 the loader keeps
the block scales, and after loading every MLA layer's fused_qkv_a_proj,
q_b_proj and o_proj are replaced by weight-only FP8 (Marlin W8A16) weights
rebuilt from the loaded BF16 shard and the kept scales: BF16 holds
fp8 * scale with a relative error far below half an FP8 step, so dividing by
the scale recovers the checkpoint's FP8 values exactly. That is checked bit
for bit (the BF16 shard must come back from the FP8 copy unchanged) and a
mismatch is an error. All batch sizes then run on the FP8 weights and the
BF16 weights are freed (about 240 MB per GPU under TP8, which goes to the KV
cache): on one L20, Marlin at prefill-sized batches costs about as much as
cuBLAS BF16 for these shapes, while decode reads half the bytes.
"""

import re

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)

logger = init_logger(__name__)

_BLOCK = 128
_SCALES: dict[tuple[int, str], torch.Tensor] = {}
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_ATTN_RE = re.compile(r"layers\.(\d+)\.(?:mtp_block\.)?self_attn$")


def stash_scale(layer_prefix: str, key: str, scale_inv: torch.Tensor) -> None:
    """Keep the full (unsharded) block scale of one MLA projection."""
    m = _LAYER_RE.search(layer_prefix + ".")
    assert m is not None, layer_prefix
    _SCALES[(int(m.group(1)), key)] = scale_inv.detach().to(torch.float32).cpu()


class _Fp8BlockMarlin(nn.Module):
    def __init__(self, w_bf16: torch.Tensor, scale_blocks: torch.Tensor, name: str):
        super().__init__()
        n, k = w_bf16.shape
        s = scale_blocks.to(device=w_bf16.device, dtype=torch.float32)
        assert s.shape == (-(-n // _BLOCK), -(-k // _BLOCK)), (name, s.shape, w_bf16.shape)
        s_full = s.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)[:n, :k]
        q = (w_bf16.float() / s_full).to(torch.float8_e4m3fn)
        back = (q.float() * s_full).to(torch.bfloat16)
        if not torch.equal(back, w_bf16):
            bad = int((back != w_bf16).sum())
            raise RuntimeError(
                f"GLM53_MLA_FP8_DECODE: {name}: {bad} of {w_bf16.numel()} weights do "
                "not round-trip through the checkpoint FP8 scale"
            )
        del back, s_full
        self.weight = nn.Parameter(q, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(s, requires_grad=False)
        self.weight_block_size = [_BLOCK, _BLOCK]
        self.output_size_per_partition = n
        self.input_size_per_partition = k
        self.orig_dtype = torch.bfloat16
        prepare_fp8_layer_for_marlin(self, size_k_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=self.weight,
            weight_scale=self.weight_scale_inv,
            workspace=self.workspace,
            size_n=self.output_size_per_partition,
            size_k=self.input_size_per_partition,
            bias=None,
        )


class _Fp8Method(QuantizeMethodBase):
    """Quant method that runs the layer on its FP8 Marlin weight. Weight
    creation and post-load processing still go to the original method
    (unquantized: nothing to do after loading on CUDA)."""

    def __init__(self, orig, fp8: _Fp8BlockMarlin):
        self.orig = orig
        self.fp8 = fp8

    def create_weights(self, *args, **kwargs):
        return self.orig.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.orig.process_weights_after_loading(layer)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        assert bias is None, "GLM53_MLA_FP8_DECODE: MLA projections have no bias"
        return self.fp8(x)

    def __getattr__(self, name):
        return getattr(self.orig, name)


def _wrap(linear: nn.Module, scale_blocks: torch.Tensor, name: str) -> None:
    w = linear.weight
    fp8 = _Fp8BlockMarlin(w.data, scale_blocks, name)
    # Plain attribute on the wrapper, not a registered submodule of the model.
    linear.quant_method = _Fp8Method(linear.quant_method, fp8)
    # Nothing reads the BF16 weight any more: free it.
    w.data = torch.empty(0, dtype=w.dtype, device=w.device)


def install(model: nn.Module) -> int:
    """Move every loaded MLA layer of `model` to FP8 projections."""
    rank = get_tensor_model_parallel_rank()
    count = 0
    for name, mod in model.named_modules():
        m = _ATTN_RE.search(name)
        if m is None or getattr(mod, "fused_qkv_a_proj", None) is None:
            continue
        idx = int(m.group(1))
        keys = ("q_a", "kv_a", "q_b", "o_proj")
        if not all((idx, k) in _SCALES for k in keys):
            raise RuntimeError(f"GLM53_MLA_FP8_DECODE: missing FP8 scales for layer {idx}")
        qa, kva, qb, o = (_SCALES.pop((idx, k)) for k in keys)
        # fused_qkv_a_proj is replicated: q_a rows then kv_a rows.
        _wrap(mod.fused_qkv_a_proj, torch.cat([qa, kva], 0), f"{name}.fused_qkv_a_proj")
        # q_b_proj is column parallel: this rank's output rows.
        nb = mod.q_b_proj.output_size_per_partition // _BLOCK
        _wrap(mod.q_b_proj, qb[rank * nb : (rank + 1) * nb], f"{name}.q_b_proj")
        # o_proj is row parallel: this rank's input columns.
        kb = mod.o_proj.input_size_per_partition // _BLOCK
        _wrap(mod.o_proj, o[:, rank * kb : (rank + 1) * kb], f"{name}.o_proj")
        count += 1
    logger.info("GLM53_MLA_FP8_DECODE: FP8 projections for %d MLA layers", count)
    return count
