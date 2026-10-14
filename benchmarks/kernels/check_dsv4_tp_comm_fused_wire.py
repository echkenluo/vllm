# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-GPU checks for the DSv4 TP comm fused FP8 wire and the tiled mHC pre-norm GEMM.

Run inside the vLLM image with this branch mounted; no distributed init. Writes one JSON object to --output and
exits 1 when a check fails (thresholds of amendment 1 of the 2026-10-14 comm gain gap round):

  wire     _unpack_fused against _unpack (bits), and _unpack_sum against the reference tail (unpack to
           FP32, sum over ranks, cast), both compared to an FP64 sum of the dequantized values; the
           difference is measured in BF16 ulps at the scale of the summands (sum of |v_w|), because the
           eight partial sums often cancel and an ulp of the result itself is then meaningless; plus
           CUDA-event timings.
  prenorm  hc_prenorm_gemm_tiled (GLM53_MHC_PRENORM_TILED) against the TileLang kernel the DSv4 path
           uses today, at the DSv4 shape (24 x 16384) for 512 and 4096 rows, both against FP64.

Timings are the wall time per call of 50 back-to-back eager launches (CUDA events), so small kernels include
launch cost, unlike the GPU kernel time in a profiler trace.
"""

import argparse
import json

import torch

from vllm.model_executor.kernels.mhc import prenorm_tiled
from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
from vllm.models.deepseek_v4.nvidia import tp_comm as tc

WORLD, HIDDEN = 8, 4096


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) / iters * 1000, 2)  # us


def ulp_distance(a, b):
    """BF16 distance in units in the last place (sign-magnitude bits mapped to ordered integers)."""

    def ordered(t):
        bits = t.view(torch.int16).int()
        mag = bits & 0x7FFF
        return torch.where(bits < 0, -mag, mag)

    return (ordered(a) - ordered(b)).abs()


def sample(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, device="cuda", generator=g) * 3
    x[0].zero_()  # all-zero groups: absmax falls back to eps
    x[1] *= 1e-6
    x[2] *= 1e4
    return x.bfloat16()


def wire():
    out = {}
    for rows in (511, 512, 4096):
        ref = tc._pack(sample(rows, rows))
        out[f"unpack_rows{rows}_bitwise_equal"] = bool(
            torch.equal(tc._unpack(ref, HIDDEN, rows, torch.bfloat16),
                        tc._unpack_fused(ref, HIDDEN, rows, torch.bfloat16)))
    # ReduceScatter tail at 4096 rows: rank w sent its shard-th partial rows to us, row block w of received
    rows, shard = 4096, 4096 // WORLD
    received = torch.cat([tc._pack(sample(shard, 1000 + w)) for w in range(WORLD)])
    old = (tc._unpack(received, HIDDEN, rows, torch.float32).view(WORLD, shard, HIDDEN).sum(dim=0)
           .to(torch.bfloat16))
    new = tc._unpack_sum(received, HIDDEN, WORLD, torch.bfloat16)
    values = tc._unpack(received, HIDDEN, rows, torch.float32).double().view(WORLD, shard, HIDDEN)
    exact = values.sum(0)
    abs_sum = values.abs().sum(0)
    ulp_of_abs_sum = torch.exp2(torch.floor(torch.log2(abs_sum.clamp_min(1e-30))) - 7)  # BF16: 8 significant bits
    out["rs_tail_bf16_mismatches"] = int((old != new).sum())
    out["rs_tail_elements"] = old.numel()
    out["rs_tail_max_ulp_of_result"] = int(ulp_distance(old, new).max())  # blows up where the sum cancels
    out["rs_tail_max_diff_in_ulps_of_abs_sum"] = float(((old.double() - new.double()).abs() / ulp_of_abs_sum).max())
    out["rs_tail_old_max_abs_err_vs_fp64"] = float((old.double() - exact).abs().max())
    out["rs_tail_new_max_abs_err_vs_fp64"] = float((new.double() - exact).abs().max())
    out["rs_tail_old_mean_abs_err_vs_fp64"] = float((old.double() - exact).abs().mean())
    out["rs_tail_new_mean_abs_err_vs_fp64"] = float((new.double() - exact).abs().mean())
    # the same tails before the BF16 cast (recorded, not a check): the fused kernel may contract value * scale and
    # the add into one FMA, the reference rounds each product first, so they should differ by a few FP32 ulps at
    # the summands' scale and the BF16 mismatches are results that sit near a rounding boundary
    old32 = tc._unpack(received, HIDDEN, rows, torch.float32).view(WORLD, shard, HIDDEN).sum(dim=0)
    new32 = tc._unpack_sum(received, HIDDEN, WORLD, torch.float32)
    fp32_ulp_of_abs_sum = torch.exp2(torch.floor(torch.log2(abs_sum.clamp_min(1e-30))) - 23)
    d32 = (old32.double() - new32.double()).abs() / fp32_ulp_of_abs_sum
    out["rs_tail_fp32_max_diff_in_ulps_of_abs_sum"] = float(d32.max())
    out["rs_tail_fp32_mean_diff_in_ulps_of_abs_sum"] = float(d32.mean())
    out["rs_tail_fp32_mismatches"] = int((old32 != new32).sum())
    # timings, us per call, the shapes of one 4K prefill forward
    x512 = sample(512, 8)
    full = torch.cat([tc._pack(x512)] * WORLD)
    out["us_ag_unpack_4096_old"] = bench(lambda: tc._unpack(full, HIDDEN, 4096, torch.bfloat16))
    out["us_ag_unpack_4096_new"] = bench(lambda: tc._unpack_fused(full, HIDDEN, 4096, torch.bfloat16))
    out["us_rs_tail_old"] = bench(lambda: tc._unpack(received, HIDDEN, rows, torch.float32)
                                  .view(WORLD, shard, HIDDEN).sum(dim=0).to(torch.bfloat16))
    out["us_rs_tail_new"] = bench(lambda: tc._unpack_sum(received, HIDDEN, WORLD, torch.bfloat16))
    return out


def prenorm():
    out = {}
    hc_mult, n_out = 4, 24
    k = hc_mult * HIDDEN
    g = torch.Generator(device="cuda").manual_seed(5)
    fn = torch.randn(n_out, k, device="cuda", generator=g) * 0.02
    for rows in (512, 4096):
        x = torch.randn(rows, k, device="cuda", generator=g).bfloat16()
        ref_out = x.double() @ fn.double().t()
        ref_sq = x.double().square().sum(-1)
        tl_out = torch.empty(1, rows, n_out, device="cuda")
        tl_sq = torch.empty(1, rows, device="cuda")
        _tilelang_hc_prenorm_gemm(x, fn, tl_out, tl_sq, HIDDEN, hc_mult)
        ti_out = torch.empty(prenorm_tiled.N_SPLITS, rows, n_out, device="cuda")
        ti_sq = torch.empty(prenorm_tiled.N_SPLITS, rows, device="cuda")
        prenorm_tiled.hc_prenorm_gemm_tiled(x, fn, ti_out, ti_sq)

        def rel(a, b):
            return float(((a.double() - b).abs().max() / b.abs().max()))

        out[f"rows{rows}_tilelang_rel_err"] = rel(tl_out[0], ref_out)
        out[f"rows{rows}_tiled_rel_err"] = rel(ti_out.sum(0), ref_out)
        out[f"rows{rows}_tilelang_sq_rel_err"] = rel(tl_sq[0], ref_sq)
        out[f"rows{rows}_tiled_sq_rel_err"] = rel(ti_sq.sum(0), ref_sq)
        out[f"rows{rows}_us_tilelang"] = bench(lambda: _tilelang_hc_prenorm_gemm(x, fn, tl_out, tl_sq, HIDDEN, hc_mult))
        out[f"rows{rows}_us_tiled"] = bench(lambda: prenorm_tiled.hc_prenorm_gemm_tiled(x, fn, ti_out, ti_sq))
    return out


def failures(w, p):
    bad = []
    if not all(w[f"unpack_rows{r}_bitwise_equal"] for r in (511, 512, 4096)):
        bad.append("fused AllGather dequantize differs from _unpack")
    if w["rs_tail_max_diff_in_ulps_of_abs_sum"] > 1.0 or w["rs_tail_bf16_mismatches"] > 0.01 * w["rs_tail_elements"]:
        bad.append("fused ReduceScatter tail differs from the reference by more than 1 ulp at the summands' scale "
                   "or in more than 1% of elements")
    if w["rs_tail_new_mean_abs_err_vs_fp64"] > 1.1 * w["rs_tail_old_mean_abs_err_vs_fp64"]:
        bad.append("fused ReduceScatter tail is less accurate than the reference")
    for r in (512, 4096):
        if p[f"rows{r}_tiled_rel_err"] > 1e-5 or p[f"rows{r}_tiled_sq_rel_err"] > 1e-5:
            bad.append(f"tiled pre-norm GEMM relative error above 1e-5 at {r} rows")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    w, p = wire(), prenorm()
    bad = failures(w, p)
    with open(a.output, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "wire": w, "prenorm": p, "failures": bad}, f, indent=1)
    print("CHECK_FAILED " + "; ".join(bad) if bad else "CHECK_PASS", flush=True)
    raise SystemExit(1 if bad else 0)
