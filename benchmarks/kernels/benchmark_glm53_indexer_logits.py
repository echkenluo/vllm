# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 kpool indexer decode logits on SM89: Triton fallback vs TileLang.

Builds the tensors ``sparse_attn_indexer_kpool`` hands to the paged FP8 MQA
logits call during MTP decode on SM89 (flattened layout: one Q row per decode
token, per-row block table and pool-granular context length) and times

  (a) ``vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits`` as the indexer calls
      it; on SM89 this dispatches to ``sm12x_mqa.fp8_paged_mqa_logits_triton``;
  (b) ``tilelang_fp8_paged_mqa_logits`` (``GLM53_TILELANG_INDEXER=1`` path).

The index-K cache is written by the production writer
(``kpool_compress_and_write_cache``: softmax pooling of 4 tokens, Hadamard,
FP8 with power-of-two scales) into randomly permuted physical pages, and Q
comes from the production ``fwht128_quant_fp8`` with its scale folded into
the head weights, so both kernels read exactly the bytes vLLM stores. One
JSON line per case. Times are medians of CUDA-event-timed CUDA-graph replays
(the decode path runs inside FULL graphs), with L2 flushed before each replay
unless --no-flush-l2.

--check runs correctness only and exits 1 on any failure. Tolerances:

* Logits: FP8 e4m3 products are exact in FP32 (and in the fallback's TF32,
  since e4m3 has 3 mantissa bits), so the kernels differ only in FP32
  summation order over 128 dims and the heads, plus the FP8 MMA's internal
  accumulation. SGLang measured a 2e-4 relative difference between the same
  two kernel families on L20. The check allows max |diff| <= 5e-3 of the
  row's max |logit| (25x that observation); any layout error (wrong page,
  entry, scale or head) is O(1) on this scale.
* Top-k (512 pools = index_topk 2048 / kpool 4): noise of that size can only
  swap near-ties at the selection boundary (SGLang saw 99.98% overlap). The
  check requires >= 98% overlap per row (about 10 swaps of 512), both for
  torch.topk over the valid prefix and for vLLM's persistent_topk run on the
  full rows. The TileLang output is pre-filled with +inf past each row's
  length, so a top-k that read past the length would fail the check.
* Also checked: all valid logits finite, persistent_topk indices inside the
  row, and a CUDA graph captured with one set of context lengths replayed
  with another (lengths and block tables are read on device only).

Usage inside the image (one GPU, no model weights needed):
  python3 benchmark_glm53_indexer_logits.py [--split auto,8,16,32,64]
  python3 benchmark_glm53_indexer_logits.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

KPOOL = 4
HEAD_DIM = 128
INDEX_TOPK = 2048
MTP_ROWS_PER_REQUEST = 6  # 1 + num_speculative_tokens (5)
MLA_LAYERS_PER_TARGET_FORWARD = 11


def _imports():
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        fwht128_quant_fp8,
        kpool_compress_and_write_cache,
    )
    from vllm.models.glm5next.nvidia.ops.tilelang_paged_mqa_logits import (
        default_split_kv,
        tilelang_fp8_paged_mqa_logits,
    )
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

    return (
        fwht128_quant_fp8,
        kpool_compress_and_write_cache,
        default_split_kv,
        tilelang_fp8_paged_mqa_logits,
        fp8_fp4_paged_mqa_logits,
    )


(
    fwht128_quant_fp8,
    kpool_compress_and_write_cache,
    default_split_kv,
    tilelang_fp8_paged_mqa_logits,
    fp8_fp4_paged_mqa_logits,
) = _imports()


def build_case(
    req_contexts: list[int],
    heads: int,
    page_size: int,
    max_model_len: int,
    next_n: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Decode inputs for len(req_contexts) requests of next_n rows each.

    ``req_contexts`` are token counts including the next_n tokens being
    verified; row j of a request sits at position ctx - next_n + j, so its
    pool-granular length is (ctx - next_n + j + 1) // KPOOL (complete pools
    only; the in-progress pool comes from the tail cache, not this kernel).
    A context of 0 models a CUDA-graph padding request (length 0).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    nreq = len(req_contexts)
    rows = nreq * next_n
    tokens_per_page = page_size * KPOOL
    table_width = max_model_len // tokens_per_page
    req_pools = [c // KPOOL for c in req_contexts]
    req_pages = [-(-p // page_size) for p in req_pools]
    num_pages = sum(req_pages) + 1
    perm = torch.randperm(num_pages, generator=gen, device=device).to(torch.int32)

    kv3 = torch.zeros(
        (num_pages, page_size, HEAD_DIM + 4), dtype=torch.uint8, device=device
    )
    req_tables = torch.zeros((nreq, table_width), dtype=torch.int32, device=device)
    ape = torch.randn(KPOOL, HEAD_DIM, generator=gen, device=device) * 0.1
    cursor = 0
    for r, (pools, pages) in enumerate(zip(req_pools, req_pages)):
        req_tables[r, :pages] = perm[cursor : cursor + pages]
        cursor += pages
        # Write the request's pools with the production writer, in chunks.
        for start in range(0, pools, 65536):
            n = min(65536, pools - start)
            pool_ids = torch.arange(start, start + n, device=device)
            loc = (
                req_tables[r, pool_ids // page_size].to(torch.int64) * page_size
                + pool_ids % page_size
            )
            slot_k = torch.randn(
                n, KPOOL, HEAD_DIM, generator=gen, device=device
            ).to(torch.bfloat16)
            slot_score = torch.randn(
                n, KPOOL, HEAD_DIM, generator=gen, device=device
            ).to(torch.bfloat16)
            kpool_compress_and_write_cache(
                kv3,
                slot_k,
                slot_score,
                ape,
                loc,
                pool_size=KPOOL,
                head_dim=HEAD_DIM,
                round_scale=True,
            )

    # Flattened layout (SM89 has no native next_n > 2 path): one block-table
    # row and one length per decode token.
    block_table = req_tables.repeat_interleave(next_n, dim=0).contiguous()
    offs = torch.arange(next_n, device=device, dtype=torch.int64)
    ctx = torch.tensor(req_contexts, device=device, dtype=torch.int64)
    tok_len = (ctx[:, None] - next_n + offs[None, :] + 1).clamp_min(0)
    tok_len = torch.where(ctx[:, None] > 0, tok_len, torch.zeros_like(tok_len))
    seq_lens = (tok_len // KPOOL).to(torch.int32).reshape(rows, 1).contiguous()

    q_bf16 = torch.randn(rows, heads, HEAD_DIM, generator=gen, device=device)
    q_fp8, q_scale = fwht128_quant_fp8(q_bf16.to(torch.bfloat16).view(-1, HEAD_DIM))
    q_fp8 = q_fp8.view(rows, 1, heads, HEAD_DIM)
    w_raw = torch.randn(rows, heads, generator=gen, device=device)
    weights = (
        w_raw * q_scale.view(rows, heads).float() * (HEAD_DIM**-0.5 * heads**-0.5)
    ).contiguous()
    return dict(
        q=q_fp8,
        kv=kv3.unsqueeze(-2),  # kv_cache_as_quant_view: [pages, page, 1, D+4]
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        max_model_len=max_model_len,
        rows=rows,
        max_seq_len_tokens=max(req_contexts),
        contexts=list(req_contexts),
        schedule_metadata=torch.zeros(
            (torch.cuda.get_device_properties(device).multi_processor_count + 1, 2),
            dtype=torch.int32,
            device=device,
        ),
    )


def run_fallback(c: dict) -> torch.Tensor:
    return fp8_fp4_paged_mqa_logits(
        (c["q"], None),
        c["kv"],
        c["weights"],
        c["seq_lens"],
        c["block_table"],
        c["schedule_metadata"],
        max_model_len=c["max_model_len"],
        clean_logits=False,
    )


def run_tilelang(c: dict, split: int | None, out: torch.Tensor | None = None):
    logits = tilelang_fp8_paged_mqa_logits(
        c["q"],
        c["kv"],
        c["weights"],
        c["seq_lens"],
        c["block_table"],
        c["max_model_len"],
        split_kv=split,
        out=out,
    )
    if logits is None:
        raise RuntimeError("TileLang wrapper rejected the vLLM GLM layout")
    return logits


def time_us(fn, iters: int, warmup: int, flush: torch.Tensor | None, graph: bool):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    run = fn
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return t[len(t) // 2], t[len(t) // 10], t[(len(t) * 9) // 10]


def compare(c: dict, ref: torch.Tensor, got: torch.Tensor, select_k: int) -> dict:
    """Numerics over each row's valid prefix, plus top-k set overlap."""
    lens = c["seq_lens"].view(-1).tolist()
    max_abs = max_norm = max_rel = 0.0
    overlaps: list[float] = []
    nonfinite = 0
    for r, n in enumerate(lens):
        if n <= 0:
            continue
        a, b = got[r, :n], ref[r, :n]
        nonfinite += int((~torch.isfinite(a)).sum())
        d = (a - b).abs()
        scale = float(b.abs().max())
        max_abs = max(max_abs, float(d.max()))
        max_norm = max(max_norm, float(d.max()) / max(scale, 1e-30))
        max_rel = max(
            max_rel, float((d / b.abs().clamp_min(1e-3 * max(scale, 1e-30))).max())
        )
        k = min(select_k, n)
        sa = set(torch.topk(a, k).indices.tolist())
        sb = set(torch.topk(b, k).indices.tolist())
        overlaps.append(len(sa & sb) / k)
    out = dict(
        max_abs_diff=max_abs,
        max_norm_diff=max_norm,
        max_rel_diff=max_rel,
        nonfinite=nonfinite,
        topk_overlap_min=min(overlaps) if overlaps else 1.0,
        topk_overlap_mean=sum(overlaps) / len(overlaps) if overlaps else 1.0,
    )
    out.update(persistent_topk_compare(c, ref, got, select_k))
    return out


def persistent_topk_compare(c, ref, got, select_k) -> dict:
    op = getattr(torch.ops._C, "persistent_topk", None)
    if op is None:
        return dict(ptopk="unavailable")
    rows = c["rows"]
    ws = torch.zeros(1024 * 1024, dtype=torch.uint8, device=ref.device)
    res = []
    for logits in (ref, got):
        dst = torch.empty((rows, select_k), dtype=torch.int32, device=ref.device)
        op(logits, c["seq_lens"], dst, ws, select_k, c["max_seq_len_tokens"])
        res.append(dst)
    torch.cuda.synchronize()
    lens = c["seq_lens"].view(-1).tolist()
    overlaps, out_of_range = [], 0
    for r, n in enumerate(lens):
        if n <= 0:
            continue
        a = [i for i in res[1][r].tolist() if i >= 0]
        b = [i for i in res[0][r].tolist() if i >= 0]
        out_of_range += sum(1 for i in a if i >= n)
        k = min(select_k, n)
        overlaps.append(len(set(a) & set(b)) / k)
    return dict(
        ptopk_overlap_min=min(overlaps) if overlaps else 1.0,
        ptopk_out_of_range=out_of_range,
    )


def check_ok(m: dict, args) -> list[str]:
    bad = []
    if m["nonfinite"]:
        bad.append(f"nonfinite={m['nonfinite']}")
    if m["max_norm_diff"] > args.norm_tol:
        bad.append(f"max_norm_diff={m['max_norm_diff']:.3e}>{args.norm_tol}")
    if m["topk_overlap_min"] < args.overlap_min:
        bad.append(f"topk_overlap_min={m['topk_overlap_min']:.4f}")
    if m.get("ptopk") != "unavailable":
        if m["ptopk_overlap_min"] < args.overlap_min:
            bad.append(f"ptopk_overlap_min={m['ptopk_overlap_min']:.4f}")
        if m["ptopk_out_of_range"]:
            bad.append(f"ptopk_out_of_range={m['ptopk_out_of_range']}")
    return bad


def emit(rec: dict, fh) -> None:
    line = json.dumps(rec, sort_keys=True)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rows", default="6,12,24")
    p.add_argument("--context", default="8192,32768,65536,131072")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--next-n", type=int, default=MTP_ROWS_PER_REQUEST)
    p.add_argument("--split", default="auto", help="comma list; 'auto' = default")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--no-flush-l2", action="store_true")
    p.add_argument("--eager", action="store_true", help="time without CUDA graph")
    p.add_argument("--check", action="store_true", help="correctness only")
    p.add_argument("--norm-tol", type=float, default=5e-3)
    p.add_argument("--overlap-min", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="also append JSON lines here")
    args = p.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(device)
    fh = open(args.out, "a") if args.out else None
    select_k = INDEX_TOPK // KPOOL
    rows_list = [int(x) for x in args.rows.split(",")]
    ctx_list = [int(x) for x in args.context.split(",")]
    splits = [None if s == "auto" else int(s) for s in args.split.split(",")]
    base = dict(
        gpu=torch.cuda.get_device_name(device),
        capability=f"{cap[0]}.{cap[1]}",
        fallback_impl="sm12x_triton_tf32" if cap == (8, 9) else "deep_gemm_or_other",
        heads=args.heads,
        page_size=args.page_size,
        max_model_len=args.max_model_len,
        next_n=args.next_n,
    )
    failures = 0

    if args.check:
        cases = [
            (f"uniform-r{r}-c{c}", [c] * (r // args.next_n))
            for r in rows_list
            for c in ctx_list
        ]
        # Mixed lengths: partial page, just past the 512-pool top-k threshold,
        # long, and a CUDA-graph padding request of length 0.
        cases.append(("mixed", [70, 2051, 131072, 0]))
        for name, reqs in cases:
            c = build_case(
                reqs, args.heads, args.page_size, args.max_model_len,
                args.next_n, args.seed, device,
            )
            ref = run_fallback(c)
            got = torch.full_like(ref, float("inf"))
            run_tilelang(c, None, out=got)
            torch.cuda.synchronize()
            m = compare(c, ref, got, select_k)
            bad = check_ok(m, args)
            failures += bool(bad)
            emit(dict(base, mode="check", case=name, rows=c["rows"],
                      contexts=reqs, passed=not bad, failures=bad, **m), fh)

        # Graph replay with changed lengths / block tables (same shapes).
        a = build_case([4096, 131072], args.heads, args.page_size,
                       args.max_model_len, args.next_n, args.seed, device)
        b = build_case([131072, 777], args.heads, args.page_size,
                       args.max_model_len, args.next_n, args.seed + 1, device)
        static = dict(a)
        static["kv"] = torch.zeros(
            (max(a["kv"].shape[0], b["kv"].shape[0]),) + tuple(a["kv"].shape[1:]),
            dtype=torch.uint8, device=device,
        )
        static["kv"][: a["kv"].shape[0]].copy_(a["kv"])
        out = torch.full(
            (a["rows"], args.max_model_len), float("inf"), device=device
        )
        run_tilelang(static, None, out=out)  # compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_tilelang(static, None, out=out)
        for src in (a, b):
            static["kv"][: src["kv"].shape[0]].copy_(src["kv"])
            for key in ("q", "weights", "seq_lens", "block_table"):
                static[key].copy_(src[key])
            out.fill_(float("inf"))
            g.replay()
            torch.cuda.synchronize()
            ref = run_fallback(dict(src))
            m = compare(dict(src), ref, out, select_k)
            bad = check_ok(m, args)
            failures += bool(bad)
            emit(dict(base, mode="check", case="graph-replay",
                      contexts=src["contexts"], passed=not bad, failures=bad,
                      **m), fh)
        print(f"CHECK {'FAILED' if failures else 'PASSED'} ({failures} failing)",
              file=sys.stderr)
        return 1 if failures else 0

    flush = None
    if not args.no_flush_l2:
        flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    for rows in rows_list:
        assert rows % args.next_n == 0, "rows must be a multiple of --next-n"
        for ctx in ctx_list:
            t0 = time.time()
            c = build_case(
                [ctx] * (rows // args.next_n), args.heads, args.page_size,
                args.max_model_len, args.next_n, args.seed, device,
            )
            ref = run_fallback(c)
            got = run_tilelang(c, None)
            torch.cuda.synchronize()
            m = compare(c, ref, got, select_k)
            fb = time_us(lambda: run_fallback(c), args.iters, args.warmup, flush,
                         not args.eager)
            sweep = {}
            for s in splits:
                key = "auto" if s is None else str(s)
                sweep[key] = time_us(lambda s=s: run_tilelang(c, s), args.iters,
                                     args.warmup, flush, not args.eager)[0]
            tl_key = "auto" if None in splits else next(iter(sweep))
            rec = dict(
                base,
                mode="bench",
                rows=rows,
                requests=rows // args.next_n,
                context_tokens=ctx,
                pooled_len=ctx // KPOOL,
                split_auto=default_split_kv(rows, c["block_table"].shape[1], device),
                timing="eager" if args.eager else "cuda_graph",
                l2_flush=flush is not None,
                fallback_us=fb[0],
                fallback_p10_us=fb[1],
                fallback_p90_us=fb[2],
                tilelang_us=sweep[tl_key],
                tilelang_split_us=sweep,
                speedup=fb[0] / sweep[tl_key],
                saved_x11_layers_us=MLA_LAYERS_PER_TARGET_FORWARD
                * (fb[0] - sweep[tl_key]),
                check_failures=check_ok(m, args),
                setup_s=round(time.time() - t0, 1),
                **m,
            )
            emit(rec, fh)
            del c, ref, got
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
