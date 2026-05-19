#!/usr/bin/env python3
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License.

Runner for the pto-dsl flash-attention port. By default this invokes
`bash compile.sh` for the current `FA_Q_ROWS` (total Q seqlen), loads the
resulting .so, then benchmarks each `FA_BENCH_LENGTHS` entry as the KV seqlen
S1. Pass `--no-build` to reuse an existing `build_artifacts/fa.so`.

When `FA_Q_ROWS` and `FA_BENCH_LENGTHS` are not set, the default benchmark
matrix matches `fa_benchmark_shapes.md` and is named case1..case8.
"""

import argparse
import csv
import ctypes
import datetime
import math
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401  -- registers the npu backend

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "kernels"))
import fa_builder  # noqa: E402

from ptodsl import do_bench  # noqa: E402
from ptodsl.utils.npu_info import get_num_cube_cores, get_test_device  # noqa: E402

ARTIFACT_DIR = os.path.join(THIS_DIR, "build_artifacts")
COMPILE_SCRIPT = os.path.join(THIS_DIR, "compile.sh")
SUMMARY_TSV = os.environ.get("FA_SUMMARY_TSV", "")
CASE_ID = os.environ.get("FA_CASE_ID", "direct")

# Default single-case length. The top-level default path runs DEFAULT_BENCH_CASES
# unless FA_Q_ROWS or FA_BENCH_LENGTHS is explicitly set.
DEFAULT_BENCH_LENGTHS = (1024,)
DEFAULT_BENCH_CASES = (
    ("case1", 1024, 1024),
    ("case2", 2048, 2048),
    ("case3", 4096, 4096),
    ("case4", 8192, 8192),
    ("case5", 16384, 16384),
    ("case6", 32768, 32768),
    ("case7", 65536, 65536),
    ("case8", 131072, 131072),
)

ATOL = 1e-3
RTOL = 1e-3
# Skip the host fp32 reference once the QK matrix would exceed ~256M fp32
# elements (1 GiB host RAM). Above this we still assert against the NPU fused
# reference (torch_npu.npu_fused_infer_attention_score), with looser tolerance.
HOST_REF_MAX_QK_ELEMS = 256 * 1024 * 1024
ATOL_FUSED_ONLY = 5e-3
RTOL_FUSED_ONLY = 5e-3


def _manual_target_summary():
    return (
        "manual target: S0={s0} HEAD={head} CUBE_S0={cube_s0} "
        "CUBE_S1={cube_s1} TILE_S1={tile_s1} QK_PRELOAD={preload} "
        "causal={causal}"
    ).format(
        s0=fa_builder.MANUAL_S0,
        head=fa_builder.MANUAL_HEAD,
        cube_s0=fa_builder.MANUAL_CUBE_S0,
        cube_s1=fa_builder.MANUAL_CUBE_S1,
        tile_s1=fa_builder.MANUAL_TILE_S1,
        preload=fa_builder.MANUAL_QK_PRELOAD,
        causal=fa_builder.MANUAL_CAUSAL_MASK,
    )


def _dsl_effective_summary():
    return (
        "dsl effective: Q_ROWS={q_rows} HEAD={head} CUBE_S0={cube_s0} "
        "S1_TILE={s1_tile} QK_PRELOAD={preload} NUM_Q_BLOCKS={q_blocks}"
    ).format(
        q_rows=fa_builder.Q_ROWS,
        head=fa_builder.HEAD,
        cube_s0=fa_builder.S0,
        s1_tile=fa_builder.S1_TILE,
        preload=fa_builder.QK_PRELOAD,
        q_blocks=fa_builder.NUM_Q_BLOCKS,
    )


def _bench_lengths():
    raw = os.environ.get("FA_BENCH_LENGTHS", "")
    if not raw:
        return DEFAULT_BENCH_LENGTHS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _bench_iters():
    return int(os.environ.get("FA_BENCH_ITERS", "100"))


def _bench_warmup():
    return int(os.environ.get("FA_BENCH_WARMUP", "10"))


def _default_suite_requested(args):
    return args.case is not None or ("FA_Q_ROWS" not in os.environ and "FA_BENCH_LENGTHS" not in os.environ)


def _selected_default_cases(case_arg):
    if not case_arg:
        return DEFAULT_BENCH_CASES

    lookup = {case_id: (case_id, s0, s1) for case_id, s0, s1 in DEFAULT_BENCH_CASES}
    selected = []
    for name in (part.strip() for part in case_arg.split(",")):
        if not name:
            continue
        if name not in lookup:
            valid = ", ".join(case_id for case_id, _, _ in DEFAULT_BENCH_CASES)
            raise ValueError("unknown case '{}'; valid cases: {}".format(name, valid))
        selected.append(lookup[name])
    return tuple(selected)


def _default_summary_tsv():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(ARTIFACT_DIR, "fa_summary_{}_{}.tsv".format(stamp, os.getpid()))


def _summary_fields():
    return [
        "case_id",
        "q_rows",
        "seq_len",
        "tiles",
        "status",
        "fa_us",
        "fa_tflops",
        "fused_us",
        "fused_tflops",
        "speedup",
        "err_kernel",
        "err_fused",
        "ref",
        "note",
    ]


def _append_summary_row(row):
    if not SUMMARY_TSV:
        return
    exists = os.path.exists(SUMMARY_TSV)
    summary_dir = os.path.dirname(SUMMARY_TSV)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(SUMMARY_TSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_summary_fields(), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in _summary_fields()})


def _num_tiles(seq_len):
    if seq_len % fa_builder.S1_TILE != 0:
        raise ValueError("seq_len {} is not a multiple of S1_TILE={}".format(seq_len, fa_builder.S1_TILE))
    num_tiles = seq_len // fa_builder.S1_TILE
    if num_tiles < fa_builder.QK_PRELOAD or (num_tiles - fa_builder.QK_PRELOAD) % fa_builder.QK_PRELOAD != 0:
        raise ValueError(
            "seq_len {} maps to {} tiles; the current QK_PRELOAD={} runtime loop requires "
            "num_tiles >= QK_PRELOAD and (num_tiles - QK_PRELOAD) % QK_PRELOAD == 0".format(
                seq_len, num_tiles, fa_builder.QK_PRELOAD
            )
        )
    return num_tiles


def _lib_path():
    return os.path.join(ARTIFACT_DIR, "fa.so")


def _require_lib():
    p = _lib_path()
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Missing prebuilt .so: {}\n"
            "    Run `bash compile.sh` (or `python3 run.py` without --no-build) for the current "
            "FA_Q_ROWS={} to build the runtime-S1 kernel.".format(p, fa_builder.Q_ROWS)
        )
    return p


def _build_lib():
    print("[fa] compiling PTODSL flash kernel...")
    subprocess.run(["bash", COMPILE_SCRIPT], cwd=THIS_DIR, check=True)
    return _require_lib()


def _load_lib(path):
    lib = ctypes.CDLL(path)
    lib.call_kernel.argtypes = [ctypes.c_uint32] + [ctypes.c_void_p] * 6 + [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def _to_void_p(t):
    return ctypes.c_void_p(t.data_ptr())


def _block_dim():
    return min(fa_builder.NUM_Q_BLOCKS, get_num_cube_cores())


def _slot_elems(block_dim):
    return fa_builder.GM_ELEMS_PER_BLOCK * block_dim


def attn_flops_matmul_softmax_scale(
    batch_size, s_q, s_k, h, include_scale=True, count_exp_as_flop=True, count_max_as_flop=True
):
    flops_matmul = 4 * batch_size * s_q * s_k * h
    flops_scale = batch_size * s_q * s_k if include_scale else 0

    rows = batch_size * s_q
    softmax_ops = 0
    if count_max_as_flop:
        softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k
    if count_exp_as_flop:
        softmax_ops += rows * s_k
    softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k

    return flops_matmul + flops_scale + softmax_ops


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def fa_reference(q, k, v):
    """Plain torch fp32 reference: O = softmax(QK^T * scale) @ V."""
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    return (torch.softmax(scores, dim=-1) @ v.float()).float()


def fused_attention(q, k, v):
    """torch_npu fused reference for benchmarking the speedup ratio."""
    scale = 1.0 / math.sqrt(q.shape[1])
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), num_heads=1, input_layout="BSH", scale=scale, next_tokens=65535
    )
    return out.squeeze(0)


def _alloc_io(seq_len, device):
    Q_ROWS = fa_builder.Q_ROWS
    HEAD = fa_builder.HEAD
    block_dim = _block_dim()

    q = torch.randn((Q_ROWS, HEAD), dtype=torch.float16, device=device)
    k = torch.randn((seq_len, HEAD), dtype=torch.float16, device=device)
    v = torch.randn((seq_len, HEAD), dtype=torch.float16, device=device)
    gm_slot = torch.zeros((_slot_elems(block_dim),), dtype=torch.float32, device=device)
    o = torch.zeros((Q_ROWS, HEAD), dtype=torch.float32, device=device)
    return q, k, v, gm_slot, o, block_dim


def _invoke(lib, block_dim, gm_slot, q, k, v, o, stream_ptr):
    lib.call_kernel(
        block_dim,
        stream_ptr,
        _to_void_p(gm_slot),
        _to_void_p(q),
        _to_void_p(k),
        _to_void_p(v),
        _to_void_p(o),
        q.shape[0],
        k.shape[0],
    )


def benchmark(lib, device, num_tiles, warmup, iters):
    torch.manual_seed(0)
    seq_len = fa_builder.S1_TILE * num_tiles
    q, k, v, gm_slot, o, block_dim = _alloc_io(seq_len, device)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    def run_kernel():
        _invoke(lib, block_dim, gm_slot, q, k, v, o, stream_ptr)

    def run_fused():
        fused_attention(q, k, v)

    kernel_us = do_bench(run_kernel, warmup_iters=warmup, benchmark_iters=iters, unit="us")
    fused_us = do_bench(run_fused, warmup_iters=warmup, benchmark_iters=iters, unit="us")

    # One untimed correctness probe per length so silent miscompiles don't
    # hide behind a passing benchmark.
    run_kernel()
    torch.npu.synchronize()
    o_kernel = o.clone()
    o_fused = fused_attention(q, k, v)
    torch.npu.synchronize()

    qk_elems = q.shape[0] * k.shape[0]
    use_host_ref = qk_elems <= HOST_REF_MAX_QK_ELEMS
    if use_host_ref:
        o_golden = fa_reference(q, k, v)
        err_kernel = (o_kernel.cpu().float() - o_golden.cpu()).abs().max().item()
        err_fused = (o_fused.cpu().float() - o_golden.cpu()).abs().max().item()
        torch.testing.assert_close(o_kernel.cpu().float(), o_golden.cpu(), rtol=RTOL, atol=ATOL)
        ref_label = "host_fp32"
    else:
        # Host fp32 reference would need ~{qk_elems*4} bytes for the QK^T
        # matrix; fall back to the NPU fused reference as the correctness
        # baseline with a slightly looser tolerance.
        err_kernel = (o_kernel.cpu().float() - o_fused.cpu().float()).abs().max().item()
        err_fused = 0.0
        torch.testing.assert_close(
            o_kernel.cpu().float(), o_fused.cpu().float(), rtol=RTOL_FUSED_ONLY, atol=ATOL_FUSED_ONLY
        )
        ref_label = "npu_fused (host fp32 ref skipped, qk_elems={:.1e})".format(qk_elems)

    matmul_flops = 4 * fa_builder.Q_ROWS * fa_builder.HEAD * seq_len
    attention_flops = attn_flops_matmul_softmax_scale(1, fa_builder.Q_ROWS, seq_len, fa_builder.HEAD)
    kernel_ms = kernel_us / 1000.0
    fused_ms = fused_us / 1000.0
    return {
        "seq_len": seq_len,
        "num_tiles": num_tiles,
        "kernel_us": kernel_us,
        "fused_us": fused_us,
        "kernel_gflops": matmul_flops / (kernel_us * 1e-6) / 1e9,
        "fused_gflops": matmul_flops / (fused_us * 1e-6) / 1e9,
        "kernel_tflops": tflops(attention_flops, kernel_ms),
        "fused_tflops": tflops(attention_flops, fused_ms),
        "speedup": fused_us / kernel_us,
        "err_kernel": err_kernel,
        "err_fused": err_fused,
        "ref": ref_label,
    }


def _print_row(r):
    print(
        "  s0={s0:>6} s1={seq_len:>6}  tiles={num_tiles:>3}  "
        "fa={kernel_us:8.2f}us ({kernel_gflops:7.1f} GF/s, {kernel_tflops:6.2f} TFLOP/s)  "
        "npu_fused_infer_attention={fused_us:8.2f}us "
        "({fused_gflops:7.1f} GF/s, {fused_tflops:6.2f} TFLOP/s)  "
        "speedup={speedup:.2f}x  "
        "err: ours={err_kernel:.2e} npu_fused_infer_attention={err_fused:.2e}  "
        "ref={ref}".format(s0=fa_builder.Q_ROWS, **r)
    )


def _append_benchmark_summary(result, status="OK", note=""):
    _append_summary_row(
        {
            "case_id": CASE_ID,
            "q_rows": fa_builder.Q_ROWS,
            "seq_len": result["seq_len"],
            "tiles": result["num_tiles"],
            "status": status,
            "fa_us": "{:.2f}".format(result["kernel_us"]),
            "fa_tflops": "{:.2f}".format(result["kernel_tflops"]),
            "fused_us": "{:.2f}".format(result["fused_us"]),
            "fused_tflops": "{:.2f}".format(result["fused_tflops"]),
            "speedup": "{:.2f}".format(result["speedup"]),
            "err_kernel": "{:.2e}".format(result["err_kernel"]),
            "err_fused": "{:.2e}".format(result["err_fused"]),
            "ref": result["ref"],
            "note": note,
        }
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Build/run the pto-dsl flash-attention benchmark.")
    parser.add_argument(
        "--no-build", action="store_true", help="skip compile.sh and load the existing build_artifacts/fa.so"
    )
    parser.add_argument(
        "--case", help="comma-separated default benchmark cases to run (case1..case8); defaults to all cases"
    )
    return parser.parse_args()


def _run_default_suite(args):
    cases = _selected_default_cases(args.case)
    if not cases:
        raise ValueError("no benchmark cases selected")
    if args.no_build and len(cases) != 1:
        raise ValueError(
            "--no-build is only valid for a single selected case because build_artifacts/fa.so is rebuilt per FA_Q_ROWS"
        )

    summary_tsv = SUMMARY_TSV or _default_summary_tsv()
    print("[fa] running default benchmark suite from fa_benchmark_shapes.md")
    print("[fa] selected cases: {}".format(", ".join(case_id for case_id, _, _ in cases)))
    print("[fa] summary TSV: {}".format(summary_tsv))

    for case_id, s0, s1 in cases:
        print("\n{:=^110}".format(" {} S0={} S1={} ".format(case_id, s0, s1)))
        env = os.environ.copy()
        env.update(
            {"FA_CASE_ID": case_id, "FA_Q_ROWS": str(s0), "FA_BENCH_LENGTHS": str(s1), "FA_SUMMARY_TSV": summary_tsv}
        )
        cmd = [sys.executable, os.path.abspath(__file__)]
        if args.no_build:
            cmd.append("--no-build")
        subprocess.run(cmd, cwd=THIS_DIR, env=env, check=True)

    print("\n[fa] default benchmark suite done.")
    print("[fa] summary TSV: {}".format(summary_tsv))


def main():
    args = _parse_args()
    if _default_suite_requested(args):
        _run_default_suite(args)
        return

    device = get_test_device()
    torch.npu.set_device(device)

    lengths = _bench_lengths()
    targets = [(L, _num_tiles(L)) for L in lengths]
    lib_path = _require_lib() if args.no_build else _build_lib()
    lib = _load_lib(lib_path)

    warmup = _bench_warmup()
    iters = _bench_iters()

    print("\n{:=^110}".format(" Benchmark (pto-dsl fa) "))
    print("  " + _manual_target_summary())
    print("  " + _dsl_effective_summary())
    print("  same host-visible shape and QK_PRELOAD as the manual non-causal S0=128 path")
    print("  TFLOP/s counts matmul + scale + softmax operations, matching 140tflops/run.py")
    print("  reference kernel: torch_npu.npu_fused_infer_attention_score")
    print("  host fp32 reference is skipped when Q_ROWS*S1 > {}".format(HOST_REF_MAX_QK_ELEMS))
    print("  cores={}  warmup={}  iters={}".format(get_num_cube_cores(), warmup, iters))
    print("  S0(Q_ROWS)={}  S1 lengths: {}".format(fa_builder.Q_ROWS, list(lengths)))
    print("-" * 110)

    for _, nt in targets:
        result = benchmark(lib, device, num_tiles=nt, warmup=warmup, iters=iters)
        _print_row(result)
        _append_benchmark_summary(result)

    print("=" * 110)


if __name__ == "__main__":
    main()
