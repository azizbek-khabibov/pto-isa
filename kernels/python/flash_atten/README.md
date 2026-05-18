# Python DSL Flash Attention Example

## Overview

This example demonstrates a high-performance Flash Attention implementation written with the PTO Python DSL (`ptodsl`). It is a Python-DSL port and parity experiment for the manual kernel in `kernels/manual/common/flash_atten`, and it follows the same four-stage software pipeline:

```text
compute_qk (Cube) -> compute_p (Vector) -> compute_pv (Cube) -> compute_gu (Vector)
```

The implementation also references the Huawei CSL PTO DSL AOT Flash Attention 140 TFLOPS example:

```text
https://github.com/huawei-csl/pto-dsl/tree/main/examples/aot/flash_attention/140tflops
```

The case is useful for validating that the Python DSL can express a production-style Flash Attention pipeline, including Cube/Vector cooperation, runtime S1 looping, software FIFO staging through global memory, correctness checks, and performance comparison against `torch_npu.npu_fused_infer_attention_score`.

## Supported Platform

- Ascend A3-class target (`--pto-arch=a3`, `--npu-arch=dav-2201` in `compile.sh`)
- CANN environment with `bisheng`
- PTO assembler `ptoas`
- Python environment with `ptodsl`, `torch`, and `torch_npu`

## Directory Layout

```text
kernels/python/flash_atten/
├── caller.cpp              # Host shim exported as call_kernel for ctypes
├── compile.sh              # Generates MLIR/C++ and builds build_artifacts/fa.so
├── kernels/
│   └── fa_builder.py       # PTO Python DSL Flash Attention kernel builder
└── run.py                  # Build, run, verify, and benchmark entry point
```

Generated files are placed under `build_artifacts/`:

```text
build_artifacts/fa.mlir     # MLIR emitted by fa_builder.py
build_artifacts/fa.cpp      # C++ emitted by ptoas
build_artifacts/fa.so       # Shared library loaded by run.py
build_artifacts/fa_summary_*.tsv
```

## Kernel Scope

Current shape and feature constraints are intentionally aligned with the manual parity target:

- `HEAD = 128`
- `S0 = 128` per Q block
- `TILE_S1 = 256`
- `CUBE_S1 = 128`
- `QK_PRELOAD = 4`
- non-causal attention only
- total Q rows are configured by `FA_Q_ROWS` and must be a multiple of `128`
- total KV rows are supplied at runtime by `run.py`; each S1 length must be compatible with `S1_TILE=256` and `QK_PRELOAD=4`

The generated shared library is specialized for the current `FA_Q_ROWS`, while S1 is handled at runtime.

## Build and Run

1. Configure the Ascend CANN environment.

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Enter the example directory and set the PTO include path.

```bash
cd ${git_clone_path}/kernels/python/flash_atten
export PTO_LIB_PATH=${git_clone_path}
```

If `ptoas` or `bisheng` are not in `PATH`, set them explicitly:

```bash
export PTOAS=/path/to/ptoas
export BISHENG=/path/to/bisheng
```

3. Run one default benchmark case.

```bash
python3 run.py --case case1
```

4. Run the full default benchmark suite.

```bash
python3 run.py
```

The default suite runs `case1` to `case8` and recompiles the kernel for each `FA_Q_ROWS` value:

| Case | Q rows (S0 total) | KV rows (S1) |
| --- | ---: | ---: |
| `case1` | 1024 | 1024 |
| `case2` | 2048 | 2048 |
| `case3` | 4096 | 4096 |
| `case4` | 8192 | 8192 |
| `case5` | 16384 | 16384 |
| `case6` | 32768 | 32768 |
| `case7` | 65536 | 65536 |
| `case8` | 131072 | 131072 |

## Custom Cases

Run a custom shape by setting `FA_Q_ROWS` and `FA_BENCH_LENGTHS`:

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py
```

Run several S1 lengths for one compiled Q shape:

```bash
FA_Q_ROWS=2048 FA_BENCH_LENGTHS=1024,2048,4096 python3 run.py
```

Control benchmark iterations:

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 FA_BENCH_WARMUP=20 FA_BENCH_ITERS=200 python3 run.py
```

Reuse an existing `build_artifacts/fa.so` when it was already compiled for the same `FA_Q_ROWS`:

```bash
FA_Q_ROWS=1024 bash compile.sh
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py --no-build
```

## Output and Correctness

`run.py` prints latency, throughput, speedup, and max error for each shape. It compares the DSL kernel with:

- a host FP32 PyTorch reference when `Q_ROWS * S1` is small enough
- `torch_npu.npu_fused_infer_attention_score` for all benchmark sizes

Throughput is reported as TFLOP/s using matmul, scale, and softmax operation counts, following the 140 TFLOPS reference script convention.

A summary TSV is generated automatically for the default suite. You can choose the output path with `FA_SUMMARY_TSV`:

```bash
FA_SUMMARY_TSV=/tmp/fa_summary.tsv python3 run.py --case case1
```

## Notes

- `compile.sh` defaults `PTO_LIB_PATH` to `/sources/pto-isa`; set `PTO_LIB_PATH=${git_clone_path}` when working from this repository.
- `--no-build` is only suitable for one selected case because `fa.so` is rebuilt per `FA_Q_ROWS`.
- Large sequence lengths can skip the host FP32 reference to avoid allocating a very large QK matrix; correctness is then checked against the NPU fused reference with a looser tolerance.
