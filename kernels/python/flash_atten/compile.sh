#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# Build the pto-dsl flash-attention runtime-S1 .so. The generated kernel loops
# over s1 / S1_TILE at runtime, so one fa.{mlir,cpp,so} covers all supported
# benchmark lengths.
#
# Usage:
#   bash compile.sh                                # build build_artifacts/fa.so
#   PTO_LIB_PATH=/abs/pto-isa bash compile.sh      # override include path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
PTOAS="${PTOAS:-ptoas}"
BISHENG="${BISHENG:-bisheng}"

mkdir -p "${ARTIFACT_DIR}"

MLIR_PATH="${ARTIFACT_DIR}/fa.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/fa.cpp"
LIB_PATH="${ARTIFACT_DIR}/fa.so"

echo "==> Building runtime-S1 fa -> ${LIB_PATH}"
rm -f "${MLIR_PATH}" "${GENERATED_CPP}" "${LIB_PATH}"

python "${SCRIPT_DIR}/kernels/fa_builder.py" > "${MLIR_PATH}"
"${PTOAS}" --pto-arch=a3 --enable-insert-sync "${MLIR_PATH}" > "${GENERATED_CPP}"

"${BISHENG}" \
    -I"${PTO_LIB_PATH}/include" \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -cce-enable-mix \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    -DKERNEL_CPP="\"${GENERATED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Done. Built ${LIB_PATH}"
