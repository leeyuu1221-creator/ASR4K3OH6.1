#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$script_dir/.." && pwd)
workspace_dir=$(cd -- "$source_dir/.." && pwd)

export RISCV_ROOT_PATH="${RISCV_ROOT_PATH:-$workspace_dir/spacemit-toolchain-linux-musl-x86_64-oh-20260630}"
SPACEMIT_LLAMA_DIR="${SPACEMIT_LLAMA_DIR:-$workspace_dir/spacemit-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$source_dir/build-riscv64-ohos}"
JOBS="${JOBS:-$(nproc)}"

test -x "$RISCV_ROOT_PATH/bin/clang"
test -d "$RISCV_ROOT_PATH/sysroot"
test -f "$SPACEMIT_LLAMA_DIR/cmake/riscv64-spacemit-ohos.cmake"

cmake -S "$source_dir/runtime/llama.cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$SPACEMIT_LLAMA_DIR/cmake/riscv64-spacemit-ohos.cmake" \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA="$SPACEMIT_LLAMA_DIR" \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_RISCV64_SPACEMIT=ON \
  -DGGML_CPU_REPACK=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_RVV=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_RV_ZBA=ON \
  -DLLAMA_CURL=OFF

targets=(
  llama-funasr-cli
  llama-funasr-stream
  llama-funasr-encoder
  llama-funasr-embd
  llama-funasr-sensevoice
  llama-funasr-paraformer
  llama-funasr-vad
  llama-funasr-vad-stream
)
cmake --build "$BUILD_DIR" -j"$JOBS" --target "${targets[@]}"

package_dir="$BUILD_DIR/package"
rm -rf "$package_dir"
mkdir -p "$package_dir/bin" "$package_dir/lib" "$package_dir/logs"
for target in "${targets[@]}"; do
  cp "$BUILD_DIR/bin/$target" "$package_dir/bin/"
done
file "$package_dir/bin/llama-funasr-sensevoice" | tee "$package_dir/logs/file.txt"
readelf -h "$package_dir/bin/llama-funasr-sensevoice" | tee "$package_dir/logs/readelf-header.txt"
readelf -d "$package_dir/bin/llama-funasr-sensevoice" | tee "$package_dir/logs/readelf-dynamic.txt"
echo "Package: $package_dir"
