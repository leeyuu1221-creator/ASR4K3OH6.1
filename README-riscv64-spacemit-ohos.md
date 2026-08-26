# FunASR llama.cpp runtime for riscv64 SpaceMIT OHOS

`llama-funasr-sensevoice` runs SenseVoice GGUF models through ggml. Its default
backend is CPU; `--backend cpu` may be supplied explicitly. The former SpaceMIT
ONNX Runtime Execution Provider path is no longer supported.

## Prerequisites

Place these directories next to the FunASR checkout, or override their environment variables:

```text
spacemit-toolchain-linux-musl-x86_64-oh-20260630/
spacemit-llama.cpp/                  # tag v0.1.7
FunASR/                              # tag runtime-llamacpp-v0.1.9
```

## Build

```bash
cd FunASR
scripts/build-riscv64-spacemit-ohos.sh
```

Overrides:

```bash
RISCV_ROOT_PATH=/path/to/toolchain \
SPACEMIT_LLAMA_DIR=/path/to/spacemit-llama.cpp \
BUILD_DIR=/tmp/funasr-ohos JOBS=8 \
scripts/build-riscv64-spacemit-ohos.sh
```

The script builds and stages all runtime executables in `build-riscv64-ohos/package/bin`:

```text
llama-funasr-cli
llama-funasr-encoder
llama-funasr-embd
llama-funasr-sensevoice
llama-funasr-paraformer
llama-funasr-vad
llama-funasr-vad-stream
```

It also records SenseVoice `file`/`readelf` output under `package/logs`.

## Models and input signature

CPU requires a SenseVoice GGUF produced by `runtime/llama.cpp/sensevoice/export_sensevoice_gguf.py`.

## Run

On the target, stage `package/bin`, the GGUF model, and a 16 kHz audio file.

CPU/GGUF:

```bash
package/bin/llama-funasr-sensevoice \
  -m sensevoice-small-q8.gguf -a zh.wav
```

## Acceptance and benchmark

Retain stdout/stderr and compare decoded text as well as timing.

Record:

```bash
file build-riscv64-ohos/package/bin/llama-funasr-sensevoice
readelf -h build-riscv64-ohos/package/bin/llama-funasr-sensevoice
readelf -d build-riscv64-ohos/package/bin/llama-funasr-sensevoice
```

The expected ELF is ELF64 RISC-V with the musl loader. The SenseVoice binary no
longer has ONNX Runtime or SpaceMIT EP dynamic dependencies.
