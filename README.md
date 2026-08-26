# ASR4K3OH6.1

RISC-V OHOS FunASR Runtime for the SpaceMIT K3/K1 platform.

This repository contains the C++/GGUF deployment subset of FunASR. It does not
require Python or PyTorch on the target device.

## Included targets

```text
llama-funasr-cli
llama-funasr-stream
llama-funasr-encoder
llama-funasr-embd
llama-funasr-sensevoice
llama-funasr-paraformer
llama-funasr-vad
llama-funasr-vad-stream
```

Prebuilt RISC-V OHOS binaries are under
`build-riscv64-ohos/package/bin/`. GGUF model files are intentionally not
stored in this repository; download or generate them according to the model
README files.

## Build

Prerequisites:

- SpaceMIT RISC-V OHOS toolchain
- SpaceMIT llama.cpp checkout and its `riscv64-spacemit-ohos.cmake` toolchain
  file
- CMake and a host C++ build environment

From the FunASR checkout:

```bash
scripts/build-riscv64-spacemit-ohos.sh
```

The script builds and stages the eight targets listed above, then records ELF
header and dynamic dependency information under `build-riscv64-ohos/package/logs/`.

## Board deployment

The binaries are tested on RISC-V OHOS with:

```bash
export SPACEMIT_DISABLE_TCM=1
```

See [README-riscv64-spacemit-ohos.md](README-riscv64-spacemit-ohos.md) for
model paths, `hdc` deployment commands, Nano streaming input, and acceptance
results. See [VALIDATION-riscv64-spacemit-ohos.md](VALIDATION-riscv64-spacemit-ohos.md)
for validation notes.

## Runtime scope

The target package uses GGUF + ggml/llama.cpp and supports offline ASR,
streaming PCM ASR, SenseVoice, Paraformer, offline VAD, and streaming VAD.
CAM++ speaker recognition, IME2-specific SenseVoice loading, and the batch
benchmark executable are not included.
