# Historical validation record (superseded)

> This record predates removal of the SenseVoice SpaceMIT ONNX Runtime backend.
> Its EP linkage and dependency observations do not describe the current
> `llama-funasr-sensevoice` binary, which now defaults to the ggml CPU path.

Date: 2026-08-13

## Build and static checks

The following targets were built successfully with the SpaceMIT OHOS clang 21.1.8 toolchain and SpaceMIT llama.cpp v0.1.7:

```text
llama-funasr-cli
llama-funasr-encoder
llama-funasr-embd
llama-funasr-sensevoice
llama-funasr-paraformer
llama-funasr-vad
```

All six packaged executables were checked as ELF64 RISC-V PIE executables using the musl loader.

The staged SenseVoice executable is:

```text
build-riscv64-ohos/package/bin/llama-funasr-sensevoice
```

Observed `file` summary:

```text
ELF 64-bit LSB pie executable, UCB RISC-V, RVC, double-float ABI,
dynamically linked, interpreter /lib/ld-musl-riscv64.so.1
```

Observed ELF header fields:

```text
Class:   ELF64
Machine: RISC-V
Flags:   RVC, double-float ABI
```

Observed dynamic entries:

```text
RUNPATH: $ORIGIN/../lib
NEEDED:  libonnxruntime.so.1
NEEDED:  libspacemit_ep.so.2
NEEDED:  libatomic.so.1
NEEDED:  libgcc_s.so.1
NEEDED:  libc.so
```

Complete generated output is in `build-riscv64-ohos/package/logs/`.

Source-level backend tests passed (`4 passed`), as did shell syntax checks for both scripts.

## Target execution status

No HDC target was connected when validation was run: `hdc list targets` returned an empty target list. Consequently, CPU/GGUF transcription, SpaceMIT EP transcription, transcript comparison, and CPU-versus-EP timing could not be truthfully recorded in this workspace run.

Do not treat the successful cross-build or EP linkage as runtime acceptance. Connect the K3/K1 target and follow `README-riscv64-spacemit-ohos.md`; retain both command logs and the comparison table before declaring target acceptance.
