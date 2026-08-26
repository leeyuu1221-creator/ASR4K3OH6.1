# RISC-V OHOS 编译与部署

本文说明如何将 FunASR Runtime 编译为 SpaceMIT K3 + OpenHarmony 6.1 可运行的 RISC-V 版本。

## 1. 编译工具

主机端需要：

- Linux x86_64
- Git
- CMake 3.16 或更高版本
- Bash、Make 或 Ninja
- `file`、`readelf`、`nproc`
- `hdc`（推送和运行板端程序）

外部组件：

```text
spacemit-toolchain-linux-musl-x86_64-oh-20260630/
spacemit-llama.cpp/
```

工具链中必须存在：

```text
bin/clang
sysroot/
```

SpaceMIT llama.cpp 中必须存在：

```text
cmake/riscv64-spacemit-ohos.cmake
```

## 2. 推荐目录结构

```text
/home/user/project/
├── FunASR/
├── spacemit-toolchain-linux-musl-x86_64-oh-20260630/
└── spacemit-llama.cpp/
```

## 3. 编译

进入 FunASR 目录执行：

```bash
cd FunASR
scripts/build-riscv64-spacemit-ohos.sh
```

如路径不同，可以手动指定：

```bash
RISCV_ROOT_PATH=/path/to/toolchain \
SPACEMIT_LLAMA_DIR=/path/to/spacemit-llama.cpp \
BUILD_DIR=/tmp/funasr-ohos \
JOBS=8 \
scripts/build-riscv64-spacemit-ohos.sh
```

脚本会使用 Release 模式、C++17 和 SpaceMIT RISC-V CPU backend，并启用 RVV、ZFH、ZVFH 等指令集。

当前会构建以下八个目标：

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

## 4. 编译产物

```text
build-riscv64-ohos/package/bin/
```

脚本会使用 `file` 和 `readelf` 检查产物。正常结果应为：

```text
ELF64 RISC-V
PIE executable
musl loader: /lib/ld-musl-riscv64.so.1
```

## 5. 模型准备

编译不包含模型文件。根据使用的 Runtime 准备对应 GGUF：

```text
Fun-ASR-Nano:
  funasr-encoder-f16.gguf
  qwen3-0.6b-q4km.gguf
  fsmn-vad.gguf             # 使用 VAD 时需要

SenseVoice:
  sensevoice-small-q8.gguf

Paraformer:
  paraformer-f16.gguf
```

模型转换在主机端完成，板端只需要最终的 GGUF 文件。

## 6. 使用 hdc 推送

检查板端连接：

```bash
~/project/toolchains/hdc list targets
```

推送八个二进制：

```bash
~/project/toolchains/hdc \
  -t <device-id> \
  file send build-riscv64-ohos/package/bin \
  /data/data/funasr/bin
```

设置执行权限：

```bash
~/project/toolchains/hdc \
  -t <device-id> \
  shell 'chmod 755 /data/data/funasr/bin/*'
```

## 7. 板端运行

如果设备没有 `/dev/tcm_sync_mem`，先设置：

```bash
export SPACEMIT_DISABLE_TCM=1
```

SenseVoice：

```bash
/data/data/funasr/bin/llama-funasr-sensevoice \
  -m /data/data/funasr/sensevoice-small/sensevoice-small-q8.gguf \
  -a /data/data/funasr/test.wav
```

Fun-ASR-Nano CLI：

```bash
/data/data/funasr/bin/llama-funasr-cli \
  --enc /data/data/funasr/fun-asr-nano/funasr-encoder-f16.gguf \
  -m /data/data/funasr/fun-asr-nano/qwen3-0.6b-q4km.gguf \
  -a /data/data/funasr/test.wav \
  --output jsonl
```

Fun-ASR-Nano 流式识别：

```bash
dd if=/data/data/funasr/test.wav bs=44 skip=1 2>/dev/null | \
/data/data/funasr/bin/llama-funasr-stream \
  --enc /data/data/funasr/fun-asr-nano/funasr-encoder-f16.gguf \
  -m /data/data/funasr/fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --vad /data/data/funasr/fsmn-vad.gguf \
  --stdin-s16le \
  --output jsonl
```

流式输出为 JSONL 事件，包括：

```text
ready
partial
final
done
```

## 8. 验证要求

至少确认：

- 程序退出码为 0；
- 能加载对应 GGUF 模型；
- 识别结果正常输出；
- `file`/`readelf` 确认目标为 RISC-V ELF；
- 流式模式能够输出 `ready`、`final` 和 `done` 事件。

当前仓库的详细验证记录见
[VALIDATION-riscv64-spacemit-ohos.md](VALIDATION-riscv64-spacemit-ohos.md)。
