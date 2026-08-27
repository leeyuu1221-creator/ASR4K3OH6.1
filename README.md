# ASR4K3OH6.1

面向 SpaceMIT K3 + OpenHarmony 6.1 的本地 ASR 部署仓库。

## 0. 仓库结构与内容

本仓库是面向 RISC-V OHOS 的 ASR 部署集合，不包含完整的 Python 训练工程。仓库同时整理了两条运行路线：基于 GGUF/ggml 的 `funasr-cpp`，以及基于 ONNX Runtime 的 `sherpa-onnx`。目标设备不需要安装 Python 或 PyTorch；使用 sherpa-onnx 时需要随程序部署对应的 ONNX Runtime 动态库。

```text
ASR4K3OH6.1/
├── README.md                      # 仓库总览和部署说明
├── funasr-cpp/                    # FunASR llama.cpp/ggml C++ Runtime
│   ├── runtime/llama.cpp/         # C++/ggml/llama.cpp 推理源码
│   │   ├── fun-asr-nano/          # Fun-ASR-Nano CLI、流式、Encoder
│   │   ├── sensevoice/            # SenseVoice CTC Runtime
│   │   ├── paraformer/            # Paraformer Runtime
│   │   ├── funasr-vad/            # 离线 FSMN-VAD
│   │   ├── funasr-vad-stream/     # 流式 FSMN-VAD
│   │   ├── funasr-common/         # 音频、VAD、公共头文件
│   │   ├── tests/                 # Runtime 测试和基准样例
│   │   └── CMakeLists.txt         # Runtime 构建入口
│   ├── scripts/                   # RISC-V OHOS 交叉编译脚本
│   ├── build-riscv64-ohos/package/# 已编译的板端二进制和 ELF 检查日志
│   └── BUILD_RISCV64_OHOS.md      # funasr-cpp 编译与部署说明
└── sherpa-onnx/                   # sherpa-onnx K3/OpenHarmony 构建资料
    ├── source/                    # sherpa-onnx 源码
    ├── dependencies/spacemit-ort/# SpacemiT ONNX Runtime 和 EP
    ├── dependencies/alsa/         # 可选 RISC-V64 musl ALSA 依赖
    ├── toolchain/                 # K3/OpenHarmony 交叉编译配置
    └── BUILD_K3_OHOS.md           # sherpa-onnx 编译、部署指南
```

当前仓库保留八个板端目标：`cli`、`stream`、`encoder`、`embd`、`sensevoice`、`paraformer`、`vad` 和 `vad-stream`。CAM++、SenseVoice IME2 和 CLI Bench 不在当前部署范围内。

## 1. 方案概述

本文主要说明在 **SpaceMIT K3 + OpenHarmony 6.1** 平台上部署本地 ASR 的可选方案，并对当前已验证的推理框架、模型性能、适用场景及工程集成方式进行对比。

目前在 K3 OpenHarmony 6.1 环境中，已重点验证以下两类 ASR 推理框架：

- **sherpa-onnx**
- **llama-funasr / llama-funasr-cpp**

两种方案的技术路线存在明显差异：

- **sherpa-onnx** 基于 ONNX Runtime，属于通用型语音推理框架，模型支持范围广，语音相关功能完整，目前支持在 K3 的通用 CPU 上进行推理。实测在 AI CPU 上推理速度更慢。
- **llama-funasr-cpp** 基于 llama.cpp / ggml 体系，主要面向 FunASR 模型的本地 C/C++ 推理，适合 Fun-ASR-Nano 等包含语言模型的 ASR 模型，支持使用 K3 的 AI CPU 加速。

从当前 K3 实测结果来看，推荐使用：

- **sherpa-onnx + SenseVoice-small INT8 ONNX**：4 核通用 CPU，RTF 约为 0.07，识别稳定，速度更快，并且模型自带标点与 ITN 规范。
- **llama-funasr-cpp + Fun-ASR-Nano**：8 核 AI CPU，RTF 约为 0.25～0.3，具有更强的上下文理解能力与专业名词准确性。

## 2. 推理框架对比

### 2.1 sherpa-onnx

sherpa-onnx 是基于 ONNX Runtime 的跨平台语音推理框架，覆盖 ASR、VAD、关键词检测、说话人处理、TTS 等多类语音任务。

目前 sherpa-onnx 的 K3/OpenHarmony 构建资料已整理在 [`sherpa-onnx/`](sherpa-onnx/) 中，其中包含源码、SpacemiT ONNX Runtime/EP、可选 ALSA 依赖和交叉编译配置。在 K3 上部署时需要：

1. 准备 `sherpa-onnx/` 中的源码和依赖；
2. 准备 K3 OpenHarmony 对应的交叉编译工具链；
3. 按 `sherpa-onnx/BUILD_K3_OHOS.md` 交叉编译 sherpa-onnx 及相关依赖；
4. 根据运行需求选择通用 CPU 后端或 SpaceMIT ONNX Runtime Execution Provider；
5. 将模型、动态库和应用程序部署到 K3 设备进行测试。

sherpa-onnx 的优势在于模型生态成熟，能够方便地替换不同 ASR 模型，而不需要大幅调整上层业务接口。

### 2.2 llama-funasr-cpp

llama-funasr-cpp 基于 llama.cpp、ggml 等基础组件实现，适合在 K3/OpenHarmony 上进行 FunASR 模型的本地 C/C++ 推理。当前支持 Fun-ASR-Nano、SenseVoice 和 Paraformer，并使用针对该 Runtime 导出的 GGUF 权重。

与 sherpa-onnx 相比，llama-funasr-cpp 的模型范围更集中，但对包含语言模型解码过程的 Fun-ASR-Nano 更有针对性。当前仓库在官方离线识别基础上补充了 Fun-ASR-Nano + FSMN-VAD 的分段式流式识别能力。

这里的“流式”是持续接收 PCM、进行 VAD 分段并输出 `partial/final` 事件；当前模型仍按音频窗口重新推理，不等同于严格的 causal streaming encoder。

### 2.3 框架能力对比

| 对比项 | sherpa-onnx | llama-funasr-cpp |
|---|---|---|
| 框架定位 | 通用 ONNX 语音推理框架 | 基于 llama.cpp / ggml 的 FunASR C/C++ 边缘推理框架 |
| 主要模型格式 | ONNX | GGUF |
| 运行时 | ONNX Runtime | ggml / llama.cpp 体系 |
| K3 运行方式 | X100 通用 CPU / SpaceMIT ONNX Runtime EP | AI CPU 相关优化 |
| 模型覆盖范围 | 较广 | 主要覆盖 FunASR 模型 |
| VAD | 支持 | 支持 FSMN-VAD |
| 流式 ASR | 支持 | 支持分段式流式输出 |
| 离线 ASR | 支持 | 支持 |
| 量化模型 | ONNX INT8 等 | GGUF Q4/Q5/Q8/F16 等 |
| 功能完整度 | 高 | 主要聚焦 ASR/VAD |
| 主要优势 | 架构成熟、模型丰富、接口稳定 | Runtime 轻量、量化友好、适合本地部署 |
| 主要不足 | SpaceMIT EP 对部分模型加速收益有限 | 模型数量和通用接口丰富度较低 |

需要特别注意，AI CPU / NPU 并不意味着一定比通用 CPU 更快。部署时应针对具体模型分别测试通用 CPU、AI CPU、SpaceMIT EP、线程数和量化精度。

## 3. K3 部署方式

### 3.1 sherpa-onnx

sherpa-onnx 的源码、SpacemiT ONNX Runtime/EP 依赖、可选 ALSA 依赖和构建指南位于 `sherpa-onnx/`。完整交叉工具链因体积过大未提交，具体获取方式见 `sherpa-onnx/BUILD_K3_OHOS.md`。模型和最终应用程序仍需根据实际业务单独准备。

### 3.2 llama-funasr-cpp

本仓库提供 RISC-V OHOS 交叉编译脚本：

```bash
scripts/build-riscv64-spacemit-ohos.sh
```

脚本依赖：

- SpaceMIT RISC-V OHOS toolchain；
- SpaceMIT llama.cpp checkout；
- 对应的 `riscv64-spacemit-ohos.cmake` toolchain 文件。

构建产物位于：

```text
build-riscv64-ohos/package/bin/
```

模型文件不存储在 GitHub 仓库中，需要根据各 Runtime README 下载或转换 GGUF 模型。

## 4. K3 上效果较优的 ASR 模型

### 4.1 SenseVoice-small

SenseVoice-small 参数量约为 **234M**，采用非自回归端到端架构。本方案使用 **INT8 ONNX** 量化模型，通过 sherpa-onnx 推理。在 K3 通用 CPU 上速度较快，适合对实时性、资源占用和工程稳定性要求较高的场景。

### 4.2 Fun-ASR-Nano

Fun-ASR-Nano 参数量约为 **800M**，采用 Audio Encoder + Adaptor + LLM 的架构，其中包含约 0.6B 参数的 Qwen3 LLM。当前 GGUF 部署方案中，Encoder/Adaptor 使用 F16 权重，LLM 可使用 Q4_K_M 或 Q5_K_M 量化。

相比 SenseVoice-small，Fun-ASR-Nano 模型更大、解码更复杂，但在上下文理解、专业名词、同音词和复杂文本场景下具有更高的识别准确率。

### 4.3 性能对比

| 对比项 | SenseVoice-small | Fun-ASR-Nano |
|---|---:|---:|
| 参数量 | 约 234M | 约 800M |
| 推理框架 | sherpa-onnx | llama-funasr-cpp |
| 模型格式 | ONNX | GGUF |
| 量化方式 | INT8 | Encoder/Adaptor F16 + LLM Q4_K_M/Q5_K_M |
| K3 计算单元 | X100 通用 CPU | AI CPU |
| 当前测试核数 | 4 核 | 8 核 |
| RTF | 约 0.07 | 约 0.25～0.3 |
| 实时性能 | 较高 | 中等 |
| 识别准确率 | 较高 | 更高 |
| 上下文理解 | 一般 | 较强 |
| 专业名词识别 | 较好 | 更好 |
| 标点 / ITN | 模型输出较完整 | 支持较完整文本输出 |
| VAD 集成 | sherpa-onnx 原生支持 | FSMN-VAD |

### 4.4 AISHELL-1 准确率测试

AISHELL-1 test 包含 7176 条普通话音频，累积音频时长约 10 小时。

识别正确率定义为：`1 - CER = 1 -（替换数 + 删除数 + 插入数）/ 参考字符数`。

| 测试指标 | Fun-ASR-Nano | SenseVoice-small |
|---|---:|---:|
| 替换数 | **1,621** | 2,892 |
| 删除数 | 119 | **82** |
| 插入数 | **110** | 124 |
| 总错误数 | **1,850** | 3,098 |
| 识别正确率 | **98.23%** | **97.04%** |

## 5. 命令行使用

## 5.1 sherpa-onnx使用sensevoice指令

SenseVoice：
```bash
cd /path/to/sherpa-onnx/
./bin/sherpa-onnx-vad-with-offline-asr \
  --silero-vad-model=silero_vad.onnx \
  --sense-voice-model=./asr/sensevoice/model_quant_optimized.onnx \
  --tokens=./asr/sensevoice/tokens.sherpa.txt \
  --sense-voice-use-itn=true \
  --num-threads=4 \
  [audio_file.wav]
```

## 5.2 llama-funasr使用funasr-nano指令

在执行llama.cpp推理框架之前，先执行：
mount -o rw,remount /
spacemit-tcm-smi -c

Fun-ASR-Nano CLI：
```bash
cd /path/to/funasr
./bin/llama-funasr-cli \
  --enc ./fun-asr-nano/funasr-encoder-f16.gguf \
  -m ./fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --vad fsmn-vad.gguf
  -a test.wav \
  --output text
```

Fun-ASR-Nano 流式识别：

```bash
arecord -q \
  -D plughw:1,0 \
  -t raw \
  -f S16_LE \
  -c 1 \
  -r 16000 | \
./bin/llama-funasr-stream \
  --enc fun-asr-nano/funasr-encoder-f16.gguf \
  -m fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --stdin-s16le \
  --vad fsmn-vad.gguf
  --output text
```


## 6. 相关文档

- [FunASR C++ RISC-V OHOS 编译与部署步骤](funasr-cpp/BUILD_RISCV64_OHOS.md)
- [sherpa-onnx K3 OpenHarmony 编译指南](sherpa-onnx/BUILD_K3_OHOS.md)
- [Fun-ASR-Nano Runtime](funasr-cpp/runtime/llama.cpp/fun-asr-nano/README.md)
- [流式识别说明](funasr-cpp/runtime/llama.cpp/fun-asr-nano/funasr-stream/README.md)
- [Runtime 设计说明](funasr-cpp/runtime/llama.cpp/DESIGN.md)

## 7. 参考资料
- sherpa-onnx：https://github.com/k2-fsa/sherpa-onnx
- funasr.cpp: https://www.funasr.com/llama-cpp.html
- K3 鸿蒙交叉编译工具链oh-20260630： https://www.funasr.com/llama-cpp.html
- spacemit-onnxruntime: https://github.com/spacemit-com/onnxruntime/releases/tag/2.0.6
- sensevoice ONNX模型权重: https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz
- funasr-nano GGUF模型权重：https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF
- 
