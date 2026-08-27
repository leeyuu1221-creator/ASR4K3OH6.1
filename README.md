# ASR4K3OH6.1

面向 **SpaceMIT K3 + OpenHarmony 6.1** 的本地 ASR 部署仓库。

本仓库主要整理两条本地语音识别部署路线：

* **sherpa-onnx**：基于 ONNX Runtime
* **llama-funasr-cpp**：基于 llama.cpp / ggml

目标设备无需安装 Python 或 PyTorch，可直接运行 C/C++ 推理程序。

---

## 0. 仓库结构

```text
ASR4K3OH6.1/
├── README.md
├── funasr-cpp/
│   ├── runtime/llama.cpp/
│   │   ├── fun-asr-nano/
│   │   ├── sensevoice/
│   │   ├── paraformer/
│   │   ├── funasr-vad/
│   │   ├── funasr-vad-stream/
│   │   ├── funasr-common/
│   │   ├── tests/
│   │   └── CMakeLists.txt
│   ├── scripts/
│   ├── build-riscv64-ohos/package/
│   └── BUILD_RISCV64_OHOS.md
└── sherpa-onnx/
    ├── source/
    ├── dependencies/spacemit-ort/
    ├── dependencies/alsa/
    ├── toolchain/
    └── BUILD_K3_OHOS.md
```

### 当前包含的板端目标

仓库当前保留 8 个 Runtime 目标：

* `cli`
* `stream`
* `encoder`
* `embd`
* `sensevoice`
* `paraformer`
* `vad`
* `vad-stream`

以下组件暂不在当前部署范围：

* CAM++
* SenseVoice IME2
* CLI Bench

> 本仓库主要用于 **RISC-V OpenHarmony 端侧部署**，不包含完整的 Python 模型训练工程。

---

# 1. 方案概述

目前在 **SpaceMIT K3 + OpenHarmony 6.1** 上重点验证了两套 ASR 推理方案。

| 方案               | 推理框架             | 推荐模型             | 主要计算单元      |
| ---------------- | ---------------- | ---------------- | ----------- |
| sherpa-onnx      | ONNX Runtime     | SenseVoice-small | X100 通用 CPU |
| llama-funasr-cpp | llama.cpp / ggml | Fun-ASR-Nano     | AI CPU      |

## 推荐配置

### SenseVoice-small

推荐：

```text
sherpa-onnx
+
SenseVoice-small INT8 ONNX
+
4 核通用 CPU
```

当前实测：

* RTF：约 **0.07**
* 推理速度快
* 资源占用较低
* 识别稳定
* 模型自带标点
* 支持 ITN

适合：

* 实时语音识别
* 长时间连续运行
* 对资源占用敏感的场景
* 对工程稳定性要求较高的产品

---

### Fun-ASR-Nano

推荐：

```text
llama-funasr-cpp
+
Fun-ASR-Nano
+
8 核 AI CPU
```

当前实测：

* RTF：约 **0.25～0.3**
* 上下文理解能力较强
* 专业名词识别效果更好
* 同音词判断能力更强
* 复杂文本场景准确率较高

适合：

* 专业领域语音识别
* 对上下文理解要求较高的场景
* 对识别准确率优先级高于推理速度的场景

---

# 2. 推理框架对比

## 2.1 sherpa-onnx

`sherpa-onnx` 是基于 **ONNX Runtime** 的跨平台语音推理框架。

主要覆盖：

* ASR
* VAD
* 关键词检测
* 说话人处理
* TTS
* 其他语音任务

K3/OpenHarmony 相关资料位于：

```text
sherpa-onnx/
```

其中包含：

```text
source/                     sherpa-onnx 源码
dependencies/spacemit-ort/ SpaceMIT ONNX Runtime / EP
dependencies/alsa/         可选 ALSA 依赖
toolchain/                  交叉编译配置
BUILD_K3_OHOS.md            编译与部署说明
```

### K3 部署流程

1. 准备 sherpa-onnx 源码；
2. 准备 SpaceMIT ONNX Runtime / EP；
3. 准备 K3 OpenHarmony 交叉编译工具链；
4. 按 `BUILD_K3_OHOS.md` 完成交叉编译；
5. 选择通用 CPU 或 SpaceMIT EP；
6. 部署模型、动态库和程序到 K3；
7. 根据模型测试线程数和推理后端。

### 优点

* 模型生态成熟
* ASR 模型覆盖范围广
* 接口稳定
* VAD 等语音能力完整
* 更换模型时上层业务改动较小

### 注意

使用 sherpa-onnx 时，板端需要同时部署对应的：

```text
ONNX Runtime 动态库
```

当前实测中，部分模型使用 **SpaceMIT AI CPU / EP** 后并不一定比通用 CPU 更快。

---

## 2.2 llama-funasr-cpp

`llama-funasr-cpp` 基于：

* llama.cpp
* ggml
* FunASR Runtime

主要用于 FunASR 模型的本地 C/C++ 推理。

当前支持：

* Fun-ASR-Nano
* SenseVoice
* Paraformer
* FSMN-VAD

模型主要使用针对当前 Runtime 导出的 **GGUF 权重**。

### 主要特点

* Runtime 较轻量
* 适合端侧部署
* GGUF 量化方便
* 对 Fun-ASR-Nano 支持较有针对性
* 可使用 K3 AI CPU 相关优化

---

### Fun-ASR-Nano 流式能力

当前仓库在离线识别基础上增加了：

```text
Fun-ASR-Nano
+
FSMN-VAD
+
分段式流式识别
```

数据流程：

```text
PCM 输入
   ↓
FSMN-VAD
   ↓
语音分段
   ↓
Fun-ASR-Nano
   ↓
partial / final 输出
```

需要注意：

> 当前“流式识别”指持续接收 PCM，并通过 VAD 分段后输出 `partial/final` 事件。

模型本身仍会针对音频窗口重新执行推理，因此：

```text
分段式流式识别 ≠ 严格的 causal streaming encoder
```

---

## 2.3 框架能力对比

| 对比项        | sherpa-onnx             | llama-funasr-cpp          |
| ---------- | ----------------------- | ------------------------- |
| 框架定位       | 通用 ONNX 语音推理框架          | FunASR C/C++ 边缘推理 Runtime |
| 模型格式       | ONNX                    | GGUF                      |
| 推理 Runtime | ONNX Runtime            | ggml / llama.cpp          |
| K3 运行方式    | 通用 CPU / SpaceMIT EP    | AI CPU 相关优化               |
| 模型覆盖       | 广                       | 主要为 FunASR                |
| VAD        | 支持                      | FSMN-VAD                  |
| 离线 ASR     | 支持                      | 支持                        |
| 流式 ASR     | 支持                      | 分段式流式输出                   |
| 量化         | INT8 等                  | Q4 / Q5 / Q8 / F16        |
| 功能完整度      | 高                       | 聚焦 ASR / VAD              |
| 主要优势       | 成熟、模型丰富、接口稳定            | Runtime 轻量、量化友好           |
| 主要不足       | 部分模型使用 SpaceMIT EP 收益有限 | 模型覆盖和通用接口较少               |

---

## 2.4 关于 AI CPU / NPU 加速

需要特别注意：

> **AI CPU / NPU 并不意味着一定比通用 CPU 更快。**

实际性能取决于：

* 模型结构
* 算子支持情况
* 数据搬运开销
* Execution Provider 实现
* 线程数量
* 模型量化方式
* 内存带宽
* 输入音频长度

因此建议分别测试：

```text
通用 CPU
AI CPU
SpaceMIT EP
不同线程数
不同量化精度
```

最终以板端实测结果为准。

---

# 3. K3 部署方式

## 3.1 sherpa-onnx

K3/OpenHarmony 相关资料位于：

```text
sherpa-onnx/
```

包含：

* sherpa-onnx 源码
* SpaceMIT ONNX Runtime / EP
* 可选 ALSA 依赖
* toolchain 配置
* 编译文档

完整交叉编译工具链由于体积较大未提交仓库。

获取方式见：

```text
sherpa-onnx/BUILD_K3_OHOS.md
```

模型和最终业务程序需根据实际项目单独准备。

---

## 3.2 llama-funasr-cpp

仓库提供 RISC-V OpenHarmony 交叉编译脚本：

```bash
scripts/build-riscv64-spacemit-ohos.sh
```

### 编译依赖

需要准备：

* SpaceMIT RISC-V OHOS toolchain
* SpaceMIT llama.cpp checkout
* `riscv64-spacemit-ohos.cmake`

### 编译产物

默认位于：

```text
build-riscv64-ohos/package/bin/
```

### 模型权重

GGUF 模型文件不存储在本 GitHub 仓库中。

请根据各 Runtime README：

* 下载官方 GGUF 模型；
* 或自行转换模型。

---

# 4. K3 上效果较优的 ASR 模型

## 4.1 SenseVoice-small

SenseVoice-small 参数量约：

```text
234M
```

架构：

```text
非自回归端到端 ASR
```

当前 K3 方案使用：

```text
SenseVoice-small
+
INT8 ONNX
+
sherpa-onnx
+
4 核通用 CPU
```

### 特点

* 推理速度快
* RTF 较低
* 资源占用较低
* 输出稳定
* 标点较完整
* 支持 ITN
* 适合长时间运行

当前实测 RTF：

```text
约 0.07
```

---

## 4.2 Fun-ASR-Nano

Fun-ASR-Nano 参数量约：

```text
800M
```

基本架构：

```text
Audio Encoder
    ↓
Adaptor
    ↓
Qwen3 LLM
```

其中包含约：

```text
0.6B Qwen3 LLM
```

当前 GGUF 部署配置：

| 模块      | 权重              |
| ------- | --------------- |
| Encoder | F16             |
| Adaptor | F16             |
| LLM     | Q4_K_M / Q5_K_M |

### 优势

相比 SenseVoice-small，Fun-ASR-Nano 在以下场景表现更好：

* 上下文理解
* 专业名词
* 同音词
* 长句识别
* 复杂文本
* 语言模型相关纠错

代价是：

* 模型更大
* 解码链路更复杂
* 推理速度更慢
* 内存占用更高

---

### 4.2.1 主要缺陷：LLM 幻觉

Fun-ASR-Nano 属于 **基于 LLM 的生成式 ASR 模型**。

与 SenseVoice-small 等传统端到端 ASR 相比，其语言生成能力更强，同时也带来了更加明显的 **Hallucination（幻觉）风险**。

以下情况尤其容易出现问题：

* 长时间静音
* 持续背景噪声
* 低信噪比音频
* 非语音声音
* 语音内容模糊
* VAD 切分不准确
* 上下文存在较强语言提示

模型可能受到：

```text
历史上下文
+
LLM 语言先验
```

影响，从而生成音频中实际不存在的词句。

典型现象：

```text
没有人说话
    ↓
持续存在环境噪声
    ↓
模型仍输出完整语句
```

因此，在以下场景中需要重点处理：

* 长时间连续监听
* 无人值守 ASR
* 会议设备
* 语音助手
* 误触发敏感系统

建议结合：

```text
VAD
+
静音过滤
+
最小时长限制
+
置信度判定
+
输出过滤
+
上下文重置
```

降低幻觉输出的概率。

---

## 4.3 性能对比

| 对比项      | SenseVoice-small |                            Fun-ASR-Nano |
| -------- | ---------------: | --------------------------------------: |
| 参数量      |           约 234M |                                  约 800M |
| 推理框架     |      sherpa-onnx |                        llama-funasr-cpp |
| 模型格式     |             ONNX |                                    GGUF |
| 量化方式     |             INT8 | Encoder/Adaptor F16 + LLM Q4_K_M/Q5_K_M |
| K3 计算单元  |      X100 通用 CPU |                                  AI CPU |
| 测试核数     |              4 核 |                                     8 核 |
| RTF      |       **约 0.07** |                          **约 0.25～0.3** |
| 实时性能     |                高 |                                      中等 |
| 识别准确率    |               较高 |                                      更高 |
| 上下文理解    |               一般 |                                      较强 |
| 专业名词     |               较好 |                                      更好 |
| 标点 / ITN |              较完整 |                                     较完整 |
| VAD      | sherpa-onnx 原生支持 |                                FSMN-VAD |
| 幻觉风险     |               较低 |                                  **较高** |

### 简单选型建议

如果优先考虑：

```text
速度
稳定性
低资源占用
长时间运行
```

推荐：

```text
SenseVoice-small
```

如果优先考虑：

```text
准确率
上下文理解
专业名词
复杂文本
```

推荐：

```text
Fun-ASR-Nano
```

---

# 5. AISHELL-1 准确率测试

AISHELL-1 `test` 数据集包含：

```text
7176 条普通话音频
约 10 小时音频
```

识别正确率计算方式：

```text
识别正确率 = 1 - CER
```

其中：

```text
CER =（替换数 + 删除数 + 插入数）/ 参考字符数
```

## 测试结果

| 指标    | Fun-ASR-Nano | SenseVoice-small |
| ----- | -----------: | ---------------: |
| 替换数   |    **1,621** |            2,892 |
| 删除数   |          119 |           **82** |
| 插入数   |      **110** |              124 |
| 总错误数  |    **1,850** |            3,098 |
| 识别正确率 |   **98.23%** |       **97.04%** |

从当前测试结果看：

```text
Fun-ASR-Nano
98.23%

SenseVoice-small
97.04%
```

Fun-ASR-Nano 的整体字符识别准确率更高，但其部署成本、推理速度和幻觉风险也明显高于 SenseVoice-small。

---

# 6. 命令行使用

## 6.1 sherpa-onnx + SenseVoice

进入 sherpa-onnx 目录：

```bash
cd /path/to/sherpa-onnx/
```

执行：

```bash
./bin/sherpa-onnx-vad-with-offline-asr \
  --silero-vad-model=silero_vad.onnx \
  --sense-voice-model=./asr/sensevoice/model_quant_optimized.onnx \
  --tokens=./asr/sensevoice/tokens.sherpa.txt \
  --sense-voice-use-itn=true \
  --num-threads=4 \
  [audio_file.wav]
```

当前推荐：

```text
SenseVoice-small INT8
+
4 threads
+
通用 CPU
```

---

## 6.2 llama-funasr-cpp + Fun-ASR-Nano

执行 llama.cpp Runtime 前：

```bash
mount -o rw,remount /
spacemit-tcm-smi -c
```

---

### 6.2.1 离线识别

```bash
cd /path/to/funasr
```

执行：

```bash
./bin/llama-funasr-cli \
  --enc ./fun-asr-nano/funasr-encoder-f16.gguf \
  -m ./fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --vad fsmn-vad.gguf \
  -a test.wav \
  --output text
```

---

### 6.2.2 流式识别

从 ALSA 持续读取：

```text
16 kHz
16-bit PCM
Mono
```

执行：

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
  --vad fsmn-vad.gguf \
  --output text
```

处理链路：

```text
麦克风
  ↓
arecord
  ↓
16 kHz / S16_LE / Mono PCM
  ↓
FSMN-VAD
  ↓
Fun-ASR-Nano
  ↓
partial / final
```

---

# 7. 方案选择

## 优先实时性

推荐：

```text
sherpa-onnx
+
SenseVoice-small INT8
```

适用于：

* 实时字幕
* 语音命令
* 智能终端
* 长时间监听
* 低功耗设备

---

## 优先准确率

推荐：

```text
llama-funasr-cpp
+
Fun-ASR-Nano
```

适用于：

* 专业词汇识别
* 复杂语境
* 会议转写
* 文本上下文要求较高的场景

但需要额外处理：

```text
VAD
幻觉
静音
噪声
长时间上下文
```

---

# 8. 相关文档

* [FunASR C++ RISC-V OHOS 编译与部署步骤](funasr-cpp/BUILD_RISCV64_OHOS.md)
* [sherpa-onnx K3 OpenHarmony 编译指南](sherpa-onnx/BUILD_K3_OHOS.md)
* [Fun-ASR-Nano Runtime](funasr-cpp/runtime/llama.cpp/fun-asr-nano/README.md)
* [流式识别说明](funasr-cpp/runtime/llama.cpp/fun-asr-nano/funasr-stream/README.md)
* [Runtime 设计说明](funasr-cpp/runtime/llama.cpp/DESIGN.md)

---

# 9. 参考资料

* sherpa-onnx
  https://github.com/k2-fsa/sherpa-onnx

* FunASR llama.cpp / funasr.cpp
  https://www.funasr.com/llama-cpp.html

* SpaceMIT K3 OpenHarmony 交叉编译工具链 `oh-20260630`

* SpaceMIT ONNX Runtime
  https://github.com/spacemit-com/onnxruntime/releases/tag/2.0.6

* SenseVoice ONNX 模型
  https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz

* Fun-ASR-Nano GGUF
  https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF

* ASR 学习资料
  https://zsc.github.io/asr_sd_tutorial/html/index.html
