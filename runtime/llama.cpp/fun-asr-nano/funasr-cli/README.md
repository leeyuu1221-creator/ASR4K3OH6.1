# `llama-funasr-cli` 使用教程

`llama-funasr-cli` 是 Fun-ASR-Nano 的 C++ 离线识别程序。它加载 Fun-ASR-Nano
音频 Encoder/Adaptor 和 Qwen3 LLM，将 WAV 音频转换为文本。程序运行时只依赖
GGUF 模型和 llama.cpp/ggml，不需要 Python 或 PyTorch。

> 本程序是离线识别 CLI。需要流式输入、`partial/final` 事件时，请使用同目录
> 上级的 [`funasr-stream`](../funasr-stream/README.md)。

## 1. 工作流程

```text
16 kHz mono WAV
        ↓
80-mel Fbank + LFR(7/6)
        ↓
SAN-M Encoder + Adaptor
        ↓
音频 embedding
        ↓
Qwen3-0.6B LLM
        ↓
转写文本或 JSONL
```

长音频建议使用固定窗口识别，例如 `--chunk 15`。每个窗口会独立推理并输出一个
结果；`--output jsonl` 可以输出窗口的起止时间、推理耗时和汇总 RTF。

## 2. 编译

### 主机编译

在仓库根目录执行：

```bash
cmake -S runtime/llama.cpp -B build-native \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA=../spacemit-llama.cpp \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-native -j --target llama-funasr-cli
```

产物位于：

```text
build-native/bin/llama-funasr-cli
```

### RISC-V OHOS 编译

准备 SpaceMIT RISC-V OHOS 工具链和 SpaceMIT llama.cpp 后，在仓库根目录执行：

```bash
scripts/build-riscv64-spacemit-ohos.sh
```

板端版本位于：

```text
build-riscv64-ohos/package/bin/llama-funasr-cli
```

完整的交叉编译工具和步骤见仓库根目录的
[`BUILD_RISCV64_OHOS.md`](../../../../BUILD_RISCV64_OHOS.md)。

## 3. 模型文件

CLI 需要两个模型：

| 参数 | 模型 | 作用 |
|---|---|---|
| `--enc` | `funasr-encoder-f16.gguf` | 音频 Encoder 和 Adaptor |
| `-m` | `qwen3-0.6b-q4km.gguf` | Qwen3 语言模型 |

如果启用 VAD，还需要：

```text
fsmn-vad.gguf
```

模型不会随源码自动下载。GGUF 模型可以从 FunASR Runtime 对应模型仓库下载，
也可以参考 [`fun-asr-nano/README.md`](../README.md) 中的转换说明生成。

## 4. 命令行格式

```text
llama-funasr-cli \
  --enc <encoder.gguf> \
  -m <qwen3.gguf> \
  -a <audio.wav> \
  [--chunk <seconds>] [--segment-ms <milliseconds>] \
  [--overlap-ms <milliseconds>] [--vad <fsmn-vad.gguf>] \
  [--language auto|zh|en|ja] [-n <tokens>] [-t <threads>] \
  [--rep <float>] [--output text|jsonl]
```

必需参数：

- `--enc`：Fun-ASR-Nano Encoder GGUF；
- `-m` 或 `--model`：Qwen3 LLM GGUF；
- `-a` 或 `--audio`：输入音频文件。

常用参数：

| 参数 | 说明 | 默认值 |
|---|---|---:|
| `--chunk N` | 固定窗口长度，单位秒 | 整个文件 |
| `--segment-ms N` | 固定窗口长度，单位毫秒 | 不设置 |
| `--overlap-ms N` | 相邻窗口重叠时间 | 0 |
| `--vad PATH` | 使用 FSMN-VAD 分段 | 不启用 |
| `--vad-maxseg N` | VAD 最大语音段长度，单位毫秒 | 20000 |
| `--vad-threshold F` | VAD 语音阈值 | 0.5 |
| `--language LANG` | 语言提示 | `auto` |
| `-n N` | 最大生成 token 数 | 512 |
| `-t N` | Encoder 和 LLM 线程数 | 8 |
| `--rep F` | 重复惩罚系数 | 1.05 |
| `--output FORMAT` | `text` 或 `jsonl` | `text` |

查看完整帮助：

```bash
build-native/bin/llama-funasr-cli --help
```

## 5. 基本使用

### 普通离线识别

音频建议为 16 kHz、单声道、PCM16 WAV：

```bash
build-native/bin/llama-funasr-cli \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  -a audio.wav
```

### 长音频分段识别

使用 15 秒窗口：

```bash
build-native/bin/llama-funasr-cli \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  -a audio.wav \
  --chunk 15
```

也可以使用毫秒参数，并设置 1 秒重叠：

```bash
build-native/bin/llama-funasr-cli \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  -a audio.wav \
  --segment-ms 15000 \
  --overlap-ms 1000
```

### 启用 VAD

```bash
build-native/bin/llama-funasr-cli \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  -a audio.wav \
  --vad fsmn-vad.gguf \
  --vad-maxseg 20000
```

启用 VAD 后，程序会先检测语音区间，再将语音段送入 Nano 识别。VAD 模型与
Encoder 模型不同，需要单独提供。

### 输出 JSONL

```bash
build-native/bin/llama-funasr-cli \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  -a audio.wav \
  --chunk 15 \
  --output jsonl
```

输出包含：

- `final`：每个音频窗口的识别结果；
- `utterance_id`：窗口编号；
- `begin_ms`、`end_ms`：音频时间范围；
- `inference_ms`：推理耗时；
- `done`：全部窗口完成后的总耗时和 RTF。

示例：

```json
{"type":"final","utterance_id":1,"revision":1,"begin_ms":0,"end_ms":5592,"text":"开放时间早上九点至下午五点。","inference_ms":2083}
{"type":"done","windows":1,"audio_ms":5592,"total_processing_ms":2083,"total_inference_ms":2083,"rtf":0.372616}
```

## 6. RISC-V OHOS 板端使用

使用 `hdc` 推送二进制后，在板端执行命令前先运行：

```bash
mount -o rw,remount /
spacemit-tcm-smi -c
export SPACEMIT_DISABLE_TCM=1
```

然后执行：

```bash
/data/data/funasr/bin/llama-funasr-cli \
  --enc /data/data/funasr/fun-asr-nano/funasr-encoder-f16.gguf \
  -m /data/data/funasr/fun-asr-nano/qwen3-0.6b-q4km.gguf \
  -a /data/data/funasr/test.wav \
  --output jsonl
```

如果使用 VAD：

```bash
/data/data/funasr/bin/llama-funasr-cli \
  --enc /data/data/funasr/fun-asr-nano/funasr-encoder-f16.gguf \
  -m /data/data/funasr/fun-asr-nano/qwen3-0.6b-q4km.gguf \
  -a /data/data/funasr/test.wav \
  --vad /data/data/funasr/fsmn-vad.gguf \
  --output jsonl
```

## 7. 常见问题

### 模型加载失败

确认 `--enc` 和 `-m` 路径正确，并确认两个文件是 Fun-ASR-Nano 对应的 GGUF，
不能将 SenseVoice 或 Paraformer GGUF 当作 Nano 模型使用。

### 长音频结果重复或异常

不要将很长音频作为一个窗口直接识别，建议使用：

```bash
--chunk 15
```

### 板端启动崩溃

先执行：

```bash
mount -o rw,remount /
spacemit-tcm-smi -c
export SPACEMIT_DISABLE_TCM=1
```

并确认二进制是 `ELF64 RISC-V`，且已经执行：

```bash
chmod 755 /data/data/funasr/bin/llama-funasr-cli
```
