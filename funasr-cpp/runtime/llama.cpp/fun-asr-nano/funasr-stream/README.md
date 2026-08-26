# `llama-funasr-stream` 使用教程

`llama-funasr-stream` 是 Fun-ASR-Nano 的流式识别程序。它会在进程
启动时加载一次 SAN-M 音频编码器和 Qwen3 语言模型，然后持续接收音频，按窗口
生成中间识别结果（`partial`）和最终识别结果（`final`）。

本程序提供的是“流式输入 + 增量输出”，不是严格的因果编码器缓存：Nano 的
SAN-M 编码器是双向编码器，因此每次 `partial` 或 `final` 都会对当前音频窗口
重新计算一遍假设。程序支持默认固定窗口模式，以及使用 FSMN-VAD 检测语音起止的
VAD 模式；两种模式都通过有界任务队列控制延迟与内存使用。

## 1. 工作方式

音频会按 `--input-block-ms` 指定的大小进入当前 utterance（默认 100 ms）。
当累计音频达到 `--partial-interval-ms` 的时间门限（默认 10000 ms）时，程序
提交一次 `partial` 推理任务。

默认固定窗口模式下，当当前 utterance 达到 `--segment-ms`（默认 30000 ms）时，
程序提交一个 `final`，并创建下一个 utterance。如果设置了 `--overlap-ms`，前一
个窗口的末尾音频会复制到下一个窗口的开头，以降低切段时截断词语的风险。

使用 `--vad` 后，程序先由 FSMN-VAD 检测整段音频中的语音区间，只将语音区间送入
识别。静音不会触发 partial 或 final；一个 VAD 区间结束时提交
`reason: "vad"` 的 final，因此不会在普通静音处按固定时长硬切。`--vad-maxseg`
是 VAD 的单段最大长度保护，超长连续语音仍可能在该上限处闭合。

`--audio` 文件输入继续使用原有批处理 VAD：先读取完整文件、完成分段，再按区间
送入 ASR。`--stdin-s16le` 配合 `--vad` 时使用流式 VAD，VAD 的前端残余、FSMN
历史和端点状态跨 PCM block 保留。

相邻 `final` 文本会自动进行简单的前后缀去重。

输入结束时，尚未达到最大段长的尾部窗口会以 `reason: "eof"` 生成一个 `final`。
达到最大段长生成的最终结果，其原因是 `reason: "max_segment"`。

注意：`partial` 是当前窗口的临时假设，同一个 utterance 中后续结果可能修正它；
`final` 才是应用应该保存或提交的稳定结果。

## 2. 构建

请在仓库根目录 `/home/cs/project/K3/FunASR` 执行：

```bash
cmake -S runtime/llama.cpp -B build-native \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA=../spacemit-llama.cpp \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-native -j --target llama-funasr-stream
```

构建完成后，可执行文件位于：

```text
build-native/bin/llama-funasr-stream
```

如果不使用仓库中的本地 `spacemit-llama.cpp`，可以去掉
`-DFETCHCONTENT_SOURCE_DIR_LLAMA=../spacemit-llama.cpp`，让 CMake 使用项目配置
的 llama.cpp 依赖。首次构建可能需要下载依赖。

## 3. 模型文件

程序需要两个 GGUF 文件：

| 参数 | 文件 | 作用 |
| --- | --- | --- |
| `--enc` | Fun-ASR-Nano 编码器 GGUF | 将音频转换为音频 embedding |
| `-m` | Qwen3 语言模型 GGUF | 根据音频 embedding 生成转写文本 |

例如：

```text
funasr-encoder-f16.gguf
qwen3-0.6b-q4km.gguf
```

模型路径可以是绝对路径，也可以是相对于当前工作目录的路径。程序启动时会
先加载编码器和语言模型，加载成功后输出 `ready` 事件；在看到 `ready` 之前，
上游程序不应假设模型已经可以处理音频。

当前实现通过 llama.cpp 的 CPU 路径执行 LLM 推理，并将上下文、批处理和微批处理
上限固定为 2048。`-t` 参数同时控制编码器、LLM generation、LLM batch，以及
VAD（启用时）的线程数，默认值为 `4`。

## 4. 命令行格式

完整格式如下：

```text
llama-funasr-stream --enc encoder.gguf -m qwen3.gguf \
  (--audio file | --stdin-s16le) \
  [--language auto|zh|en|ja] \
  [--vad fsmn-vad.gguf [--vad-maxseg N] [--vad-merge-gap N] \
    [--vad-threshold F] [--vad-chunk-ms N] \
    [--vad-max-end-silence-ms N]] \
  [--realtime] [--input-block-ms N] [--partial-interval-ms N] \
  [--segment-ms N] [--overlap-ms N] [-n N] [-t N] [--rep F] \
  [--output jsonl|text]
```

`--audio` 和 `--stdin-s16le` 必须二选一，不能同时使用。两种输入都支持 `--vad`，
但两者使用不同的 VAD 路径。所有带数值的时间参数单位都是毫秒。

### 必选参数

#### `--enc encoder.gguf`

指定 Fun-ASR-Nano 音频编码器模型。

#### `-m qwen3.gguf`

指定 Qwen3 语言模型。

#### `--audio file`

从音频文件读取输入。该路径由项目内置音频读取器处理，并转换为 16 kHz、单声道
浮点 PCM；可以使用项目音频读取器支持的常见音频文件格式。

#### `--stdin-s16le`

从标准输入读取原始 PCM。输入格式必须严格为：

- 采样率：16 kHz
- 声道数：1（单声道）
- 采样格式：有符号 16 位整数
- 字节序：little-endian
- 输入内容：不带 WAV 文件头的裸 PCM 字节流

程序会将每两个字节转换为一个浮点采样点，并归一化到约 `[-1, 1)`。如果输入
最后只剩一个字节，该不完整采样会被忽略并在标准错误中提示。

### 语言参数

#### `--language auto|zh|en|ja`

设置语言提示，默认是 `auto`：

- `auto`：使用原有的“语音转写”提示，由模型自动判断语言；
- `zh`：提示模型转写为中文；
- `en`：提示模型转写为英文；
- `ja`：提示模型转写为日文。

例如，限制为中文识别：

```bash
build-native/bin/llama-funasr-stream \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  --language zh \
  --audio sample.wav
```

该参数通过修改 LLM prompt 提供语言提示，并不是强制性的 token 级语言约束；
模型仍可能在音频质量较差或包含其他语言时输出混合语言。未指定时保持原有行为。

### 输入和切分参数

#### `--vad fsmn-vad.gguf`

启用 FSMN-VAD，并指定 VAD 模型 GGUF。当前仅支持 stdin 输入的流式 VAD，
`ready.mode` 为 `streaming-vad`。
VAD 区间结束时的 `final` 事件带有 `reason: "vad"`。

VAD 模型不是 `--enc` 编码器模型，必须单独提供。流式模式使用持久化的
VAD 前端、FSMN 历史和端点状态。

#### `--vad-maxseg N`

设置 FSMN-VAD 单个语音区间的最大长度，默认是 `12000` ms。连续语音达到该长度
时，VAD 会闭合当前区间并开始新的区间。该参数用于限制单次识别上下文和内存，必须
大于 0。

#### `--vad-merge-gap N`

设置文件批量 VAD 相邻语音区间的最大合并间隔，默认 `500` ms。两个区间之间的
静音不超过该值时，会合并后再提交 ASR；设置为 `0` 可关闭合并。流式 stdin VAD
不使用该参数。

#### `--vad-threshold F`

设置 VAD 的 speech/noise 判定阈值，默认值为 `0.5`，取值范围为 `0.0` 到 `1.0`。
实际判定逻辑为：

```text
speech_prob >= noise_prob + speech_noise_thres
```

阈值越大，语音判定越严格，可能减少误检但漏掉更多弱语音；阈值越小，语音判定
越宽松，可能检出更多弱语音但增加噪声误检。例如：

```bash
--vad fsmn-vad.gguf \
--vad-threshold 0.3
```

该参数只影响 VAD 后处理判定，不会修改 VAD GGUF 模型权重。未指定时保持原有
`speech_noise_thres=0.5` 行为。

#### `--vad-chunk-ms N`

设置 stdin 流式 VAD 的内部处理块，默认 `60` ms。任意长度的 PCM 输入先与上次
不足一个块的 `prev_samples` 拼接，再按该大小处理；前端残余、FSMN cache、窗口状态
和绝对时间轴都跨块保留。该参数不影响文件批量 VAD。

#### `--vad-max-end-silence-ms N`

设置固定的流式尾静音阈值。指定正数后固定值优先，并关闭动态尾静音调整；不指定时
默认采用流式 schedule：当前段 `≤5s / ≤10s / ≤15s / ≤30s / ≤45s / >45s`
分别使用 `2000 / 1500 / 1000 / 800 / 400 / 100` ms 的尾静音阈值。短语音允许
较长停顿，当前语音段越长，切分越积极。该参数不影响文件批量 VAD。

#### `--input-block-ms N`

设置每次消费的基础音频块大小，默认是 `100`。它影响流式循环的时间粒度：

- 数值较小：时间响应更细，但循环和任务调度次数更多；
- 数值较大：调度开销较低，但 partial 触发时间粒度更粗。

该参数不会让程序在每个 block 都执行推理，实际 partial 仍由
`--partial-interval-ms` 控制。

#### `--partial-interval-ms N`

设置两次 partial 提交之间的最小音频时间间隔，默认是 `10000`。partial 只在
音频时间轴达到门限后提交；它不是严格的墙上时钟间隔。

例如，设置为 `1000` 表示大约每累计 1 秒音频提交一次当前窗口识别；设置为
`3000` 则大约每 3 秒提交一次。

#### `--segment-ms N`

在默认固定窗口模式下，设置单个 utterance 的最大窗口长度，默认是 `30000`。
达到该长度后会生成 `reason: "max_segment"` 的 final，然后开始下一个 utterance。

在 VAD 模式下，语音起止由 VAD 决定，`--segment-ms` 不用于普通语音段切分；
VAD 的单段上限由 `--vad-maxseg` 控制。

该值必须大于 `--overlap-ms`。由于输入按 block 追加，实际 `end_ms` 可能略微
超过设定值，超过量通常不大于一个输入 block。

#### `--overlap-ms N`

固定窗口模式下，设置相邻 utterance 之间的音频重叠长度，默认是 `1000`，必须满足：

```text
0 <= overlap_ms < segment_ms
```

重叠有助于避免词语在窗口边界被截断，但会增加后续窗口的推理音频量。程序会
对相邻 `final` 文本做简单去重；如果模型在重叠区域产生了不一致的文本，去重
不能保证语义级别的完美合并。

VAD 模式下，`--overlap-ms` 不用于 VAD 端点切分；VAD 已经给出了每个语音区间的
边界。

#### `--realtime`

仅对 `--audio` 文件输入有效。每处理一个输入 block 后，程序会休眠约
`--input-block-ms` 毫秒，使文件回放速度接近实时采集速度。

该参数不能与 `--stdin-s16le` 一起使用，因为标准输入本身已经由上游生产者决定
输入速度。文件回放和实时回放示例见下文。

### 解码和输出参数

#### `-n N`

设置每个窗口最多生成的 token 数，默认是 `512`。达到该上限或模型输出结束标记
（EOG）时停止生成。数值过小可能导致文本被截断；数值过大则可能增加推理时间。

#### `-t N`

设置编码器、LLM 和 VAD 使用的线程数，默认值是 `4`。LLM 的 generation 和
prompt/audio batch 使用相同的线程数。该参数必须大于 0，例如：

```bash
-t 8
```

#### `--rep F`

设置重复惩罚，默认是 `1.01`；`1.0` 表示不施加重复惩罚。只有不等于 `1.0` 时才会将
惩罚采样器加入 llama.cpp 的采样器链；之后仍使用 greedy 解码。

#### `--output jsonl|text`

设置输出格式，默认是 `jsonl`：

- `jsonl`：输出完整的机器可读事件流，包括 `ready`、`partial`、`final` 和 `done`；
- `text`：抑制 `ready`、`partial` 和 `done`，只在产生 `final` 时输出最终文本，
  每条文本带 `mm:ss-mm:ss` 起止时间并占一行。

## 5. 文件识别

最基本的文件识别命令：

```bash
cd data/data/funasr
./bin/llama-funasr-stream \
  --enc  fun-asr-nano/funasr-encoder-f16.gguf \
  -m fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --audio test.wav
```

一个更适合观察中间结果的配置：

```bash
./bin/llama-funasr-stream \
  --enc  fun-asr-nano/funasr-encoder-f16.gguf \
  -m fun-asr-nano/qwen3-0.6b-q4km.gguf \
  --audio test.wav \
  --partial-interval-ms 1000 \
  --segment-ms 12000 \
  --overlap-ms 1000 \
  --output jsonl
```

文件模式默认会尽快读完整个文件并将音频 block 提交给后台推理线程，不会主动
等待实时播放速度。若要模拟麦克风按实时速度输入，使用：

```bash
build-native/bin/llama-funasr-stream \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  --audio sample.wav \
  --realtime \
  --input-block-ms 100 \
  --partial-interval-ms 1000
```

使用 `--output text` 获取纯最终文本：

```bash
build-native/bin/llama-funasr-stream \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  --audio sample.wav \
  --output text
```

## 6. 使用 FSMN-VAD 分段

准备编码器、Qwen3 和 VAD 三个模型后，执行：

```bash
build-native/bin/llama-funasr-stream \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  --vad fsmn-vad.gguf \
  --vad-maxseg 30000 \
  --vad-threshold 0.5 \
  --audio sample.wav \
  --partial-interval-ms 1000 \
  --output jsonl
```

处理流程如下：

文件输入：

1. 读取完整文件并执行批量 VAD；
2. 跳过静音区间，将语音区间按 block 送入 ASR；
3. 每个语音区间结束时提交 `reason: "vad"` 的 `final`。

stdin 输入：

1. 每个 PCM block 进入流式 fbank/LFR/FSMN-VAD；
2. speech-start 时从约 1 秒环形历史中补回句首；
3. 语音期间按 `--partial-interval-ms` 输出 partial；
4. speech-end 时裁掉尾部静音并提交 final；
5. `--vad-maxseg` 触发时强制闭合当前语音段。

`--vad-maxseg` 应根据模型上下文和设备性能设置。较小的值可降低单次推理延迟，
但可能把连续语音拆成多个 final；较大的值能保留更多上下文，但单次推理更慢。

VAD 模式下 `ready` 示例：

```json
{"type":"ready","sample_rate":16000,"mode":"vad-segmented","model_loaded_ms":1234}
```

VAD final 示例：

```json
{"type":"final","utterance_id":1,"revision":4,"begin_ms":820,"end_ms":4680,"text":"你好世界。","inference_ms":1200,"reason":"vad"}
```

## 7. 麦克风和管道输入

Linux 上可以使用 `arecord` 将麦克风转换为程序要求的裸 PCM：

```bash
arecord -q -t raw -f S16_LE -c 1 -r 16000 | \
  build-native/bin/llama-funasr-stream \
    --enc funasr-encoder-f16.gguf \
    -m qwen3-0.6b-q4km.gguf \
    --stdin-s16le \
    --partial-interval-ms 1000 \
    --segment-ms 12000 \
    --overlap-ms 1000
```

也可以从其他采集程序或网络程序接收裸 PCM：

```bash
your-capture-program | \
  build-native/bin/llama-funasr-stream \
    --enc /path/to/funasr-encoder-f16.gguf \
    -m /path/to/qwen3-0.6b-q4km.gguf \
    --stdin-s16le \
    --output jsonl
```

如果上游输出的不是 16 kHz 单声道 S16LE，需要先完成重采样、混音和格式转换，
再通过管道传入。`--stdin-s16le` 不会解析 WAV 头，也不会自动识别采样率或声道数。

## 8. JSONL 输出协议

默认情况下，每个事件严格占一行，方便使用 `jq`、Python 或其他流式程序逐行处理。

### `ready` 事件

模型初始化成功后输出一次：

```json
{"type":"ready","sample_rate":16000,"mode":"fixed-window","model_loaded_ms":1234}
```

字段说明：

- `type`：固定为 `ready`；
- `sample_rate`：固定为 `16000`；
- `mode`：固定为 `fixed-window`；
- `model_loaded_ms`：模型加载和运行时初始化耗时，单位为毫秒。

### `partial` 事件

每次临时识别成功后输出，例如：

```json
{"type":"partial","utterance_id":1,"revision":2,"begin_ms":0,"end_ms":3000,"text":"你好世界","inference_ms":850}
```

字段说明：

- `utterance_id`：当前最终分段编号，从 `1` 开始；
- `revision`：当前 utterance 内的结果版本号，从 `1` 开始递增；
- `begin_ms`、`end_ms`：本次识别窗口在整段输入中的起止时间；
- `text`：当前窗口的临时识别文本；
- `inference_ms`：本次推理耗时，单位为毫秒。

### `final` 事件

达到最大段长、VAD 端点或输入结束时输出，例如：

```json
{"type":"final","utterance_id":1,"revision":3,"begin_ms":0,"end_ms":12000,"text":"你好世界。","inference_ms":2100,"reason":"max_segment"}
```

`final` 比 `partial` 多一个 `reason` 字段：

- `max_segment`：达到 `--segment-ms`；
- `vad`：FSMN-VAD 检测到语音区间结束；
- `eof`：输入结束，提交剩余尾部窗口。

`final` 的 `text` 是程序对前面相邻 final 窗口进行简单重叠去重后的累计文本，
不是只包含当前窗口新增部分的文本。因此应用通常应保存每个 `final` 的最新累计文本，
而不是把所有 `final.text` 直接拼接一次。

### `done` 事件

输入结束且所有排队任务都完成后输出一次：

```json
{"type":"done","utterances":2,"partial_dropped":0,"audio_ms":18500,"total_processing_ms":4300,"total_inference_ms":4200,"rtf":0.232432}
```

字段说明：

- `utterances`：当前计数器的值；
- `partial_dropped`：因推理速度跟不上输入速度而丢弃的排队 partial 数量；
- `audio_ms`：实际读取到的音频时长，单位为毫秒。
- `total_processing_ms`：模型加载完成后的总墙钟处理时间，包括音频读取、VAD、
  队列等待和 ASR；批量 VAD 模型加载时间会扣除；
- `total_inference_ms`：所有成功 ASR 窗口推理耗时之和，单位为毫秒；不包含模型加载、
  音频读取、队列等待和 VAD 耗时；
- `rtf`：实时率，计算公式为
  `total_processing_ms / audio_ms`。小于 `1.0` 表示整体处理速度快于实时。

`total_inference_ms` 会累计实际执行的 partial 和 final 推理。由于 partial 是对
不断增长窗口的重复识别，文件离线模式下 RTF 可能大于只执行一次最终识别时的 RTF。
启用 VAD 时，RTF 的分母仍是完整输入文件时长（包括静音），但分子不包含 VAD 模型
推理耗时。

`done` 只在 `jsonl` 模式输出，`text` 模式不会输出该事件。

在 `--output text` 模式下，统计信息不会污染标准输出，而是写入标准错误：

```text
stream: total_processing_ms=4300 total_inference_ms=4200 rtf=0.232432 audio_ms=18500
```

## 9. 处理延迟和丢帧行为

主线程负责读取音频，后台线程负责模型推理。队列最多保留少量任务：

- 排队中的旧 `partial` 可以被最新的 partial 快照替换；
- 生成 `final` 时，过期的排队 partial 会被删除；
- `final` 不会像 partial 一样静默丢弃，程序会在必要时等待队列空间；
- 正在执行的推理任务不会被中断或替换。

因此，当单次推理时间大于音频输入间隔时，可能看不到每一个计划的 partial，
并且 `partial_dropped` 会大于 0。这是有意设计，用于优先保证实时性和最终结果。

## 10. 参数调优建议

可以从以下配置开始：

| 使用场景 | `input-block-ms` | `partial-interval-ms` | `segment-ms` | `overlap-ms` |
| --- | ---: | ---: | ---: | ---: |
| 低延迟交互 | 50 | 500–1000 | 8000–12000 | 500–1000 |
| 普通实时识别 | 100 | 1000–3000 | 12000 | 1000 |
| 追求吞吐量 | 200 | 3000–5000 | 12000–20000 | 500–1000 |

调参时注意：

1. 减小 `partial-interval-ms` 会增加重复推理次数；
2. 增大 `segment-ms` 会提高单次推理的音频量和延迟；
3. 增大 `overlap-ms` 有助于边界识别，但会重复计算更多音频；
4. 如果 `partial_dropped` 持续增加，应先增大 partial 间隔或缩短输入压力，
   而不是无限增加队列；
5. 音频短于模型所需的最小窗口时不会触发推理。当前实现的最小判断约为
   400 个采样点，即约 25 ms；过短输入可能只输出 `ready` 和 `done`。

## 11. K3 注意事项

在 K3 平台上，如果独立的 TCM 问题尚未解决，请在运行前设置：

```bash
export SPACEMIT_DISABLE_TCM=1
```

然后再执行 `llama-funasr-stream`。该环境变量是否生效取决于底层平台运行时；
它不是 `llama-funasr-stream` 自身解析的命令行参数。

文件 `--audio` 使用批处理 VAD；`--stdin-s16le` 使用流式 VAD。stdin 必须保持
16 kHz、单声道、S16LE 裸 PCM 格式。

## 12. 常见问题

### 启动后直接打印 usage

通常是必选参数缺失、同时指定了 `--audio` 和 `--stdin-s16le`，或数值参数非法。
检查以下约束：

```text
block_ms > 0
partial_ms > 0
segment_ms > 0
0 <= overlap_ms < segment_ms
npred > 0
n_threads > 0
output 必须是 jsonl 或 text
```

### `--stdin-s16le` 识别结果异常

确认上游没有输出 WAV 头，并确认采样率、声道数、位深和字节序完全匹配。可以用
`arecord -t raw -f S16_LE -c 1 -r 16000` 作为参考。

### 没有看到 `final`

只有长度至少达到模型最小窗口的输入才会提交推理。如果输入为空或极短，程序
可能只输出 `ready` 和 `done`。对于正常长度的输入，程序在 EOF 收尾时会等待最后
一个 final 完成后才输出 `done`。

### `partial` 数量少于预期

这可能是正常现象：partial 是按时间门限提交的，而且后台队列会替换过期 partial。
查看 `done` 事件中的 `partial_dropped`，并根据需要增大 `--partial-interval-ms`。

### 如何只获取最终文本

使用 `--output text`：

```bash
build-native/bin/llama-funasr-stream \
  --enc funasr-encoder-f16.gguf \
  -m qwen3-0.6b-q4km.gguf \
  --audio sample.wav \
  --output text
```

## 13. 用 Python 逐行消费 JSONL

下面的示例只保存 `final` 事件，并将进程错误输出保留在终端：

```python
import json
import subprocess

cmd = [
    "build-native/bin/llama-funasr-stream",
    "--enc", "funasr-encoder-f16.gguf",
    "-m", "qwen3-0.6b-q4km.gguf",
    "--audio", "sample.wav",
    "--output", "jsonl",
]

with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
    for line in proc.stdout:
        event = json.loads(line)
        if event["type"] == "partial":
            print("临时结果:", event["text"])
        elif event["type"] == "final":
            print("最终结果:", event["text"])
        elif event["type"] == "done":
            print("处理完成，音频时长:", event["audio_ms"], "ms")
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"llama-funasr-stream exited with {return_code}")
```

生产环境中建议以 `final` 为准，以 `utterance_id` 和 `revision` 记录状态，并将
`partial` 仅用于实时界面展示。
