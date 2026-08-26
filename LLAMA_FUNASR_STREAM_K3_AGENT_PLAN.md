# K3 `llama-funasr-stream` 实施计划（Agent 执行文档）

> 适用仓库：`/home/cs/project/K3/FunASR`  
> 调研日期：2026-08-18（Asia/Shanghai）  
> 目标：新增可交叉编译、可部署到 K3、可持续接收音频并增量输出识别结果的 `llama-funasr-stream`。

## 1. 先读：目标边界与关键结论

本任务的第一版目标是**工程可用的分段流式识别**：模型常驻内存，持续接收 PCM 音频，以短块驱动端点检测，在说话过程中按较低频率产生可修订的 `partial`，在端点产生 `final`。第一版不是训练意义上的严格因果流式模型。

必须在实现和 README 中如实使用下列术语：

- `streaming I/O`：持续读入音频，不等待完整文件。
- `partial result`：对当前未结束语音段重新推理得到的临时结果，允许修订。
- `final result`：VAD 端点或最大语音段触发后得到的最终结果。
- `true/causal streaming`：encoder/adaptor/decoder 都只计算新增帧并复用状态；**当前 Fun-ASR-Nano 权重和 C++ 图尚不满足此定义**。

当前 Nano encoder 的 SAN-M 层包含整段双向 self-attention，FSMN kernel 也是左右对称；adaptor 同样是整段 self-attention。现有 `--chunk` 只是切分离线窗口，每个窗口都会执行：

```cpp
llama_memory_clear(llama_get_memory(ctx), true);
```

因此不要直接把 `--chunk` 政名后称为严格流式，也不要尝试把新 audio embedding 直接追加到已经开始生成文本的 LLM KV cache。旧 audio embedding 位于 prompt 中，变化后旧 KV 已失效；partial 必须从干净的 LLM 上下文重新解码。

## 2. 本地代码与版本基线

### 2.1 仓库状态

- FunASR revision：`9b918d74d7090b38211a6414c6ee08893a34afd4`
- describe：`runtime-llamacpp-v0.1.9-dirty`
- SpaceMIT llama.cpp：`../spacemit-llama.cpp`，tag `v0.1.7`
- SpaceMIT llama.cpp revision：`c9af964b52911bb83e6fe745dd69cefd48c58b26`
- toolchain：`../spacemit-toolchain-linux-musl-x86_64-oh-20260630`
- clang：21.1.8，target `riscv64-unknown-linux-musl`
- CMake：3.22.1

工作区已有用户修改和未跟踪文件。Agent 必须先运行 `git status --short`，保留所有既有修改，不得 reset、checkout 或覆盖无关内容。

### 2.2 现有离线链路

主实现位于：

- `runtime/llama.cpp/fun-asr-nano/funasr-cli/funasr-cli.cpp`
- `runtime/llama.cpp/funasr-common/funasr_audio.h`
- `runtime/llama.cpp/funasr-common/funasr_vad.h`
- `runtime/llama.cpp/CMakeLists.txt`
- `scripts/build-riscv64-spacemit-ohos.sh`

数据流为：

```text
完整音频
  -> 16 kHz mono f32
  -> 80 mel fbank + LFR(7,6)，每帧 560 维
  -> 50 层 SAN-M + 20 层 tp encoder
  -> adaptor
  -> fake_token_len 截断
  -> [prompt prefix | audio embeds | prompt suffix]
  -> Qwen3-0.6B llama_decode
  -> greedy text
```

需要注意的现状：

- `run_encoder()` 每次创建/销毁 ggml backend、graph context 和 gallocr。
- encoder 线程数硬编码为 8。
- LLM context 固定为 `n_ctx/n_batch/n_ubatch = 2048`。
- `funasr_load_audio_16k_mono()` 只面向文件 decoder，并通过 `MA_NO_DEVICE_IO` 关闭了采集。
- `funasr_vad_segments()` 面向完整 waveform，每次调用重新载入 VAD GGUF，不适合直接放进 100 ms 音频循环。
- 当前 sampler 在窗口之间未显式 reset；新流式实现的每次独立 hypothesis 必须调用 `llama_sampler_reset()`。

## 3. K3 实机基线

设备序列号：`HW3MPK3161280072`。

| 项目 | 实测值 |
|---|---|
| Kernel | Linux 6.18.3, PREEMPT_DYNAMIC |
| ABI | ELF64 RISC-V, double-float, musl loader `/lib/ld-musl-riscv64.so.1` |
| CPU | 16 核：8 × SpaceMIT X100 + 8 × SpaceMIT A100 |
| ISA | RV64 + V/RVV，包含 `zfh`、`zvfh`、`zicbop`、`zihintpause` 等 |
| RAM | 16 GiB，无 swap，调研时约 14 GiB 可用 |
| 可用存储 | `/data` 所在卷约 87 GiB 可用 |
| 录音工具 | `/bin/arecord` |
| 现有部署 | `/data/data/funasr` |

板端已有模型：

```text
/data/data/funasr/fun-asr-nano/funasr-encoder-f16.gguf  448 MiB
/data/data/funasr/fun-asr-nano/qwen3-0.6b-q4km.gguf      462 MiB
/data/data/funasr/fsmn-vad.gguf                         1.6 MiB
```

现有短音频离线基线：

```text
input: /data/data/funasr/test.wav（约 5.4 s PCM）
output: 开放时间早上九点至下午五点。
SPACEMIT_DISABLE_TCM=1 时：real 6.51 s，[done] 6.31 s
LLM KV: 224 MiB
LLM compute buffer: 1203 MiB
```

### 3.1 当前阻断问题：默认 TCM 路径会崩溃

未设置 `SPACEMIT_DISABLE_TCM=1` 时，现有 `llama-funasr-cli` 在 encoder 首次 graph compute 发生：

```text
wait tcm buffer failed for cpu_id: 4
ggml_backend_cpu_riscv64_spacemit_tcm_mem_wait_all
Signal 6
```

流式功能开发和正确性验收必须先禁用 TCM：

```bash
export SPACEMIT_DISABLE_TCM=1
```

TCM/IME2 优化是独立后端问题，不是本流式任务的前置功能。只有 CPU/HPAGE 路径通过全部验收后才允许单独恢复 TCM 做 A/B；任何 SIGABRT 都视为失败，不能通过重试掩盖。

## 4. 交付定义

### 4.1 必须交付（MVP）

1. 新可执行目标 `llama-funasr-stream`，不破坏 `llama-funasr-cli`。
2. 模型仅在进程启动时载入一次。
3. 支持两种输入：
   - `--audio FILE`：文件按固定大小分块送入，用于确定性测试；`--realtime` 可按音频时长节流。
   - `--stdin-s16le`：从 stdin 持续读取 16 kHz、mono、signed 16-bit little-endian PCM。
4. 以 JSON Lines 输出事件；stdout 只放协议数据，日志全部写 stderr。
5. 支持 `partial` 和 `final`，partial 可修订，final 不可回退。
6. 推理期间输入不能无限堆积；只能有一个 inference worker，pending partial 必须合并/丢弃旧请求。
7. EOF、SIGINT、最大段长都能 flush 当前语音并干净退出。
8. Native 和 K3 都能编译，K3 包中包含新 binary。
9. 提供自动化测试、板端验证脚本或明确可复制命令、README。

### 4.2 推荐的第一版语义

优先采用“VAD utterance 模式”：

```text
100 ms input block
  -> streaming VAD state
  -> speech buffer grows
  -> every partial_interval_ms: enqueue latest snapshot
  -> endpoint/max_utterance/eof: enqueue final snapshot
  -> Nano offline inference on snapshot
  -> partial/final JSONL
```

若持久化 FSMN-VAD 在本轮改造成本过高，允许先交付 `--segment-ms` 固定段模式，但必须：

- README 标记为 `fixed-window streaming MVP`；
- 每段输出 `final`；
- 相邻段支持 `--overlap-ms` 和文本去重合并；
- 把持久 VAD 列为紧随其后的下一里程碑；
- 不得用对完整累计 waveform 反复调用 `funasr_vad_segments()` 的方式冒充增量 VAD。

当前交付状态：本轮已实现上述 fixed-window MVP，输入为文件分块或 stdin S16LE；持久化 FSMN-VAD 尚未接入，因此不要给当前 binary 传 `--vad`。

### 4.3 本轮不做

- 不修改或重新训练 Fun-ASR-Nano 权重。
- 不承诺 encoder/adaptor 的增量 cache 与离线结果等价。
- 不实现 WebSocket/HTTP server；先稳定 stdin/stdout 协议。
- 不把麦克风设备 API 编进 miniaudio；K3 首版通过 `arecord | llama-funasr-stream` 解耦采集。
- 不在流式代码中引入 ONNX Runtime/SpaceMIT EP；Nano 仍使用 GGUF + ggml + llama.cpp。

## 5. 建议代码结构

不要继续把全部逻辑复制进第二个超大 `main()`。先做行为保持型抽取，再加 streaming orchestration。

严格抽取 `NanoEngine` 是后续整理方向。本轮 MVP 为了保持现有数值路径，stream target 直接复用离线 CLI 的实现单元，并在其上增加独立的 streaming worker；后续可无行为变化地拆成建议的 engine 文件。

建议最终整理为：

```text
runtime/llama.cpp/fun-asr-nano/funasr-common/
  funasr_nano_engine.h
  funasr_nano_engine.cpp
runtime/llama.cpp/fun-asr-nano/funasr-stream/
  funasr-stream.cpp
  pcm_stream.h
  stream_session.h
  stream_session.cpp
```

当前 MVP 只有 `funasr-stream.cpp`，通过单 TU 复用 `funasr-cli.cpp` 的数值实现；后续抽取 engine 时再补齐上述文件，并让两个 target 共享它。

### 5.1 `NanoEngine` 接口

接口至少覆盖：

```cpp
struct NanoConfig {
    std::string encoder_model;
    std::string llm_model;
    int n_threads = 8;
    int n_ctx = 2048;
    int n_predict = 256;
    float repeat_penalty = 1.0f;
};

struct NanoResult {
    std::string text;
    int64_t fbank_us = 0;
    int64_t encoder_us = 0;
    int64_t llm_prefill_us = 0;
    int64_t decode_us = 0;
};

class NanoEngine {
public:
    explicit NanoEngine(const NanoConfig & config);
    bool init(std::string & error);
    bool transcribe(const float * samples, size_t count,
                    NanoResult & result, std::string & error);
};
```

实现要求：

- `init()` 只执行一次 encoder GGUF、llama model、vocab、context、sampler 初始化。
- 每次 `transcribe()` 前清理该 hypothesis 的 LLM memory 并 reset sampler。
- `n_threads` 不得硬编码。
- 所有 ggml/llama 返回值都检查；失败通过 error 返回，main 输出 error event 后非零退出。
- 对短于 `WINLEN=400` 的输入不建图，返回空文本或明确错误。
- 检查 context 容量：`prefix + n_aud + suffix + n_predict <= n_ctx`；超限时不得越界。
- 保持 `fake_token_len` 公式与现有 CLI 完全一致。
- 第一轮抽取后，原 CLI 对同一 wav/model 的文本必须不变。

可以先保留 encoder graph 每次创建的行为保证正确性；模型/LLM context 常驻是 MVP 必须项。graph/gallocr 复用属于后续性能优化，只有 shape bucket 或最大 shape 方案经数值对齐后再做。

### 5.2 `PcmInput`

内部统一格式：16 kHz、mono、float32 `[-1, 1]`。

- 文件输入继续复用 `funasr_load_audio_16k_mono()`，但 main 必须按 `--input-block-ms` 切块送给 session，不能直接一次 transcribe 整个文件。
- stdin 每次读取完整的 little-endian `int16_t` frame；处理奇数字节和 EINTR。
- stdin EOF 时调用 `finish()`。
- 默认 `input_block_ms=100`，即每块 1600 samples / 3200 bytes。
- 不在 reader 中 sleep；仅 `--audio --realtime` 的模拟输入按 block duration 节流。

K3 麦克风使用示例：

```bash
arecord -q -t raw -f S16_LE -c 1 -r 16000 | \
  ./bin/llama-funasr-stream --stdin-s16le ...
```

### 5.3 `StreamSession`

必须分离采集与推理，避免 K3 推理慢于输入时阻塞声卡。

```text
reader/main thread -> bounded audio/session state -> one inference worker -> stdout
```

并发规则：

- 只有 inference worker 能调用 `NanoEngine::transcribe()`；llama context 不跨线程并发访问。
- `final` 请求绝不能丢。
- 同一 utterance 若已有 partial 正在计算，新 partial 只保留最新 snapshot；旧 pending partial 被覆盖。
- final 到来时丢弃该 utterance 尚未执行的 partial，final 排在当前计算后立即执行。
- 每个请求携带 `utterance_id` 和单调递增 `revision`；迟到的 partial 若其 revision 小于已输出 revision，直接丢弃。
- 队列设置硬上限，并在 stderr/metrics 中记录 dropped partial 数；禁止无界 vector/queue。
- stdout 输出需加 mutex 或只由 worker 输出，每个 JSON object 一行并 flush。

不要在每个 partial 后把生成文本 token 留在 KV 中。下一 partial 的 audio prompt 已变化，必须重新 prefill。

## 6. 输出协议与 CLI

### 6.1 CLI 最小集合

```text
llama-funasr-stream
  --enc PATH
  -m PATH
  (--audio PATH | --stdin-s16le)
  [--realtime]
  [--input-block-ms 100]
  [--partial-interval-ms 3000]
  [--segment-ms 12000]
  [--overlap-ms 1000]
  [-t 8]
  [-n 256]
  [--rep 1.0]
  [--output jsonl|text]
```

参数校验：

- 两种输入必须且只能选一种。
- block/interval/segment/max 必须为正数。
- `0 <= overlap_ms < segment_ms`。
- stdin 首版固定 16 kHz mono S16LE，收到其他格式不猜测。
- `--output text` 仅输出 final，便于 shell 使用；自动测试以 JSONL 为准。

### 6.2 JSON Lines 协议

启动成功：

```json
{"type":"ready","sample_rate":16000,"mode":"vad","model_loaded_ms":1234}
```

临时结果：

```json
{"type":"partial","utterance_id":1,"revision":2,"begin_ms":840,"end_ms":6120,"text":"开放时间早上九点","inference_ms":920,"audio_ms":5280}
```

最终结果：

```json
{"type":"final","utterance_id":1,"revision":3,"begin_ms":840,"end_ms":7340,"text":"开放时间早上九点至下午五点。","inference_ms":1080,"audio_ms":6500,"reason":"vad_endpoint"}
```

结束：

```json
{"type":"done","utterances":1,"partial_dropped":2,"audio_ms":8000}
```

错误：

```json
{"type":"error","code":"decode_failed","message":"..."}
```

JSON string 必须正确转义引号、反斜线和控制字符；不要手拼未经转义的模型文本。

## 7. VAD 实施路线

### 7.1 推荐：持久化 `StreamingVad`

将 `funasr_vad.h` 中模型、前端和状态机拆出可复用对象，避免每 100 ms 重新载入模型。

建议接口：

```cpp
enum class VadEventType { SpeechStart, SpeechEnd };
struct VadEvent { VadEventType type; int64_t time_ms; };

class StreamingVad {
public:
    bool init(const std::string & gguf, int n_threads, std::string & error);
    bool accept(const float * samples, size_t count,
                std::vector<VadEvent> & events, std::string & error);
    bool finish(std::vector<VadEvent> & events, std::string & error);
    void reset();
};
```

增量实现必须保留：

- fbank 25 ms window、10 ms shift 的跨块 sample tail；
- LFR m=5/n=1 所需的 lookahead，非 EOF 时不能用尾帧 padding 伪造未来；
- 每层 FSMN 的左侧历史，或使用经证明足够的重算上下文；
- E2E VAD state machine 的窗口、起点回看、终点回看和最大段状态；
- finish 时对残留 feature 做一次合法尾部 flush。

先做离线等价测试：将同一 waveform 随机切成 10/37/100/240 ms 块，streaming VAD 产生的 segment 与 `funasr_vad_segments()` 相比，start/end 误差目标不超过 20 ms。未通过此测试前，不上板调 Nano。

### 7.2 可接受的 MVP 回退

若本轮只实现固定段：

- 默认 `segment_ms=12000`、`overlap_ms=1000`；数值需通过板端 benchmark 调整。
- 每个段独立推理并输出 final。
- 对相邻文本做 UTF-8 codepoint 级 suffix/prefix overlap，设置最小匹配长度；不能按 byte 截断中文。
- 合并失败时保留两段并记录 debug log，不得误删文本。
- EOF 对不足一个完整段的尾音频执行 final。

## 8. 构建改动

### 8.1 CMake

修改 `runtime/llama.cpp/CMakeLists.txt`，新增目标并复用 engine source：

```cmake
funasr_add(llama-funasr-stream
    fun-asr-nano/funasr-stream/funasr-stream.cpp
    llama ggml)
```

当前 `funasr-stream.cpp` 通过单 TU 复用 `funasr-cli.cpp` 的离线数值实现；不要在后续重构时改变 encoder/fbank/fake-token 语义。

如果 CLI 也完成 engine 抽取，则给 CLI 加相同 engine source。不要链接 SpaceMIT ORT；该 target 只依赖 `llama`、`ggml`、`Threads::Threads`。

### 8.2 K3 build/package 脚本

修改 `scripts/build-riscv64-spacemit-ohos.sh`：

```bash
targets=(
  llama-funasr-cli
  llama-funasr-stream
  ...
)
```

现有脚本会将 targets 复制到 `build-riscv64-ohos/package/bin`。构建 flags 保持现状：

```text
CMAKE_TOOLCHAIN_FILE=../spacemit-llama.cpp/cmake/riscv64-spacemit-ohos.cmake
GGML_NATIVE=OFF
GGML_CPU_RISCV64_SPACEMIT=ON
GGML_RVV=ON
GGML_RV_ZVFH=ON
GGML_RV_ZFH=ON
GGML_RV_ZICBOP=ON
GGML_RV_ZIHINTPAUSE=ON
GGML_RV_ZBA=ON
GGML_OPENMP=OFF
LLAMA_CURL=OFF
```

在 package logs 中增加新 binary 的：

```bash
file package/bin/llama-funasr-stream
readelf -h package/bin/llama-funasr-stream
readelf -d package/bin/llama-funasr-stream
```

预期：ELF64 RISC-V、musl interpreter；不得意外依赖 host glibc 或 ONNX Runtime。

### 8.3 构建命令

Native 快速循环：

```bash
cmake -S runtime/llama.cpp -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA=../spacemit-llama.cpp \
  -DLLAMA_CURL=OFF
cmake --build build-native -j"$(nproc)" --target llama-funasr-cli llama-funasr-stream
```

K3 交叉编译：

```bash
scripts/build-riscv64-spacemit-ohos.sh
```

Agent 不得在旧 CMake cache 指向不同 llama.cpp 时盲目复用；先查看 `CMakeCache.txt`。确需新 build dir 时使用新的明确目录，不删除用户已有 build。

## 9. 测试计划与门禁

### 9.1 单元测试

至少覆盖：

1. S16LE 到 float32 的边界值和奇数字节处理。
2. 输入参数互斥与非法 interval/overlap。
3. JSON escaping（中文、引号、反斜线、换行）。
4. fixed-window UTF-8 overlap merge。
5. partial coalescing：慢 worker 下队列有界，final 不丢。
6. EOF/SIGINT flush。
7. 若实现 VAD：随机块大小下与离线 VAD 的 segment 对齐。

### 9.2 Native 集成测试

必须验证：

- 原 `llama-funasr-cli` 抽取前后同一输入文本一致。
- 文件分块为 100 ms 与 240 ms 时，stream final 文本/端点基本一致。
- `--audio` 不加 `--realtime` 可快速跑完，便于 CI。
- JSONL 每行都能被标准 JSON parser 解析。
- 重复运行同一输入不崩溃、不增长到无界内存。

### 9.3 K3 部署

使用已确认的 HDC：

```bash
HDC=/home/cs/project/toolchains/hdc
SERIAL=HW3MPK3161280072

$HDC -t "$SERIAL" file send \
  build-riscv64-ohos/package/bin/llama-funasr-stream \
  /data/data/funasr/bin/llama-funasr-stream

$HDC -t "$SERIAL" shell \
  "chmod 755 /data/data/funasr/bin/llama-funasr-stream"
```

文件模拟运行：

```bash
$HDC -t "$SERIAL" shell '
  cd /data/data/funasr
  export LD_LIBRARY_PATH=$PWD/lib:/system/lib64
  export SPACEMIT_DISABLE_TCM=1
  ./bin/llama-funasr-stream \
    --enc fun-asr-nano/funasr-encoder-f16.gguf \
    -m fun-asr-nano/qwen3-0.6b-q4km.gguf \
    --audio test.wav --input-block-ms 100 --output jsonl
'
```

麦克风运行：

```bash
$HDC -t "$SERIAL" shell '
  cd /data/data/funasr
  export LD_LIBRARY_PATH=$PWD/lib:/system/lib64
  export SPACEMIT_DISABLE_TCM=1
  arecord -q -t raw -f S16_LE -c 1 -r 16000 | \
    ./bin/llama-funasr-stream \
      --enc fun-asr-nano/funasr-encoder-f16.gguf \
      -m fun-asr-nano/qwen3-0.6b-q4km.gguf \
      --stdin-s16le --segment-ms 12000 --overlap-ms 1000 --output jsonl
'
```

### 9.4 功能验收

必须同时满足：

- 启动后模型加载一次，随后输出 `ready`。
- 连续输入至少 10 分钟无崩溃、死锁、无界 RSS 增长。
- 文件模拟和 stdin 都有 final；EOF 尾段不丢。
- partial 的 `(utterance_id, revision)` 单调，final 后不再出现同 utterance 的 partial。
- stdout 无模型 loader 日志污染；每行合法 JSON。
- K3 默认测试设置 `SPACEMIT_DISABLE_TCM=1` 后不出现 SIGABRT。
- 新 target 不改变原 CLI 的离线结果。

### 9.5 性能验收与记录

不要在尚无实测前写死不现实的“首字 < 500 ms”。K3 现有完整进程对约 5.4 s 音频耗时约 6.5 s，而模型常驻后实际 partial 成本尚待测量。

Agent 先记录，再据数据调参：

| 配置 | audio ms | fbank ms | encoder ms | LLM prefill ms | decode ms | total ms | RTF | RSS MiB | dropped partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 s partial | | | | | | | | | |
| 6 s partial | | | | | | | | | |
| 12 s final | | | | | | | | | |

性能门禁：

- 模型常驻后每次 inference 不重新读取 900 MiB 模型。
- 采集线程不因一次推理而永久停止读输入。
- backlog 时丢/合并 partial，不丢 final。
- 10 分钟 steady-state RSS 不随 utterance 数线性增长。
- 报告 model init 与 steady-state inference 分开，不用首次 repack 时间冒充稳态。

## 10. 分阶段执行顺序

### 阶段 A：建立可运行基线

- [ ] 保存 `git status --short` 和版本信息。
- [ ] Native、K3 各跑一次原 CLI；K3 强制 `SPACEMIT_DISABLE_TCM=1`。
- [ ] 保存输入、文本、耗时、RSS，确认复现本文件中的基线。

停止条件：原 CLI 在禁用 TCM 后仍崩溃时，先诊断后端/构建，不开始 stream 并发代码。

### 阶段 B：抽取 `NanoEngine`

- [ ] 移动 fbank/encoder/adaptor/LLM 逻辑到复用模块。
- [ ] CLI 改用 `NanoEngine`，参数和输出保持兼容。
- [ ] 增加 error handling、`-t`、context 容量检查、sampler reset。
- [ ] Native 回归同一文本。

提交建议：单独提交纯重构，便于 bisect。

### 阶段 C：固定段 streaming MVP

- [ ] 新增 stdin/file chunk reader。
- [ ] 新增单 worker、有界队列、partial coalescing/final priority。
- [ ] 实现固定段/overlap/UTF-8 merge 和 JSONL。
- [ ] 覆盖 EOF、SIGINT、慢推理 backlog。
- [ ] Native 长音频运行通过。

这是最早可以对外称为 `llama-funasr-stream` 的版本，但 README 必须标注 fixed-window。

### 阶段 D：持久 FSMN-VAD utterance 模式

- [ ] 实现 StreamingVad 并通过随机 chunk 离线等价测试。
- [ ] 接入 speech start/end、partial interval、max utterance。
- [ ] 同一 utterance 的 partial 允许修订，endpoint 产 final。
- [ ] K3 `arecord` 实测。

### 阶段 E：K3 优化

- [ ] 用 stage timing 确定 encoder 还是 LLM 是瓶颈。
- [ ] 调 `partial_interval_ms`，避免默认配置 backlog。
- [ ] 尝试减小 `n_batch/n_ubatch`，以实测内存/速度为准。
- [ ] 评估 encoder graph shape bucket/gallocr 复用。
- [ ] CPU/HPAGE 稳定后，才单独调查 TCM/IME2 SIGABRT。

### 阶段 F：严格因果流式可行性研究（不阻塞 MVP）

只有找到官方 Fun-ASR-Nano streaming 配置/权重或完成参考 PyTorch 原型后才能进入：

- encoder attention mask 是否训练过 chunk/causal 模式；
- SAN-M 对称 FSMN 如何转换为可控 lookahead；
- adaptor self-attention 的 cache/mask 语义；
- audio prompt 更新时 LLM 是否有训练匹配的增量协议；
- streaming 与离线的 CER/WER 退化。

未经模型侧验证，不允许仅通过裁 attention mask 或缓存 K/V 宣称完成 true streaming。

## 11. 常见错误与禁止项

- 禁止每个 100 ms block 跑一次完整 Nano；K3 会迅速 backlog。
- 禁止每个 partial 重载 GGUF。
- 禁止在变化的 audio prompt 后继续沿用旧 LLM KV。
- 禁止同时从两个线程调用同一 llama context。
- 禁止把 stderr loader 日志混进 JSONL stdout。
- 禁止用 byte substring 合并中文文本。
- 禁止为了“实时”无限扩大队列；实时系统宁可丢过期 partial。
- 禁止默认启用已知会 SIGABRT 的 TCM 路径。
- 禁止只测 5 秒文件就宣称稳定；至少做 10 分钟连续输入。
- 禁止修改现有模型文件或覆盖板端原 binary；首次部署建议使用新文件名并保留回滚路径。

## 12. Agent 最终交付报告模板

Agent 完成后必须给出：

1. 修改文件列表和每个文件职责。
2. `git diff --stat`，说明未触碰的用户既有改动。
3. Native build/test 命令与结果。
4. K3 ELF/readelf 结果、部署路径、运行命令。
5. 一段完整 JSONL 示例（ready → partial → final → done）。
6. 原 CLI 回归文本与新 stream final 对比。
7. 上述性能表，区分 init 和 steady-state。
8. 10 分钟稳定性结果与 RSS 起止值。
9. TCM 是否仍禁用；若启用，提供不崩溃的证据。
10. 未完成项和下一步，尤其说明当前属于 fixed-window、VAD utterance 还是 true causal streaming。

最终验收重点不是 binary 名称出现，而是：**持续输入不丢 final、模型不重复加载、慢推理不产生无界 backlog、输出协议可消费、K3 可稳定运行，并准确说明流式语义。**
