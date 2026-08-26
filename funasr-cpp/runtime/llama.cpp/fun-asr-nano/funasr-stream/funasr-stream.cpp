// llama-funasr-stream: bounded streaming wrapper for Fun-ASR-Nano, with optional FSMN-VAD.
//
// The Nano SAN-M encoder is bidirectional, so this target deliberately performs
// a fresh hypothesis on each partial/final window.  It provides streaming I/O
// and incremental JSONL output while preserving the offline model semantics.
#define main funasr_offline_main_disabled
#include "../funasr-cli/funasr-cli.cpp"
#undef main

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include "funasr_vad_stream.h"

namespace {

// ----------------------------- 参数与任务数据结构 -----------------------------
// Options 保存命令行参数。这里把“模型配置”“输入方式”“流式切分参数”集中在
// 一个结构体中，后续运行时、工作线程和主循环都只持有该配置的只读引用。
struct Options {
    // enc: 音频编码器模型路径；llm: 语言模型路径。
    std::string enc, llm;
    // 语言提示：auto 保持模型自动判断，也支持 zh/en/ja。
    std::string language = "auto";
    // VAD 模型路径；启用后仅对 VAD 检出的语音区间进行识别。
    std::string vad;
    // 为 true 时从 stdin 连续读取 16 kHz / 单声道 / signed 16-bit little-endian PCM。
    bool stdin_s16le = false;
    // 每次送入流式缓存的基础音频块长度。较小可提高时间粒度，但循环更频繁。
    int block_ms = 100;
    // 采集线程与 VAD 线程之间的环形缓冲容量，用于吸收 ASR/VAD 的瞬时抖动。
    int audio_buffer_ms = 60000;
    // partial 结果的最小时间间隔：积累到该时间点后提交一次临时识别。
    int partial_ms = 10000;
    // 单个最终识别段的最大长度。达到该长度后生成 final，并开启下一 utterance。
    int segment_ms = 30000;
    // 相邻 final 段之间保留的重叠音频，用于降低硬切分造成的词语截断风险。
    int overlap_ms = 1000;
    // 单个 VAD 语音段的最大长度；VAD 本身仍会在该长度处闭合长语音段。
    int vad_maxseg_ms = 12000;
    // VAD speech/noise 判定阈值；默认与 FSMN-VAD 原实现一致为 0.5。
    float vad_speech_noise_thres = 0.5f;
    // 流式 VAD 内部固定处理块；任意长度的 stdin 输入会先在 cache 中拼成该大小。
    int vad_chunk_ms = 60;
    // 固定流式尾静音阈值，默认 800 ms。
    int vad_max_end_silence_ms = 800;
    // LLM 最多生成的 token 数。
    int npred = 512;
    // ASR 与 VAD 分开限流，避免两个计算阶段同时抢占相同的 CPU 核。
    int asr_threads = 4;
    int vad_threads = 2;
    // repetition penalty；1.0 表示不施加重复惩罚，默认使用轻微惩罚 1.01。
    float rep = 1.01f;
    // 输出格式：jsonl 会输出 ready/partial/final/done 事件；text 只打印 final 文本。
    std::string output = "jsonl";
};

// 一次提交给后台推理线程的任务快照。
// samples 保存“提交瞬间”的完整窗口，因此主线程后续继续追加 segment 不会影响该任务。
struct Job {
    // 16 kHz 单声道浮点 PCM，通常归一化到 [-1, 1)。
    std::vector<float> samples;
    // 当前窗口在整段输入中的起止时间戳。
    int64_t begin_ms = 0;
    int64_t end_ms = 0;
    // utterance 标识一个 final 分段；revision 标识同一分段内 partial/final 的版本号。
    uint64_t utterance = 0;
    uint64_t revision = 0;
    // false=临时结果，可被更新/丢弃；true=最终结果，不能被后续 partial 越过。
    bool final = false;
    // final 产生原因，例如 max_segment 或 eof。
    std::string reason;
};

// 打印命令行使用说明。
static void stream_usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --enc <encoder.gguf> (-m|--model) <model.gguf> "
        "--stdin-s16le [options]\n"
        "\n"
        "Required arguments:\n"
        "  --enc <path>              Encoder model in GGUF format\n"
        "  -m, --model <path>        Language model in GGUF format\n"
        "\n"
        "Audio input:\n"
        "  --stdin-s16le             Read raw 16 kHz mono signed 16-bit LE PCM\n"
        "                            from standard input\n"
        "\n"
        "Language options:\n"
        "  --language <auto|zh|en|ja> Language hint for transcription\n"
        "                            (default: auto)\n"
        "\n"
        "VAD options:\n"
        "  --vad <path>              FSMN-VAD model in GGUF format\n"
        "                            streaming VAD for stdin PCM\n"
        "  --vad-maxseg <ms>         Maximum VAD speech segment length\n"
        "                            (default: 12000 ms)\n"
        "  --vad-threshold <F>     Speech/noise threshold in range 0.0-1.0\n"
        "                            (default: 0.5)\n"
        "  --vad-chunk-ms <ms>       Internal streaming VAD chunk size\n"
        "                            (default: 60 ms)\n"
        "  --vad-max-end-silence-ms <ms>\n"
        "                            Fixed endpoint silence (default: 800 ms)\n"
        "\n"
        "Streaming options:\n"
        "  --input-block-ms <ms>     Input audio block size\n"
        "                            (default: 100 ms)\n"
        "  --audio-buffer-ms <ms>    Capture/VAD ring buffer capacity\n"
        "                            (default: 60000 ms)\n"
        "  --partial-interval-ms <ms>\n"
        "                            Minimum interval between partial results\n"
        "                            (default: 10000 ms)\n"
        "  --segment-ms <ms>         Maximum recognition segment length before\n"
        "                            forced finalization\n"
        "                            (default: 30000 ms)\n"
        "  --overlap-ms <ms>         Audio overlap between adjacent fixed-window\n"
        "                            segments\n"
        "                            (default: 1000 ms; must be < --segment-ms)\n"
        "\n"
        "Inference options:\n"
        "  -n, --max-tokens <N>      Maximum generated tokens per inference\n"
        "                            (default: 512)\n"
        "  -t, --threads <N>         Alias for --asr-threads\n"
        "  --asr-threads <N>         Encoder/LLM CPU threads (default: 4)\n"
        "  --vad-threads <N>         FSMN-VAD CPU threads (default: 2)\n"
        "  --rep <F>                 Repetition penalty\n"
        "                            1.0 disables repetition penalty\n"
        "                            (default: 1.01)\n"
        "\n"
        "Output options:\n"
        "  --output <format>         Output format: jsonl or text\n"
        "                            (default: jsonl)\n"
        "                            jsonl: emit ready/partial/final/done events\n"
        "                            text:  print final transcripts to stdout and\n"
        "                                   statistics to stderr\n"
        "\n"
        "General options:\n"
        "  -h, --help                Show this help message and exit\n"
        "\n"
        "Examples:\n"
        "  ./bin/llama-funasr-stream --enc ./fun-asr-nano/funasr-encoder-f16.gguf \\\n"
        "     -m ./fun-asr-nano/qwen3-0.6b-q4km.gguf \\\n"
        "     --vad fsmn-vad.gguf --vad-maxseg 12000 \\\n"
        "     --stdin-s16le --vad-threshold 0.5 \\\n"
        "     --output jsonl\n"
        "\n"
        "  arecord -q -D plughw:1,0 -t raw -f S16_LE -c 1 -r 16000 | \\\n"
        "  ./bin/llama-funasr-stream --enc ./fun-asr-nano/funasr-encoder-f16.gguf \\\n"
        "     -m ./fun-asr-nano/qwen3-0.6b-q4km.gguf --stdin-s16le \\\n"
        "     --vad fsmn-vad.gguf --vad-maxseg 12000 \\\n"
        "     --partial-interval-ms 10000 --output jsonl\n",
        argv0);
}

// 解析并校验命令行参数。
// 返回 false 代表参数缺失、互斥条件冲突或数值范围非法，调用者随后打印 usage。
static bool parse_options(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        // 带值参数统一通过 need() 取得下一个 argv，避免每个分支重复边界检查。
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", name); return nullptr; }
            return argv[++i];
        };
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            stream_usage(argv[0]);
            exit(0);
        }
        else if (!strcmp(a, "--enc")) { const char * v = need(a); if (!v) return false; o.enc = v; }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) { const char * v = need(a); if (!v) return false; o.llm = v; }
        else if (!strcmp(a, "--language")) { const char * v = need(a); if (!v) return false; o.language = v; }
        else if (!strcmp(a, "--vad")) { const char * v = need(a); if (!v) return false; o.vad = v; }
        else if (!strcmp(a, "--vad-maxseg")) { const char * v = need(a); if (!v) return false; o.vad_maxseg_ms = atoi(v); }
        else if (!strcmp(a, "--vad-threshold")) { const char * v = need(a); if (!v) return false; o.vad_speech_noise_thres = strtof(v, nullptr); }
        else if (!strcmp(a, "--vad-chunk-ms")) { const char * v = need(a); if (!v) return false; o.vad_chunk_ms = atoi(v); }
        else if (!strcmp(a, "--vad-max-end-silence-ms")) { const char * v = need(a); if (!v) return false; o.vad_max_end_silence_ms = atoi(v); }
        else if (!strcmp(a, "--stdin-s16le")) o.stdin_s16le = true;
        else if (!strcmp(a, "--input-block-ms")) { const char * v = need(a); if (!v) return false; o.block_ms = atoi(v); }
        else if (!strcmp(a, "--audio-buffer-ms")) { const char * v = need(a); if (!v) return false; o.audio_buffer_ms = atoi(v); }
        else if (!strcmp(a, "--partial-interval-ms")) { const char * v = need(a); if (!v) return false; o.partial_ms = atoi(v); }
        else if (!strcmp(a, "--segment-ms")) { const char * v = need(a); if (!v) return false; o.segment_ms = atoi(v); }
        else if (!strcmp(a, "--overlap-ms")) { const char * v = need(a); if (!v) return false; o.overlap_ms = atoi(v); }
        else if (!strcmp(a, "-n") || !strcmp(a, "--max-tokens")) { const char * v = need(a); if (!v) return false; o.npred = atoi(v); }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads") || !strcmp(a, "--asr-threads")) { const char * v = need(a); if (!v) return false; o.asr_threads = atoi(v); }
        else if (!strcmp(a, "--vad-threads")) { const char * v = need(a); if (!v) return false; o.vad_threads = atoi(v); }
        else if (!strcmp(a, "--rep")) { const char * v = need(a); if (!v) return false; o.rep = strtof(v, nullptr); }
        else if (!strcmp(a, "--output")) { const char * v = need(a); if (!v) return false; o.output = v; }
        else { fprintf(stderr, "unknown option: %s\n", a); return false; }
    }
    // 流式模式只接受 stdin 原始 PCM；同时检查所有时间/推理参数。
    if (o.enc.empty() || o.llm.empty() || !o.stdin_s16le ||
        o.block_ms <= 0 || o.audio_buffer_ms < o.block_ms ||
        o.partial_ms <= 0 || o.segment_ms <= 0 ||
        o.overlap_ms < 0 || o.overlap_ms >= o.segment_ms || o.npred <= 0 ||
        o.asr_threads <= 0 || o.vad_threads <= 0 || o.vad_maxseg_ms <= 0 ||
        o.vad_chunk_ms <= 0 || o.vad_max_end_silence_ms <= 0 ||
        o.vad_speech_noise_thres < 0.0f || o.vad_speech_noise_thres > 1.0f ||
        (o.language != "auto" && o.language != "zh" && o.language != "en" && o.language != "ja") ||
        (o.output != "jsonl" && o.output != "text")) {
        return false;
    }
    return true;
}

// 对字符串进行最小 JSON 转义，保证识别文本可以安全嵌入单行 JSONL。
// 特别处理控制字符；普通 UTF-8 字节保持原样。
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    static const char hex[] = "0123456789abcdef";
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) { out += "\\u00"; out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]); }
            else out.push_back((char)c);
        }
    }
    return out;
}

// 合并相邻 final 窗口的文本。由于相邻音频窗口可能有 overlap，识别文本也可能重复。
// 这里寻找 old_text 的最长后缀与 next 的最长前缀匹配部分，并只追加未重复的尾部。
static std::string merge_text(const std::string & old_text, const std::string & next) {
    if (old_text.empty()) return next;
    if (next.empty()) return old_text;
    if (old_text == next) return old_text;
    const size_t max_k = std::min(old_text.size(), next.size());
    // 只接受 UTF-8 code point 边界，避免在多字节字符中间进行重叠匹配。
    for (size_t k = max_k; k > 0; --k) {
        const size_t old_start = old_text.size() - k;

        // old_text 后缀必须从 UTF-8 code point 边界开始。
        if (old_start < old_text.size() &&
            (static_cast<unsigned char>(old_text[old_start]) & 0xc0) == 0x80)
            continue;

        // next 前缀必须在 UTF-8 code point 边界结束。
        if (k < next.size() && (static_cast<unsigned char>(next[k]) & 0xc0) == 0x80) continue;

        if (old_text.compare(old_start, k, next, 0, k) == 0)
            return old_text + next.substr(k);
    }
    return old_text + next;
}

// ----------------------------- 模型推理运行时 -----------------------------
// NanoRuntime 负责模型资源的生命周期以及“一段音频 -> 一段文本”的完整推理。
// 它只在 Worker 线程中执行 transcribe，因此本类本身不额外加锁。
class NanoRuntime {
public:
    explicit NanoRuntime(const Options & o) : opt_(o) {}
    // 按与创建相反的层级释放 sampler/context/model/encoder 权重资源。
    ~NanoRuntime() {
        if (sampler_) llama_sampler_free(sampler_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_model_free(model_);
        if (em_.ctx_w) ggml_free(em_.ctx_w);
    }

    // 初始化编码器、LLM context 与采样器，并预先 tokenize 固定 prompt。
    bool init(std::string & error) {
        // 初始化 ggml 计时设施，后续用于统计加载与推理耗时。
        ggml_time_init();
        if (!load_enc(opt_.enc.c_str(), em_)) { error = "failed to load encoder: " + opt_.enc; return false; }
        // 加载可用 ggml backend；随后创建 LLM。当前显式设置 n_gpu_layers=0，即 CPU 路径。
        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(opt_.llm.c_str(), mp);
        if (!model_) { error = "failed to load LLM: " + opt_.llm; return false; }
        vocab_ = llama_model_get_vocab(model_);
        llama_context_params cp = llama_context_default_params();
        // 固定上下文和批处理上限。音频 embedding + prompt + 生成 token 均需落在上下文容量内。
        cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 2048;
        // -t 同时控制单 token 生成和 prompt/audio batch 处理线程数。
        cp.n_threads = opt_.asr_threads;
        cp.n_threads_batch = opt_.asr_threads;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) { error = "failed to create llama context"; return false; }
        auto sp = llama_sampler_chain_default_params();
        sampler_ = llama_sampler_chain_init(sp);
        if (!sampler_) { error = "failed to create sampler"; return false; }
        // 可选重复惩罚后接 greedy sampler，因此每一步选择当前最高概率 token。
        if (opt_.rep != 1.0f) llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(256, opt_.rep, 0.0f, 0.0f));
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
        // prefix/suffix 是每次识别都相同的文本 prompt，初始化时预编码可避免重复 tokenize。
        std::string language_prompt = "语音转写：";
        if (opt_.language == "zh") language_prompt = "语音转写成中文：";
        else if (opt_.language == "en") language_prompt = "语音转写成英文：";
        else if (opt_.language == "ja") language_prompt = "语音转写成日文：";
        const std::string prefix_text =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + language_prompt;
        prefix_ = tokenize(prefix_text.c_str());
        suffix_ = tokenize("<|im_end|>\n<|im_start|>assistant\n");
        return true;
    }

    // 对当前完整音频窗口重新执行一次离线语义的识别。
    // 由于 Nano SAN-M 编码器是双向模型，partial 并不是维护 encoder cache，而是对最新窗口重算。
    bool transcribe(const std::vector<float> & wav, std::string & text, int64_t & elapsed_us, std::string & error) {
        const int64_t start = ggml_time_us();
        // 音频短于模型所需最小窗口时不推理，视作合法但暂时无输出。
        if ((int)wav.size() < WINLEN) { text.clear(); elapsed_us = 0; return true; }
        int T = 0;
        // 1) PCM -> fbank 声学特征；T 为特征时间步数。
        auto fbank = compute_fbank(wav, T);
        int D = 0;
        // 2) 运行语音编码器/adapter，得到可送入 LLM 的连续 audio embeddings。
        auto adp = run_encoder(em_, std::move(fbank), T, 560, D, opt_.asr_threads);
        // 根据编码器两级下采样关系估算输出时间长度，再换算实际音频 embedding 数量。
        int ol = 1 + (T - 3 + 2) / 2; ol = 1 + (ol - 3 + 2) / 2;
        int n_aud = (ol - 1) / 2 + 1;
        if (n_aud <= 0 || (size_t)n_aud * (size_t)D > adp.size()) { error = "invalid audio embedding length"; return false; }
        // 每个 partial/final 都是独立的新假设，因此清空 KV/memory 与 sampler 状态，不沿用上次解码。
        llama_memory_clear(llama_get_memory(ctx_), true);
        llama_sampler_reset(sampler_);
        int n_past = 0;
        // 3) 按“文本前缀 -> 音频 embedding -> 文本后缀”的顺序构造 LLM 上下文。
        // 最后一段 suffix 开启 logits，供紧接着的首 token 采样。
        if (decode_batch(ctx_, (int)prefix_.size(), prefix_.data(), nullptr, 0, n_past, false) != 0 ||
            decode_batch(ctx_, n_aud, nullptr, adp.data(), D, n_past, false) != 0 ||
            decode_batch(ctx_, (int)suffix_.size(), suffix_.data(), nullptr, 0, n_past, true) != 0) {
            error = "llama_decode failed"; return false;
        }
        // 4) 自回归生成转写文本，遇到 EOG 或达到 npred 上限即停止。
        llama_token tk = llama_sampler_sample(sampler_, ctx_, -1);
        text.clear();
        for (int i = 0; i < opt_.npred && !llama_vocab_is_eog(vocab_, tk); ++i) {
            char buf[256]; int k = llama_token_to_piece(vocab_, tk, buf, sizeof(buf), 0, true);
            if (k > 0) text.append(buf, k);
            if (decode_batch(ctx_, 1, &tk, nullptr, 0, n_past, true) != 0) { error = "llama_decode token failed"; return false; }
            tk = llama_sampler_sample(sampler_, ctx_, -1);
        }
        elapsed_us = ggml_time_us() - start;
        return true;
    }

private:
    // llama_tokenize 的常见“两次调用”模式：第一次只查询所需 token 数，第二次真正写入数组。
    std::vector<llama_token> tokenize(const char * s) {
        int n = -llama_tokenize(vocab_, s, strlen(s), nullptr, 0, false, true);
        std::vector<llama_token> v((size_t)n);
        llama_tokenize(vocab_, s, strlen(s), v.data(), n, false, true);
        return v;
    }
    const Options & opt_;
    enc_model em_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    llama_sampler * sampler_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::vector<llama_token> prefix_, suffix_;
};

// ----------------------------- 专用 PCM 采集线程 -----------------------------
// fread/PCM 转换只在 reader_ 中运行。VAD 或 ASR 无论暂停多久，都不会直接
// 占用采集线程；固定容量环形缓冲负责吸收计算抖动并保持音频顺序。
class PcmInputRing {
public:
    PcmInputRing(size_t read_samples, size_t capacity_samples)
        : read_samples_(std::max<size_t>(1, read_samples)),
          data_(std::max(read_samples_, capacity_samples)) {}

    ~PcmInputRing() { stop(); join(); }

    void start() { reader_ = std::thread(&PcmInputRing::read_loop, this); }

    bool pop(std::vector<float> & out) {
        std::unique_lock<std::mutex> lock(mu_);
        data_ready_.wait(lock, [&] { return size_ > 0 || finished_; });
        if (size_ == 0) return false;
        const size_t count = std::min(size_, read_samples_);
        out.resize(count);
        const size_t first = std::min(count, data_.size()-head_);
        std::copy(data_.begin()+(ptrdiff_t)head_,
                  data_.begin()+(ptrdiff_t)(head_+first), out.begin());
        if (first < count) {
            std::copy(data_.begin(), data_.begin()+(ptrdiff_t)(count-first),
                      out.begin()+(ptrdiff_t)first);
        }
        head_ = (head_+count)%data_.size();
        size_ -= count;
        space_ready_.notify_one();
        return true;
    }

    void join() {
        if (reader_.joinable()) reader_.join();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_requested_ = true;
        }
        space_ready_.notify_all();
        data_ready_.notify_all();
    }

    bool ok() const {
        std::lock_guard<std::mutex> lock(mu_);
        return !read_error_;
    }

    bool incomplete_byte() const {
        std::lock_guard<std::mutex> lock(mu_);
        return incomplete_byte_;
    }

    size_t high_watermark_samples() const {
        std::lock_guard<std::mutex> lock(mu_);
        return high_watermark_;
    }

private:
    bool push(const std::vector<float> & samples) {
        size_t offset = 0;
        while (offset < samples.size()) {
            std::unique_lock<std::mutex> lock(mu_);
            space_ready_.wait(lock, [&] {
                return stop_requested_ || size_ < data_.size();
            });
            if (stop_requested_) return false;
            const size_t tail = (head_+size_)%data_.size();
            const size_t count = std::min(samples.size()-offset,
                std::min(data_.size()-size_, data_.size()-tail));
            std::copy(samples.begin()+(ptrdiff_t)offset,
                      samples.begin()+(ptrdiff_t)(offset+count),
                      data_.begin()+(ptrdiff_t)tail);
            offset += count;
            size_ += count;
            high_watermark_ = std::max(high_watermark_, size_);
            data_ready_.notify_one();
        }
        return true;
    }

    void read_loop() {
        std::vector<unsigned char> raw(read_samples_*sizeof(int16_t));
        unsigned char carry = 0;
        bool have_carry = false;
        for (;;) {
            const size_t got = fread(raw.data(), 1, raw.size(), stdin);
            if (got == 0) break;
            size_t begin = 0;
            std::vector<float> block;
            block.reserve((got+(have_carry ? 1 : 0))/2);
            if (have_carry) {
                const int16_t sample = (int16_t)((uint16_t)carry |
                                                 ((uint16_t)raw[0]<<8));
                block.push_back((float)sample/32768.0f);
                begin = 1;
                have_carry = false;
            }
            const size_t usable = (got-begin)&~(size_t)1;
            for (size_t i=begin; i<begin+usable; i+=2) {
                const int16_t sample = (int16_t)((uint16_t)raw[i] |
                                                 ((uint16_t)raw[i+1]<<8));
                block.push_back((float)sample/32768.0f);
            }
            if (begin+usable < got) {
                carry = raw[got-1];
                have_carry = true;
            }
            if (!block.empty() && !push(block)) break;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            read_error_ = ferror(stdin) != 0;
            incomplete_byte_ = have_carry;
            finished_ = true;
        }
        data_ready_.notify_all();
    }

    size_t read_samples_;
    std::vector<float> data_;
    size_t head_ = 0;
    size_t size_ = 0;
    size_t high_watermark_ = 0;
    mutable std::mutex mu_;
    std::condition_variable data_ready_, space_ready_;
    std::thread reader_;
    bool finished_ = false;
    bool read_error_ = false;
    bool incomplete_byte_ = false;
    bool stop_requested_ = false;
};

// ----------------------------- 后台推理工作线程 -----------------------------
// 主线程只负责持续收音频和构造 Job；Worker 串行调用 NanoRuntime，避免模型上下文并发访问。
// 队列刻意保持很小：partial 强调“新鲜度”，final 强调“绝不能丢”。
class Worker {
public:
    Worker(NanoRuntime & runtime, const Options & options)
        : runtime_(runtime), options_(options), thread_(&Worker::run, this) {}
    ~Worker() { stop(); }

    // 提交临时结果任务。若队列中已有尚未执行的 partial，则直接用更新的音频快照替换它。
    // 这样在推理速度落后于输入速度时不会积压大量已经过期的 partial。
    void submit_partial(Job job) {
        std::lock_guard<std::mutex> lock(mu_);
        // At most one queued partial is useful. If the worker is busy, replace
        // the stale queued partial with the newest audio snapshot.
        // 最多保留一个排队中的 partial；正在执行的任务不在 queue_ 中，因此不会被替换。
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if (!it->final) { *it = std::move(job); cv_.notify_one(); return; }
        }
        if (queue_.size() < 2) { queue_.push_back(std::move(job)); cv_.notify_one(); }
        else ++dropped_partial_;
    }

    // 提交 final。先删除队列中的 stale partial，再等待队列腾出空间。
    // final 表示一个 utterance 的闭合边界，因此不能像 partial 一样静默丢弃。
    void submit_final(Job job) {
        std::unique_lock<std::mutex> lock(mu_);
        // A final closes an utterance.  Any queued partial for that utterance
        // is stale and must not be allowed to overtake the final event.
        for (auto it = queue_.begin(); it != queue_.end();) {
            if (!it->final) { it = queue_.erase(it); ++dropped_partial_; }
            else ++it;
        }
        // 队列上限为 2，形成有界背压，防止 final 在极端慢推理情况下无限堆积内存。
        cv_space_.wait(lock, [&] { return stopping_ || queue_.size() < 2; });
        if (!stopping_) { queue_.push_back(std::move(job)); cv_.notify_one(); }
    }

    // 等待“队列为空且当前没有正在执行的任务”，用于 EOF 后确保全部结果已输出。
    void wait_idle() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_idle_.wait(lock, [&] { return queue_.empty() && !busy_; });
    }

    uint64_t dropped_partial() const {
        std::lock_guard<std::mutex> lock(mu_); return dropped_partial_;
    }

    // 累计所有成功 ASR 窗口推理耗时；不包含模型加载、音频读取和队列等待时间。
    int64_t total_inference_us() const {
        std::lock_guard<std::mutex> lock(mu_); return total_inference_us_;
    }

    // 幂等停止：设置 stopping_，唤醒所有可能阻塞的条件变量并 join 工作线程。
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all(); cv_space_.notify_all(); cv_idle_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    // 将一次推理结果输出到 stdout。JSONL 模式下每个事件严格占一行，便于管道程序逐行消费。
    void emit(const Job & job, const std::string & text, int64_t infer_us) {
        if (options_.output == "text") {
            if (job.final) {
                // text 模式使用 final 窗口的起止时间，格式为 mm:ss-mm:ss。
                const int64_t total_seconds = job.begin_ms / 1000;
                const int64_t minutes = total_seconds / 60;
                const int64_t seconds = total_seconds % 60;
                const int64_t end_total_seconds = job.end_ms / 1000;
                const int64_t end_minutes = end_total_seconds / 60;
                const int64_t end_seconds = end_total_seconds % 60;
                printf("%02lld:%02lld-%02lld:%02lld %s\n",
                       (long long)minutes, (long long)seconds,
                       (long long)end_minutes, (long long)end_seconds, text.c_str());
                fflush(stdout);
            }
            return;
        }
        const char * type = job.final ? "final" : "partial";
        // 只对 final 做跨分段合并；partial 是当前窗口的即时假设，不写入累计文本。
        if (job.final) merged_text_ = merge_text(merged_text_, text);
        const std::string & shown = job.final ? merged_text_ : text;
        printf("{\"type\":\"%s\",\"utterance_id\":%llu,\"revision\":%llu,\"begin_ms\":%lld,\"end_ms\":%lld,\"text\":\"%s\",\"inference_ms\":%lld",
            type, (unsigned long long)job.utterance, (unsigned long long)job.revision,
            (long long)job.begin_ms, (long long)job.end_ms, json_escape(shown).c_str(),
            (long long)(infer_us / 1000));
        if (job.final) printf(",\"reason\":\"%s\"", json_escape(job.reason).c_str());
        printf("}\n"); fflush(stdout);
    }

    // Worker 线程主循环：取任务 -> 解锁 -> 推理/输出 -> 再加锁更新 busy_。
    // 推理期间不持有 mu_，因此主线程仍可提交/替换队列中的任务。
    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) break;
                // 任务移出队列后标记 busy_。此时可立即通知可能等待队列空间的 submit_final。
                job = std::move(queue_.front()); queue_.pop_front(); busy_ = true;
                cv_space_.notify_all();
            }
            std::string text, error; int64_t elapsed = 0;
            if (runtime_.transcribe(job.samples, text, elapsed, error)) {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    total_inference_us_ += elapsed;
                }
                emit(job, text, elapsed);
            }
            else fprintf(stderr, "stream: inference failed for utterance %llu: %s\n",
                         (unsigned long long)job.utterance, error.c_str());
            {
                std::lock_guard<std::mutex> lock(mu_); busy_ = false;
                if (queue_.empty()) cv_idle_.notify_all();
            }
        }
    }

    NanoRuntime & runtime_;
    const Options & options_;
    mutable std::mutex mu_;
    std::condition_variable cv_, cv_space_, cv_idle_;
    std::deque<Job> queue_;
    std::thread thread_;
    bool stopping_ = false, busy_ = false;
    uint64_t dropped_partial_ = 0;
    int64_t total_inference_us_ = 0;
    std::string merged_text_;
};

// 根据当前 segment 状态决定是否真正生成 Job。
// 普通调用仅在达到 next_partial_ms 时提交 partial；force_final=true 时无条件提交 final。
static bool submit_block(const Options & o, Worker & worker, std::vector<float> & segment,
                         int64_t segment_start_ms, uint64_t utterance, uint64_t & revision,
                         int64_t & next_partial_ms, int64_t audio_end_ms, bool force_final,
                         const char * reason) {
    // 模型最小音频长度不足时暂不提交。
    if ((int)segment.size() < WINLEN) return true;
    // partial 使用“时间门控”而不是每个输入 block 都推理，控制计算量。
    if (!force_final && audio_end_ms < next_partial_ms) return true;
    Job job;
    // 复制当前完整 segment，形成不可变任务快照；随后主线程可以安全继续修改 segment。
    job.samples = segment;
    job.begin_ms = segment_start_ms;
    job.end_ms = audio_end_ms;
    job.utterance = utterance;
    job.revision = ++revision;
    job.final = force_final;
    job.reason = reason;
    if (force_final) worker.submit_final(std::move(job));
    else {
        worker.submit_partial(std::move(job));
        // 下次 partial 以“本次实际提交时刻 + 间隔”为基准，避免高频重复提交。
        next_partial_ms = audio_end_ms + o.partial_ms;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    // ----------------------------- 1. 参数与输入准备 -----------------------------
    Options opt;
    if (!parse_options(argc, argv, opt)) { stream_usage(argv[0]); return 2; }
    // arecord starts producing samples as soon as the pipeline is launched.
    // Start draining stdin before loading either model; otherwise the several-
    // second model initialization alone is enough to fill the pipe and trigger
    // an ALSA overrun before the program can print its ready event.
    const size_t block_samples = (size_t)opt.block_ms*16;
    PcmInputRing input_ring(block_samples, (size_t)opt.audio_buffer_ms*16);
    input_ring.start();
    // ----------------------------- 2. 初始化模型与工作线程 -----------------------------
    NanoRuntime runtime(opt);
    std::string error;
    const int64_t init_start = ggml_time_us();
    if (!runtime.init(error)) {
        input_ring.stop(); input_ring.join();
        fprintf(stderr, "stream: %s\n", error.c_str()); return 1;
    }
    const bool streaming_vad_input = !opt.vad.empty();
    FunASRStreamingVad streaming_vad;
    FunASRStreamingVadConfig streaming_vad_config;
    streaming_vad_config.max_segment_ms = opt.vad_maxseg_ms;
    streaming_vad_config.chunk_ms = opt.vad_chunk_ms;
    streaming_vad_config.speech_noise_threshold = opt.vad_speech_noise_thres;
    streaming_vad_config.max_end_silence_ms = opt.vad_max_end_silence_ms;
    if (streaming_vad_input && !streaming_vad.init(opt.vad, opt.vad_threads,
                                                   streaming_vad_config)) {
        input_ring.stop(); input_ring.join();
        fprintf(stderr, "stream: failed to initialize streaming VAD: %s\n", opt.vad.c_str());
        return 1;
    }
    // Start processing timing after runtime/model initialization.
    int64_t processing_start_us = ggml_time_us();
    // ready 是协议握手事件：通知上游模型已经加载完成，可以开始稳定接收结果。
    printf("{\"type\":\"ready\",\"sample_rate\":16000,\"mode\":\"%s\",\"model_loaded_ms\":%lld,\"asr_threads\":%d,\"vad_threads\":%d,\"audio_buffer_ms\":%d}\n",
           streaming_vad_input ? "streaming-vad" : "fixed-window",
           (long long)((ggml_time_us() - init_start) / 1000),
           opt.asr_threads, opt.vad_threads, opt.audio_buffer_ms);
    fflush(stdout);

    Worker worker(runtime, opt);
    // 采样率固定 16 kHz，因此 1 ms = 16 samples。
    const size_t segment_samples = (size_t)opt.segment_ms * 16;
    const size_t overlap_samples = (size_t)opt.overlap_ms * 16;
    // segment 是当前 utterance 的滚动音频窗口；容量额外留一个 block，减少扩容。
    std::vector<float> segment;
    segment.reserve(segment_samples + block_samples);
    // segment_start_ms 对应 segment[0] 的全局时间；next_partial_ms 是下一次临时结果门限。
    int64_t segment_start_ms = 0, next_partial_ms = opt.partial_ms;
    // utterance 每产生一个 final 后递增；revision 在同一 utterance 内从 1 开始递增。
    uint64_t utterance = 1, revision = 0;
    // Keep the authoritative input position in samples.  fread is allowed to
    // return arbitrary (including non-millisecond-aligned) block sizes, so
    // accumulating a truncated millisecond value once per block drifts over a
    // long-running stdin session.
    int64_t audio_samples = 0;
    bool vad_active = false;
    std::vector<float> vad_history;
    int64_t vad_history_begin = 0;

    auto finish_stream_vad = [&](int end_ms) {
        const int64_t wanted = std::max<int64_t>(0, ((int64_t)end_ms - segment_start_ms) * 16);
        if ((int64_t)segment.size() > wanted) segment.resize((size_t)wanted);
        if (segment.size() >= WINLEN)
            submit_block(opt, worker, segment, segment_start_ms, utterance, revision,
                         next_partial_ms, end_ms, true, "vad");
        segment.clear(); vad_active = false; segment_start_ms = end_ms;
        ++utterance; revision = 0; next_partial_ms = end_ms + opt.partial_ms;
    };

    auto handle_stream_vad = [&](const std::vector<FunASRStreamingVadEvent> & events) {
        for (const auto & event : events) {
            if (event.type == FunASRStreamingVadEvent::SpeechStart) {
                const int64_t wanted = (int64_t)event.begin_ms * 16;
                const int64_t begin = std::max(wanted, vad_history_begin);
                const int64_t end = vad_history_begin + (int64_t)vad_history.size();
                const int64_t clipped = std::min(begin, end);
                segment.assign(vad_history.begin() + (clipped - vad_history_begin), vad_history.end());
                segment_start_ms = clipped / 16; next_partial_ms = segment_start_ms + opt.partial_ms;
                vad_active = true;
            } else if (vad_active) {
                finish_stream_vad(event.end_ms);
            }
        }
        return true;
    };

    // ----------------------------- 3. 统一的流式消费逻辑 -----------------------------
    // 无论数据来自文件还是 stdin，最终都进入 consume()，按 block_samples 粒度推进时间轴。
    auto consume = [&](const float * p, size_t n) {
        size_t at = 0;
        while (at < n) {
            size_t take = std::min(block_samples, n - at);
            const float * block = p + at;
            at += take;
            audio_samples += (int64_t)take;
            const int64_t audio_ms = audio_samples * 1000 / 16000;
            if (streaming_vad_input) {
                vad_history.insert(vad_history.end(), block, block + take);
                if (vad_active) segment.insert(segment.end(), block, block + take);
                std::vector<FunASRStreamingVadEvent> events;
                if (!streaming_vad.accept(block, take, false, events)) {
                    fprintf(stderr, "stream: streaming VAD inference failed\n"); return false;
                }
                handle_stream_vad(events);
                if (vad_active) submit_block(opt, worker, segment, segment_start_ms, utterance,
                                             revision, next_partial_ms, audio_ms, false, "");
                if (!vad_active && vad_history.size() > 1U * 16000U) {
                    const size_t drop = vad_history.size() - 1U * 16000U;
                    vad_history.erase(vad_history.begin(), vad_history.begin() + drop);
                    vad_history_begin += (int64_t)drop;
                }
            } else {
                segment.insert(segment.end(), block, block + take);
            }

            if (opt.vad.empty() && segment.size() >= segment_samples) {
                submit_block(opt, worker, segment, segment_start_ms, utterance, revision,
                             next_partial_ms, audio_ms, true, "max_segment");
                // 保留末尾 overlap 作为下一窗口开头，使分割点两侧都拥有一定上下文。
                if (overlap_samples > 0 && overlap_samples < segment.size()) {
                    std::vector<float> tail(segment.end() - overlap_samples, segment.end());
                    segment.swap(tail);
                    // 新窗口的起点需要回退 overlap 时长，保证输出 begin_ms 与真实全局时间一致。
                    segment_start_ms = audio_ms - (int64_t)segment.size() * 1000 / 16000;
                } else segment.clear();
                // 新 utterance 的 revision 重新计数；partial 门限从当前音频尾部重新起算。
                ++utterance; revision = 0;
                next_partial_ms = audio_ms + opt.partial_ms;
            }
        }
        return true;
    };

    bool input_ok = true;

    // ----------------------------- 4A. stdin 原始 PCM 输入 -----------------------------
    // reader 线程持续排空 arecord 管道；当前线程只消费环形缓冲并执行 VAD。
    std::vector<float> input_block;
    while (input_ring.pop(input_block)) {
        if (!consume(input_block.data(), input_block.size())) {
            input_ok = false;
            input_ring.stop();
            break;
        }
    }
    input_ring.join();
    if (!input_ring.ok()) {
        fprintf(stderr, "stream: stdin read failed\n");
        input_ok = false;
    }
    if (input_ring.incomplete_byte())
        fprintf(stderr, "stream: ignoring incomplete trailing PCM byte\n");
    const int64_t capture_buffer_peak_ms =
        (int64_t)input_ring.high_watermark_samples()*1000/16000;
    // ----------------------------- 5. EOF 收尾 -----------------------------
    if (!input_ok) {
        worker.wait_idle();
        worker.stop();
        return 1;
    }
    if (streaming_vad_input) {
        std::vector<FunASRStreamingVadEvent> events;
        if (!streaming_vad.accept(nullptr, 0, true, events)) {
            fprintf(stderr, "stream: streaming VAD finalization failed\n");
            worker.wait_idle(); worker.stop(); return 1;
        }
        handle_stream_vad(events);
    }
    // 无 VAD 模式下，输入结束后尚未达到 segment_ms 的尾部窗口需要作为 final 提交。
    if (opt.vad.empty() && segment.size() >= WINLEN) {
        submit_block(opt, worker, segment, segment_start_ms, utterance, revision,
                     next_partial_ms, audio_samples * 1000 / 16000, true, "eof");
    }
    // 等待尾部 final（以及其前面的任务）全部推理和输出完毕，再打印 done。
    worker.wait_idle();
    // done 给出总 utterance 计数、丢弃的 partial 数、音频时长、累计 ASR 推理时间和 RTF。
    const int64_t total_audio_ms = audio_samples * 1000 / 16000;
    const int64_t total_inference_us = worker.total_inference_us();
    const int64_t total_processing_us = std::max<int64_t>(0, ggml_time_us() - processing_start_us);
    // RTF uses complete wall-clock processing time after model initialization.
    // It includes audio loading, VAD, queueing, and ASR, but excludes model loading.
    const double rtf = total_audio_ms > 0 ?
        (double)total_processing_us / ((double)total_audio_ms * 1000.0) : 0.0;
    if (opt.output == "jsonl") {
        printf("{\"type\":\"done\",\"utterances\":%llu,\"partial_dropped\":%llu,\"audio_ms\":%lld,\"capture_buffer_peak_ms\":%lld,\"total_processing_ms\":%lld,\"total_inference_ms\":%lld,\"rtf\":%.6f}\n",
               (unsigned long long)utterance, (unsigned long long)worker.dropped_partial(),
               (long long)total_audio_ms, (long long)capture_buffer_peak_ms,
               (long long)(total_processing_us / 1000),
               (long long)(total_inference_us / 1000), rtf);
        fflush(stdout);
    } else {
        // text 模式保持 stdout 纯文本，将统计信息写到 stderr。
        fprintf(stderr, "stream: total_processing_ms=%lld total_inference_ms=%lld rtf=%.6f audio_ms=%lld capture_buffer_peak_ms=%lld\n",
                (long long)(total_processing_us / 1000),
                (long long)(total_inference_us / 1000), rtf, (long long)total_audio_ms,
                (long long)capture_buffer_peak_ms);
    }
    // 显式停止并 join；析构函数也会调用 stop()，但 stop() 是幂等的。
    worker.stop();
    return 0;
}
