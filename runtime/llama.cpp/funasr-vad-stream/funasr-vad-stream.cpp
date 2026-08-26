// Standalone streaming FSMN-VAD: PCM/audio -> streaming start/end events.
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad_stream.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Options {
    std::string model;
    std::string audio;
    bool stdin_s16le = false;
    bool realtime = false;
    int input_block_ms = 100;
    int threads = 8;
    int max_segment_ms = 30000;
    int chunk_ms = 60;
    int max_end_silence_ms = 800;
    float speech_noise_threshold = 0.5f;
    std::string output = "jsonl";
};

void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s -m fsmn-vad.gguf (--audio file | --stdin-s16le) [options]\n\n"
        "  -m, --model PATH              FSMN-VAD GGUF model\n"
        "  -a, --audio PATH              audio file, converted to 16 kHz mono\n"
        "      --stdin-s16le             raw 16 kHz mono signed 16-bit LE PCM\n"
        "  -t, --threads N               VAD CPU threads (default: 8)\n"
        "      --input-block-ms N        stdin/file input block (default: 100)\n"
        "      --chunk-ms N              internal VAD chunk (default: 60)\n"
        "      --max-segment-ms N        maximum speech segment (default: 30000)\n"
        "      --max-end-silence-ms N    fixed endpoint silence (default: 800 ms)\n"
        "      --speech-noise-thres F    speech/noise threshold (default: 0.5)\n"
        "      --realtime                pace file input in real time\n"
        "      --output jsonl|text       event output (default: jsonl)\n"
        "  -h, --help                    show this help\n", argv0);
}

bool parse_options(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        auto value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", name); return nullptr; }
            return argv[++i];
        };
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) { usage(argv[0]); exit(0); }
        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) { const char *v=value(arg); if(!v)return false; o.model=v; }
        else if (!strcmp(arg, "-a") || !strcmp(arg, "--audio")) { const char *v=value(arg); if(!v)return false; o.audio=v; }
        else if (!strcmp(arg, "--stdin-s16le")) o.stdin_s16le=true;
        else if (!strcmp(arg, "--realtime")) o.realtime=true;
        else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) { const char *v=value(arg); if(!v)return false; o.threads=atoi(v); }
        else if (!strcmp(arg, "--input-block-ms")) { const char *v=value(arg); if(!v)return false; o.input_block_ms=atoi(v); }
        else if (!strcmp(arg, "--chunk-ms")) { const char *v=value(arg); if(!v)return false; o.chunk_ms=atoi(v); }
        else if (!strcmp(arg, "--max-segment-ms")) { const char *v=value(arg); if(!v)return false; o.max_segment_ms=atoi(v); }
        else if (!strcmp(arg, "--max-end-silence-ms")) { const char *v=value(arg); if(!v)return false; o.max_end_silence_ms=atoi(v); }
        else if (!strcmp(arg, "--speech-noise-thres")) { const char *v=value(arg); if(!v)return false; o.speech_noise_threshold=strtof(v,nullptr); }
        else if (!strcmp(arg, "--output")) { const char *v=value(arg); if(!v)return false; o.output=v; }
        else { fprintf(stderr, "unknown option: %s\n", arg); return false; }
    }
    return !o.model.empty() && ((o.audio.empty() && o.stdin_s16le) || (!o.audio.empty() && !o.stdin_s16le)) &&
        o.threads > 0 && o.input_block_ms > 0 && o.chunk_ms > 0 && o.max_segment_ms > 0 &&
        o.max_end_silence_ms > 0 && o.speech_noise_threshold >= 0.0f &&
        o.speech_noise_threshold <= 1.0f && (o.output == "jsonl" || o.output == "text") &&
        !(o.stdin_s16le && o.realtime);
}

void emit_events(const std::vector<FunASRStreamingVadEvent> &events, const Options &o) {
    for (const auto &event : events) {
        if (o.output == "text") {
            if (event.type == FunASRStreamingVadEvent::SpeechStart)
                printf("[%d, -1]\n", event.begin_ms);
            else
                printf("[-1, %d]\n", event.end_ms);
        } else if (event.type == FunASRStreamingVadEvent::SpeechStart) {
            printf("{\"type\":\"speech_start\",\"begin_ms\":%d,\"end_ms\":-1}\n", event.begin_ms);
        } else {
            printf("{\"type\":\"speech_end\",\"begin_ms\":-1,\"end_ms\":%d}\n", event.end_ms);
        }
    }
    if (!events.empty()) fflush(stdout);
}
}

int main(int argc, char **argv) {
    Options o;
    if (!parse_options(argc, argv, o)) { usage(argv[0]); return 2; }

    std::vector<float> source;
    if (!o.stdin_s16le && !funasr_load_audio_16k_mono(o.audio.c_str(), source)) {
        fprintf(stderr, "read audio failed\n"); return 1;
    }
    FunASRStreamingVadConfig config;
    config.max_segment_ms = o.max_segment_ms;
    config.chunk_ms = o.chunk_ms;
    config.max_end_silence_ms = o.max_end_silence_ms;
    config.speech_noise_threshold = o.speech_noise_threshold;
    FunASRStreamingVad vad;
    if (!vad.init(o.model, o.threads, config)) { fprintf(stderr, "failed to initialize VAD\n"); return 1; }

    if (o.output == "jsonl") {
        printf("{\"type\":\"ready\",\"sample_rate\":16000,\"mode\":\"streaming-vad\"}\n");
        fflush(stdout);
    }
    const size_t block_samples = (size_t)o.input_block_ms * 16;
    std::vector<float> block;
    int64_t audio_samples = 0;
    auto infer_begin = std::chrono::steady_clock::now();
    auto consume = [&](const float *samples, size_t count) -> bool {
        std::vector<FunASRStreamingVadEvent> events;
        if (!vad.accept(samples, count, false, events)) return false;
        emit_events(events, o);
        audio_samples += (int64_t)count;
        return true;
    };

    bool ok = true;
    if (o.stdin_s16le) {
        std::vector<unsigned char> raw(block_samples*2);
        unsigned char carry = 0; bool have_carry = false;
        for (;;) {
            const size_t got = fread(raw.data(), 1, raw.size(), stdin);
            if (!got) break;
            size_t begin = 0;
            block.clear();
            if (have_carry) {
                block.push_back((float)(int16_t)((uint16_t)carry | ((uint16_t)raw[0]<<8))/32768.0f);
                begin = 1; have_carry = false;
            }
            const size_t usable = (got-begin)&~(size_t)1;
            for (size_t i=begin; i<begin+usable; i+=2)
                block.push_back((float)(int16_t)((uint16_t)raw[i] | ((uint16_t)raw[i+1]<<8))/32768.0f);
            if (begin+usable<got) { carry=raw[got-1]; have_carry=true; }
            if (!consume(block.data(), block.size())) { ok=false; break; }
        }
        if (have_carry) fprintf(stderr, "warning: ignoring incomplete trailing PCM byte\n");
    } else {
        for (size_t offset=0; offset<source.size() && ok; offset+=block_samples) {
            const size_t count=std::min(block_samples, source.size()-offset);
            ok=consume(source.data()+offset, count);
            if (o.realtime) std::this_thread::sleep_for(std::chrono::milliseconds(std::max<size_t>(1,count/16)));
        }
    }
    std::vector<FunASRStreamingVadEvent> final_events;
    if (ok) ok=vad.accept(nullptr, 0, true, final_events);
    if (ok) emit_events(final_events, o);
    const double inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-infer_begin).count();
    const double audio_ms=(double)audio_samples*1000.0/16000.0;
    const double rtf=audio_ms>0?inference_ms/audio_ms:0.0;
    if (o.output == "jsonl") {
        printf("{\"type\":\"done\",\"audio_ms\":%.3f,\"inference_ms\":%.3f,\"rtf\":%.6f}\n", audio_ms, inference_ms, rtf);
        fflush(stdout);
    } else fprintf(stderr, "[vad-stream] audio_ms=%.3f inference_ms=%.3f rtf=%.6f\n", audio_ms, inference_ms, rtf);
    return ok ? 0 : 1;
}
