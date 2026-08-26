#include "funasr_vad_stream.h"

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

using funasr_vad_stream_impl::Endpoint;
using funasr_vad_stream_impl::Frontend;

static void test_fsmn_layer_cache_matches_full_history() {
    const int frames = 47;
    const int projection_dim = 7;
    const int lorder = 5;
    const int history_frames = lorder-1;

    std::vector<float> projected((size_t)frames*projection_dim);
    std::vector<float> kernel((size_t)lorder*projection_dim);
    for (size_t i = 0; i < projected.size(); ++i)
        projected[i] = std::sin((float)i*0.071f)*0.7f;
    for (size_t i = 0; i < kernel.size(); ++i)
        kernel[i] = std::cos((float)i*0.113f)*0.2f;

    // Reference: one full causal FSMN pass with zero padding on the left.
    std::vector<float> expected = projected;
    for (int t = 0; t < frames; ++t) {
        for (int j = 0; j < lorder; ++j) {
            const int source_frame = t-history_frames+j;
            if (source_frame < 0) continue;
            for (int d = 0; d < projection_dim; ++d) {
                expected[(size_t)t*projection_dim+d] +=
                    projected[(size_t)source_frame*projection_dim+d]*
                    kernel[(size_t)j*projection_dim+d];
            }
        }
    }

    // Incremental pass: use irregular chunks and retain only lorder-1 projected
    // frames, exactly as Model::infer does independently for every FSMN layer.
    std::vector<float> history((size_t)history_frames*projection_dim, 0.0f);
    std::vector<float> actual;
    const int chunk_sizes[] = {1, 6, 2, 13, 4, 17, 4};
    int offset = 0;
    for (int chunk_frames : chunk_sizes) {
        assert(offset+chunk_frames <= frames);
        std::vector<float> combined = history;
        combined.insert(combined.end(),
                        projected.begin()+(size_t)offset*projection_dim,
                        projected.begin()+(size_t)(offset+chunk_frames)*projection_dim);
        for (int t = 0; t < chunk_frames; ++t) {
            for (int d = 0; d < projection_dim; ++d) {
                float value = projected[(size_t)(offset+t)*projection_dim+d];
                for (int j = 0; j < lorder; ++j) {
                    value += combined[(size_t)(t+j)*projection_dim+d]*
                             kernel[(size_t)j*projection_dim+d];
                }
                actual.push_back(value);
            }
        }
        history.assign(combined.end()-(size_t)history_frames*projection_dim,
                       combined.end());
        offset += chunk_frames;
    }
    assert(offset == frames);
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        assert(std::fabs(actual[i]-expected[i]) < 1e-6f);
}

static void append_output(Frontend::Output output, std::vector<float> & features,
                          std::vector<float> & decibels, int & frames,
                          int lfr_n = 1) {
    features.insert(features.end(), output.features.begin(), output.features.end());
    if (output.frames) {
        const size_t frame_shift = (size_t)lfr_n*160;
        assert(output.waveform.size() == (size_t)(output.frames-1)*frame_shift+400);
        for (int frame=0; frame<output.frames; ++frame) {
            double energy = 0.0;
            for (int i=0; i<400; ++i) {
                const float sample = output.waveform[(size_t)frame*frame_shift+i];
                energy += (double)sample*sample;
            }
            decibels.push_back(10.0f*std::log10((float)energy+0.000001f));
        }
    }
    frames += output.frames;
}

static void test_frontend_wide_lfr_stride() {
    std::vector<float> waveform(12031);
    for (size_t i=0; i<waveform.size(); ++i)
        waveform[i] = 0.3f*std::sin((float)i*0.011f);

    Frontend whole(5, 3);
    auto expected = whole.push(waveform.data(), waveform.size(), true);
    std::vector<float> expected_features, expected_decibels;
    int expected_frames = 0;
    append_output(std::move(expected), expected_features, expected_decibels,
                  expected_frames, 3);

    Frontend chunked(5, 3);
    std::vector<float> features, decibels;
    int frames = 0;
    const size_t cuts[] = {1, 399, 160, 777, 2049, 17, 4096};
    size_t offset = 0;
    for (size_t count : cuts) {
        append_output(chunked.push(waveform.data()+offset, count, false),
                      features, decibels, frames, 3);
        offset += count;
    }
    append_output(chunked.push(waveform.data()+offset, waveform.size()-offset, true),
                  features, decibels, frames, 3);
    assert(frames == expected_frames);
    assert(features == expected_features);
    assert(decibels == expected_decibels);
}

static void test_frontend_chunk_cache_and_alignment() {
    std::vector<float> waveform(4097);
    for (size_t i=0; i<waveform.size(); ++i)
        waveform[i] = 0.25f*std::sin((float)i*0.017f);

    Frontend whole(5, 1);
    auto expected = whole.push(waveform.data(), waveform.size(), true);
    assert(expected.features.size() == (size_t)expected.frames*5*80);
    assert(expected.waveform.size() == (size_t)(expected.frames-1)*160+400);
    std::vector<float> expected_features, expected_decibels;
    int expected_frames = 0;
    append_output(std::move(expected), expected_features, expected_decibels,
                  expected_frames);

    Frontend chunked(5, 1);
    std::vector<float> features, decibels;
    int frames = 0;
    const size_t cuts[] = {137, 961, 73, 1400, 511};
    size_t offset = 0;
    for (size_t count : cuts) {
        append_output(chunked.push(waveform.data()+offset, count, false),
                      features, decibels, frames);
        offset += count;
    }
    append_output(chunked.push(waveform.data()+offset, waveform.size()-offset, true),
                  features, decibels, frames);
    assert(frames == expected_frames);
    assert(features == expected_features);
    assert(decibels == expected_decibels);
}

static void push_frames(Endpoint & endpoint, int count, bool speech,
                        std::vector<FunASRStreamingVadEvent> & events,
                        float decibel = 0.0f) {
    const int output_dim = 2;
    std::vector<float> scores((size_t)count*output_dim);
    std::vector<float> decibels((size_t)count, decibel);
    for (int i=0; i<count; ++i) {
        scores[(size_t)i*output_dim] = speech ? 0.01f : 0.99f;
        scores[(size_t)i*output_dim+1] = speech ? 0.99f : 0.01f;
    }
    assert(endpoint.push(scores, output_dim, decibels, events));
}

static void test_protocol_endpoint_delay_and_absolute_timeline() {
    FunASRStreamingVadConfig config;
    config.max_end_silence_ms = 300;
    config.max_segment_ms = 60000;
    Endpoint endpoint(config, 10);
    assert(endpoint.current_max_end_silence_ms() == 300);
    std::vector<FunASRStreamingVadEvent> events;

    push_frames(endpoint, 30, false, events);
    assert(events.empty());
    push_frames(endpoint, 30, true, events);
    assert(events.size() == 1);
    assert(events[0].type == FunASRStreamingVadEvent::SpeechStart);
    assert(events[0].begin_ms == 40 && events[0].end_ms == -1);

    push_frames(endpoint, 19, false, events);
    assert(events.size() == 1); // Speech2Sil alone must not close the segment.
    push_frames(endpoint, 1, false, events);
    assert(events.size() == 2);
    assert(events[1].type == FunASRStreamingVadEvent::SpeechEnd);
    assert(events[1].begin_ms == -1 && events[1].end_ms == 760);

    push_frames(endpoint, 15, true, events);
    assert(events.size() == 3);
    assert(events[2].type == FunASRStreamingVadEvent::SpeechStart);
    assert(events[2].begin_ms == 760); // ResetDetection did not reset time.
    endpoint.finish(events);
    assert(events.size() == 4);
    assert(events[3].type == FunASRStreamingVadEvent::SpeechEnd);
    assert(events[3].begin_ms == -1 && events[3].end_ms == 950);
}

static void test_decibel_gate_and_fixed_streaming_silence() {
    FunASRStreamingVadConfig config;
    config.max_segment_ms = 60000;
    Endpoint quiet(config, 10);
    std::vector<FunASRStreamingVadEvent> events;
    push_frames(quiet, 30, true, events, -120.0f);
    assert(events.empty());

    Endpoint fixed(config, 10);
    push_frames(fixed, 15, true, events);
    assert(fixed.current_max_end_silence_ms() == 800);
    push_frames(fixed, 500, true, events);
    assert(fixed.current_max_end_silence_ms() == 800);
}

int main() {
    test_fsmn_layer_cache_matches_full_history();
    test_frontend_chunk_cache_and_alignment();
    test_frontend_wide_lfr_stride();
    test_protocol_endpoint_delay_and_absolute_timeline();
    test_decibel_gate_and_fixed_streaming_silence();
    return 0;
}
