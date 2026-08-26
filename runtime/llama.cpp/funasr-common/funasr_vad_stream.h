#pragma once
#include "funasr_vad.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <memory>

struct FunASRStreamingVadEvent {
    enum Type { SpeechStart, SpeechEnd } type;
    // Streaming protocol: start=[begin_ms,-1], end=[-1,end_ms].
    int begin_ms = -1, end_ms = -1;
};

struct FunASRStreamingVadConfig {
    int max_segment_ms = 30000;
    int chunk_ms = 60;
    float speech_noise_threshold = 0.5f;
    // Fixed endpoint silence. The streaming VAD no longer uses a dynamic schedule.
    int max_end_silence_ms = 800;
    float decibel_threshold = -100.0f;
    float snr_threshold = -100.0f;
};

namespace funasr_vad_stream_impl {
class Frontend {
public:
    struct Output {
        std::vector<float> features;
        // Exact PCM span aligned to the emitted feature frames. For N frames
        // its size is (N-1)*LFR-stride + one 25 ms analysis window.
        std::vector<float> waveform;
        int frames = 0;
    };

    Frontend(int m,int n):m_(m),n_(n),left_((m-1)/2),right_(m-1-left_){
        namespace v=funasr_vad_impl; window_.resize(v::WINLEN);
        for(int i=0;i<v::WINLEN;++i)window_[i]=.54f-.46f*std::cos(2*M_PI*i/(v::WINLEN-1));
        int nb=v::NFFT/2+1; filters_.assign(v::NMEL,std::vector<float>(nb));
        float bw=(float)v::FS/v::NFFT,ml=v::melf(v::LOWF),mh=v::melf(v::HIGHF),dm=(mh-ml)/(v::NMEL+1);
        for(int m=0;m<v::NMEL;++m){float l=ml+m*dm,c=ml+(m+1)*dm,r=ml+(m+2)*dm;
            for(int k=0;k<nb;++k){float f=v::melf(bw*k);if(f>l&&f<r)filters_[m][k]=f<=c?(f-l)/(c-l):(r-f)/(r-c);}}
    }
    Output push(const float *p,size_t n,bool final){
        namespace v=funasr_vad_impl;if(n){raw_.insert(raw_.end(),p,p+n);samples_+=n;}
        while(next_sample_+v::WINLEN<=samples_){size_t off=(size_t)(next_sample_-raw_base_);const float *s=raw_.data()+off;feats_.push_back(frame(s));waveforms_.emplace_back(s,s+v::WINLEN);if(first_.empty())first_=feats_.back();++nfeat_;next_sample_+=v::SHIFT;}
        size_t drop=(size_t)(next_sample_-raw_base_);if(drop){raw_.erase(raw_.begin(),raw_.begin()+drop);raw_base_=next_sample_;}
        Output out;while(next_center_<nfeat_&&(final||next_center_+right_<nfeat_)){
            for(int j=-left_;j<=right_;++j){int64_t x=next_center_+j;const auto *f=x<0?&first_:x>=nfeat_?&feats_.back():&feats_[(size_t)(x-feat_base_)];out.features.insert(out.features.end(),f->begin(),f->end());}
            const auto &waveform=waveforms_[(size_t)(next_center_-feat_base_)];
            // Keep the waveform on the same LFR timeline as the feature rows.
            // Copying each complete analysis window also handles lfr_n values
            // whose stride is wider than the 25 ms window (there is then a gap
            // between windows).  The former tail-append scheme was only correct
            // while n * SHIFT <= WINLEN.
            const size_t waveform_offset=(size_t)out.frames*(size_t)n_*v::SHIFT;
            out.waveform.resize(waveform_offset+(size_t)v::WINLEN);
            std::copy(waveform.begin(),waveform.end(),
                      out.waveform.begin()+(ptrdiff_t)waveform_offset);
            ++out.frames;next_center_+=n_;}
        int64_t keep=std::max<int64_t>(0,next_center_-left_);while(!feats_.empty()&&feat_base_<keep){feats_.pop_front();waveforms_.pop_front();++feat_base_;}
        return out;
    }
private:
    std::vector<float> frame(const float*s)const{namespace v=funasr_vad_impl;std::vector<float>re(v::NFFT),im(v::NFFT),fr(v::WINLEN),o(v::NMEL);double mean=0;
        for(int i=0;i<v::WINLEN;++i)mean+=s[i]*32768;mean/=v::WINLEN;for(int i=0;i<v::WINLEN;++i)fr[i]=s[i]*32768-(float)mean;
        for(int i=v::WINLEN-1;i>0;--i)fr[i]-=v::PREEMPH*fr[i-1];fr[0]-=v::PREEMPH*fr[0];for(int i=0;i<v::WINLEN;++i)re[i]=fr[i]*window_[i];v::fftc(re,im,v::NFFT);
        for(int m=0;m<v::NMEL;++m){float e=0;for(int k=0;k<v::NFFT/2+1;++k)if(filters_[m][k]>0)e+=filters_[m][k]*(re[k]*re[k]+im[k]*im[k]);o[m]=std::log(e>1.1920929e-7f?e:1.1920929e-7f);}return o;}
    int m_,n_,left_,right_;std::vector<float>window_;std::vector<std::vector<float>>filters_;std::vector<float>raw_;uint64_t raw_base_=0,next_sample_=0,samples_=0;
    std::deque<std::vector<float>>feats_,waveforms_;std::vector<float>first_;int64_t feat_base_=0,nfeat_=0,next_center_=0;
};

class Model {
public:
    ~Model() {
        graph_.reset();
        if (be_) ggml_backend_free(be_);
        if (m_.ctx) ggml_free(m_.ctx);
    }

    bool load(const std::string & path, int threads) {
        gguf_init_params ip = {false, &m_.ctx};
        gguf_context * g = gguf_init_from_file(path.c_str(), ip);
        if (!g) return false;
        auto rd = [&](const char * key, int fallback) {
            const int i = gguf_find_key(g, key);
            return i < 0 ? fallback : (int)gguf_get_val_u32(g, i);
        };
        id_ = rd("vad.input_dim", 400);
        pd_ = rd("vad.proj_dim", 128);
        nl_ = rd("vad.fsmn_layers", 4);
        lo_ = rd("vad.lorder", 20);
        od_ = rd("vad.output_dim", 248);
        lm_ = rd("vad.lfr_m", 5);
        ln_ = rd("vad.lfr_n", 1);
        for (int i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            m_.t[name] = ggml_get_tensor(m_.ctx, name);
        }
        gguf_free(g);
        auto need = [&](const std::string & name) { return m_.g(name) != nullptr; };
        bool ok = need("cmvn.shift") && need("cmvn.scale") &&
                  need("encoder.in_linear1.linear.weight") &&
                  need("encoder.in_linear2.linear.weight") &&
                  need("encoder.out_linear1.linear.weight") &&
                  need("encoder.out_linear2.linear.weight");
        for (int i = 0; i < nl_ && ok; ++i) {
            const std::string p = "encoder.fsmn." + std::to_string(i) + ".";
            ok = need(p+"linear.linear.weight") &&
                 need(p+"fsmn_block.conv_left.weight") &&
                 need(p+"affine.linear.weight");
        }
        if (!ok || id_ != lm_*funasr_vad_impl::NMEL || nl_ <= 0 || lo_ <= 0) return false;
        be_ = ggml_backend_cpu_init();
        threads_ = threads;
        layer_history_.assign((size_t)nl_,
                              std::vector<float>((size_t)(lo_-1)*pd_, 0.0f));
        return be_ != nullptr;
    }

    int lm() const { return lm_; }
    int ln() const { return ln_; }
    int od() const { return od_; }
    double ms() const { return ms_; }

    // True incremental FSMN inference. Each layer caches its own last
    // (lorder-1) projected frames. Only the new frames pass through the
    // input/FSMN/output linear layers.
    bool infer(std::vector<float> x0, int nt, std::vector<float> & out) {
        out.clear();
        if (!nt) return true;
        if (x0.size() != (size_t)nt*id_) return false;

        float * shift = (float *)m_.g("cmvn.shift")->data;
        float * scale = (float *)m_.g("cmvn.scale")->data;
        for (int t = 0; t < nt; ++t) {
            for (int d = 0; d < id_; ++d) {
                x0[(size_t)t*id_+d] = (x0[(size_t)t*id_+d]+shift[d])*scale[d];
            }
        }

        if (!graph_ || graph_->frames != nt) {
            graph_.reset();
            graph_ = build_graph(nt);
            if (!graph_) return false;
        }
        CachedGraph & cached = *graph_;
        ggml_backend_tensor_set(cached.input, x0.data(), 0,
                                ggml_nbytes(cached.input));
        const int history_frames = lo_-1;
        for (int i = 0; i < nl_; ++i) {
            if (history_frames > 0) {
                ggml_backend_tensor_set(cached.history_inputs[(size_t)i],
                    layer_history_[(size_t)i].data(), 0,
                    ggml_nbytes(cached.history_inputs[(size_t)i]));
            }
        }

        ggml_backend_cpu_set_n_threads(be_, threads_);
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = ggml_backend_graph_compute(be_, cached.graph) == GGML_STATUS_SUCCESS;
        ms_ += std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-begin).count();
        if (ok) {
            out.resize((size_t)od_*nt);
            ggml_backend_tensor_get(cached.output, out.data(), 0,
                                    ggml_nbytes(cached.output));
            for (int i = 0; i < nl_; ++i) {
                std::vector<float> newest((size_t)pd_*nt);
                ggml_backend_tensor_get(cached.projected_outputs[(size_t)i],
                                        newest.data(), 0,
                                        ggml_nbytes(cached.projected_outputs[(size_t)i]));
                auto & history = layer_history_[(size_t)i];
                history.insert(history.end(), newest.begin(), newest.end());
                const size_t keep = (size_t)history_frames*pd_;
                if (history.size() > keep) {
                    history.erase(history.begin(), history.end()-(ptrdiff_t)keep);
                }
            }
        }
        return ok;
    }

private:
    struct CachedGraph {
        ~CachedGraph() {
            if (allocator) ggml_gallocr_free(allocator);
            if (ctx) ggml_free(ctx);
        }
        int frames = 0;
        ggml_context * ctx = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<ggml_tensor *> history_inputs;
        std::vector<ggml_tensor *> projected_outputs;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t allocator = nullptr;
    };

    std::unique_ptr<CachedGraph> build_graph(int nt) {
        std::unique_ptr<CachedGraph> cached(new CachedGraph());
        cached->frames = nt;
        // The graph contains only a few hundred metadata objects. Keeping one
        // 8 MiB no-alloc context avoids rebuilding the stable 60 ms VAD graph
        // for every microphone block without retaining the old 64 MiB scratch
        // context per call.
        ggml_init_params cp = {(size_t)8*1024*1024, nullptr, true};
        cached->ctx = ggml_init(cp);
        if (!cached->ctx) return nullptr;
        ggml_context * c = cached->ctx;

        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, id_, nt);
        ggml_set_input(in);
        cached->input = in;
        ggml_tensor * h = funasr_vad_impl::lin(
            c, m_.g("encoder.in_linear1.linear.weight"),
            m_.g("encoder.in_linear1.linear.bias"), in);
        h = funasr_vad_impl::lin(
            c, m_.g("encoder.in_linear2.linear.weight"),
            m_.g("encoder.in_linear2.linear.bias"), h);
        h = ggml_relu(c, h);

        cached->history_inputs.reserve((size_t)nl_);
        cached->projected_outputs.reserve((size_t)nl_);
        const int history_frames = lo_-1;

        for (int i = 0; i < nl_; ++i) {
            const std::string p = "encoder.fsmn." + std::to_string(i) + ".";
            ggml_tensor * z = ggml_mul_mat(c, m_.g(p+"linear.linear.weight"), h);
            ggml_set_output(z);
            cached->projected_outputs.push_back(z);

            ggml_tensor * history = nullptr;
            if (history_frames > 0) {
                history = ggml_new_tensor_2d(c, GGML_TYPE_F32, pd_, history_frames);
                ggml_set_input(history);
            }
            cached->history_inputs.push_back(history);
            ggml_tensor * z_all = history_frames > 0
                ? ggml_concat(c, history, z, 1)
                : z;
            z_all = ggml_cont(c, z_all);

            ggml_tensor * fsmn = z;
            ggml_tensor * kernel = m_.g(p+"fsmn_block.conv_left.weight");
            for (int j = 0; j < lo_; ++j) {
                ggml_tensor * slice = ggml_view_2d(
                    c, z_all, pd_, nt, z_all->nb[1],
                    (size_t)j*z_all->nb[1]);
                ggml_tensor * weight = ggml_view_1d(
                    c, kernel, pd_, (size_t)j*kernel->nb[1]);
                fsmn = ggml_add(c, fsmn, ggml_mul(c, slice, weight));
            }
            h = ggml_relu(c, funasr_vad_impl::lin(
                c, m_.g(p+"affine.linear.weight"),
                m_.g(p+"affine.linear.bias"), fsmn));
        }

        h = funasr_vad_impl::lin(
            c, m_.g("encoder.out_linear1.linear.weight"),
            m_.g("encoder.out_linear1.linear.bias"), h);
        h = funasr_vad_impl::lin(
            c, m_.g("encoder.out_linear2.linear.weight"),
            m_.g("encoder.out_linear2.linear.bias"), h);
        h = ggml_soft_max(c, h);
        ggml_set_output(h);
        cached->output = h;

        cached->graph = ggml_new_graph_custom(c, 32768, false);
        ggml_build_forward_expand(cached->graph, h);
        cached->allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!cached->allocator ||
            !ggml_gallocr_alloc_graph(cached->allocator, cached->graph)) return nullptr;
        return cached;
    }

    funasr_vad_impl::vad m_;
    ggml_backend_t be_ = nullptr;
    int id_ = 0, pd_ = 0, nl_ = 0, lo_ = 0, od_ = 0, lm_ = 0, ln_ = 0;
    int threads_ = 4;
    std::vector<std::vector<float>> layer_history_;
    std::unique_ptr<CachedGraph> graph_;
    double ms_ = 0;
};

enum class FrameState { Silence, Speech };
enum class AudioChangeState { Speech2Speech, Speech2Sil, Sil2Sil, Sil2Speech };
enum class VadState { StartPointNotDetected, InSpeechSegment };

class WindowDetector {
public:
    explicit WindowDetector(int frame_ms)
        : window_(std::max(1, 200/frame_ms), 0),
          sil_to_speech_frames_(std::max(1, 150/frame_ms)),
          speech_to_sil_frames_(std::max(1, 150/frame_ms)) {}

    AudioChangeState detect(FrameState state) {
        const int current = state == FrameState::Speech ? 1 : 0;
        sum_ -= window_[position_];
        sum_ += current;
        window_[position_] = current;
        position_ = (position_ + 1) % window_.size();
        if (!previous_speech_ && sum_ >= sil_to_speech_frames_) {
            previous_speech_ = true;
            return AudioChangeState::Sil2Speech;
        }
        if (previous_speech_ && sum_ <= speech_to_sil_frames_) {
            previous_speech_ = false;
            return AudioChangeState::Speech2Sil;
        }
        return previous_speech_ ? AudioChangeState::Speech2Speech
                                : AudioChangeState::Sil2Sil;
    }

    int size() const { return (int)window_.size(); }

    void reset() {
        std::fill(window_.begin(), window_.end(), 0);
        position_ = 0;
        sum_ = 0;
        previous_speech_ = false;
    }

private:
    std::vector<int> window_;
    size_t position_ = 0;
    int sum_ = 0;
    bool previous_speech_ = false;
    int sil_to_speech_frames_;
    int speech_to_sil_frames_;
};

class Endpoint {
public:
    Endpoint(const FunASRStreamingVadConfig & config, int frame_ms)
        : config_(config), frame_ms_(std::max(1, frame_ms)),
          max_segment_frames_(std::max(1, config.max_segment_ms/frame_ms_)),
          window_(frame_ms_) {
        update_end_silence_threshold(0);
    }

    bool push(const std::vector<float> & scores, int output_dim,
              const std::vector<float> & decibels,
              std::vector<FunASRStreamingVadEvent> & events) {
        if (output_dim <= 0 || scores.size() % (size_t)output_dim != 0) return false;
        const size_t frames = scores.size()/(size_t)output_dim;
        if (frames != decibels.size()) return false;
        for (size_t i=0; i<frames; ++i) {
            const int absolute_frame = frame_count_++;
            const FrameState frame_state = classify(scores.data()+i*(size_t)output_dim,
                                                     decibels[i]);
            const AudioChangeState change = window_.detect(frame_state);
            detect_one_frame(change, absolute_frame, events);
        }
        return true;
    }

    void finish(std::vector<FunASRStreamingVadEvent> & events) {
        // frame_count_ is the exclusive end frame, matching Python's final-frame
        // OnVoiceEnd(cur_frame) followed by one frame of output.
        if (state_ == VadState::InSpeechSegment) emit_end_exclusive(frame_count_, events);
        reset_detection();
    }

    bool active() const { return state_ == VadState::InSpeechSegment; }
    int absolute_frames() const { return frame_count_; }
    int current_max_end_silence_ms() const { return current_end_silence_ms_; }

private:
    FrameState classify(const float * score, float decibel) {
        const float silence = std::max(0.0f, std::min(1.0f, score[0]));
        const float speech = 1.0f-silence;
        const float snr = decibel-noise_average_decibel_;
        const float weighted_noise = std::pow(silence, speech_to_noise_ratio_);
        if (decibel < config_.decibel_threshold) return FrameState::Silence;
        if (speech >= weighted_noise+config_.speech_noise_threshold) {
            return snr >= config_.snr_threshold ? FrameState::Speech
                                                : FrameState::Silence;
        }
        if (noise_average_decibel_ < -99.9f) noise_average_decibel_ = decibel;
        else noise_average_decibel_ =
            (decibel+noise_average_decibel_*(noise_average_frames_-1))/noise_average_frames_;
        return FrameState::Silence;
    }

    void detect_one_frame(AudioChangeState change, int frame,
                          std::vector<FunASRStreamingVadEvent> & events) {
        if (state_ == VadState::StartPointNotDetected) {
            if (change != AudioChangeState::Sil2Speech) return;
            const int latency_frames = window_.size()+200/frame_ms_;
            start_frame_ = std::max(last_end_frame_, std::max(0, frame-latency_frames));
            state_ = VadState::InSpeechSegment;
            continuous_silence_frames_ = 0;
            update_end_silence_threshold((frame-start_frame_+1)*frame_ms_);
            events.push_back({FunASRStreamingVadEvent::SpeechStart,
                              start_frame_*frame_ms_, -1});
            return;
        }

        const int segment_ms = (frame-start_frame_+1)*frame_ms_;
        update_end_silence_threshold(segment_ms);
        if (change == AudioChangeState::Sil2Sil) {
            ++continuous_silence_frames_;
            if (continuous_silence_frames_ >= max_end_silence_frames_) {
                int lookback = max_end_silence_frames_-100/frame_ms_-1;
                lookback = std::max(0, lookback);
                emit_end_inclusive(frame-lookback, events);
                return;
            }
        } else {
            // Speech2Sil marks a window transition. It does not close the segment;
            // endpointing starts counting only subsequent stable-silence frames.
            continuous_silence_frames_ = 0;
        }

        if (frame-start_frame_+1 > max_segment_frames_) {
            emit_end_inclusive(frame, events);
        }
    }

    void update_end_silence_threshold(int segment_ms) {
        (void) segment_ms;
        const int silence_ms = std::max(1, config_.max_end_silence_ms);
        current_end_silence_ms_ = silence_ms;
        max_end_silence_frames_ = std::max(0, (silence_ms-150)/frame_ms_);
    }

    void emit_end_inclusive(int end_frame,
                            std::vector<FunASRStreamingVadEvent> & events) {
        emit_end_exclusive(std::min(frame_count_, end_frame+1), events);
    }

    void emit_end_exclusive(int end_frame,
                            std::vector<FunASRStreamingVadEvent> & events) {
        end_frame = std::max(start_frame_+1, std::min(end_frame, frame_count_));
        if (end_frame > last_end_frame_) {
            events.push_back({FunASRStreamingVadEvent::SpeechEnd, -1,
                              end_frame*frame_ms_});
            last_end_frame_ = end_frame;
        }
        reset_detection();
    }

    void reset_detection() {
        // Keep frame_count_, last_end_frame_, frontend/FSMN state and noise history:
        // a new utterance continues on the same absolute session timeline.
        window_.reset();
        state_ = VadState::StartPointNotDetected;
        start_frame_ = -1;
        continuous_silence_frames_ = 0;
        update_end_silence_threshold(0);
    }

    FunASRStreamingVadConfig config_;
    int frame_ms_ = 10;
    int max_segment_frames_ = 1;
    WindowDetector window_;
    VadState state_ = VadState::StartPointNotDetected;
    int frame_count_ = 0;
    int start_frame_ = -1;
    int last_end_frame_ = 0;
    int continuous_silence_frames_ = 0;
    int max_end_silence_frames_ = 1;
    int current_end_silence_ms_ = 800;
    float noise_average_decibel_ = -100.0f;
    const float speech_to_noise_ratio_ = 1.0f;
    const int noise_average_frames_ = 100;
};
}

class FunASRStreamingVad {
public:
    bool init(const std::string & path, int max_segment_ms, int threads,
              float speech_noise_threshold) {
        FunASRStreamingVadConfig config;
        config.max_segment_ms = max_segment_ms;
        config.speech_noise_threshold = speech_noise_threshold;
        return init(path, threads, config);
    }

    bool init(const std::string & path, int threads,
              const FunASRStreamingVadConfig & config) {
        if (config.chunk_ms <= 0 || config.max_segment_ms <= 0 ||
            config.max_end_silence_ms <= 0 || !model_.load(path, threads)) return false;
        config_ = config;
        chunk_samples_ = (size_t)config.chunk_ms*funasr_vad_impl::FS/1000;
        if (!chunk_samples_) return false;
        front_.reset(new funasr_vad_stream_impl::Frontend(model_.lm(), model_.ln()));
        end_.reset(new funasr_vad_stream_impl::Endpoint(config_, model_.ln()*10));
        return true;
    }

    bool accept(const float * samples, size_t count, bool final,
                std::vector<FunASRStreamingVadEvent> & events) {
        events.clear();
        if (finalized_ || (!samples && count)) return false;
        if (count) pending_samples_.insert(pending_samples_.end(), samples, samples+count);
        size_t consumed = 0;
        while (pending_samples_.size()-consumed >= chunk_samples_) {
            if (!process(pending_samples_.data()+consumed, chunk_samples_, false, events)) return false;
            consumed += chunk_samples_;
        }
        if (consumed) pending_samples_.erase(pending_samples_.begin(),
                                             pending_samples_.begin()+(ptrdiff_t)consumed);
        if (final) {
            if (!process(pending_samples_.empty()?nullptr:pending_samples_.data(),
                         pending_samples_.size(), true, events)) return false;
            pending_samples_.clear();
            end_->finish(events);
            finalized_ = true;
        }
        return true;
    }

private:
    bool process(const float * samples, size_t count, bool final,
                 std::vector<FunASRStreamingVadEvent> & events) {
        auto output = front_->push(samples, count, final);
        if (!output.frames) return true;
        const size_t frame_shift = (size_t)model_.ln()*funasr_vad_impl::SHIFT;
        const size_t expected_samples = (size_t)(output.frames-1)*frame_shift+
                                        funasr_vad_impl::WINLEN;
        if (output.waveform.size() != expected_samples) return false;
        std::vector<float> decibels((size_t)output.frames);
        for (int frame=0; frame<output.frames; ++frame) {
            const float * waveform = output.waveform.data()+(size_t)frame*frame_shift;
            double energy = 0.0;
            for (int i=0; i<funasr_vad_impl::WINLEN; ++i)
                energy += (double)waveform[i]*waveform[i];
            decibels[(size_t)frame] = 10.0f*std::log10((float)energy+0.000001f);
        }
        std::vector<float> scores;
        if (!model_.infer(std::move(output.features), output.frames, scores)) return false;
        return end_->push(scores, model_.od(), decibels, events);
    }

    FunASRStreamingVadConfig config_;
    funasr_vad_stream_impl::Model model_;
    std::unique_ptr<funasr_vad_stream_impl::Frontend> front_;
    std::unique_ptr<funasr_vad_stream_impl::Endpoint> end_;
    std::vector<float> pending_samples_;
    size_t chunk_samples_ = 0;
    bool finalized_ = false;
};
