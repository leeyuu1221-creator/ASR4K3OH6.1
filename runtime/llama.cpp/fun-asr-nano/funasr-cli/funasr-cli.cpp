// funasr-cli: end-to-end Fun-ASR-Nano in C++ on the llama.cpp / ggml stack.
//
//   wav(16k mono) -> kaldi fbank -> SAN-M encoder + adaptor (ggml) ->
//   low-frame-rate truncation -> [prefix tokens | audio embeds | suffix tokens]
//   -> Qwen3 LLM (llama.cpp) -> transcription.
//
// This is the whisper.cpp-style single-binary path: no Python at runtime.
//
//   funasr-cli --enc funasr-encoder.gguf -m qwen3-0.6b.gguf -a audio.wav

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <vector>

// any audio (wav/mp3/flac, any rate/channels) -> 16 kHz mono f32, via miniaudio
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
// built-in FSMN-VAD front end (single-binary --vad segmentation)
#include "funasr_vad.h"
#include <utility>

// ======================= kaldi fbank + LFR =======================
static const int FS=16000, WINLEN=400, SHIFT=160, NFFT=512, NMEL=80, LFR_M=7, LFR_N=6;
static const float PREEMPH=0.97f, LOWF=20.0f, HIGHF=8000.0f;
static inline float mel(float f){ return 1127.0f*logf(1.0f+f/700.0f); }
static void fft(std::vector<float>&re,std::vector<float>&im,int n){
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
    for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
        for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){
            float ur=re[i+k],ui=im[i+k];float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
            re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;
            float n2=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=n2;}}}
}
// returns [T x 560] row-major, sets T
static std::vector<float> compute_fbank(std::vector<float> wav, int & T_out) {
    for (auto & v : wav) v *= 32768.0f;
    std::vector<float> win(WINLEN);
    for (int i=0;i<WINLEN;i++) win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT, ml=mel(LOWF), mh=mel(HIGHF), dm=(mh-ml)/(NMEL+1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(NBIN,0.0f));
    for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
        for(int k=0;k<NBIN;k++){float mf=mel(bw*k); if(mf>L&&mf<R) fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
    int N=wav.size(); int T=(N-WINLEN)/SHIFT+1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(NMEL));
    std::vector<float> re(NFFT),im(NFFT),fr(WINLEN);
    const float fl=1.1920929e-07f;
    for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;
        double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
        fft(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
            feat[t][m]=logf(e>fl?e:fl);}}
    // LFR
    const int pad=(LFR_M-1)/2; int T_lfr=(T+LFR_N-1)/LFR_N;
    std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
    for(int i=0;i<pad;i++)pd.push_back(feat[0]);
    for(int t=0;t<T;t++)pd.push_back(feat[t]);
    while((int)pd.size()<(T_lfr-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
    int D=LFR_M*NMEL; std::vector<float> out((size_t)T_lfr*D);
    for(int i=0;i<T_lfr;i++)for(int j=0;j<LFR_M;j++)
        memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
    T_out=T_lfr; return out;
}

// ======================= ggml SAN-M encoder + adaptor =======================
struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,adp_llm=1024,adp_layers=2,adp_head=8; };
struct enc_model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
    ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };
static const float LN_EPS=1e-5f;
static bool load_enc(const char*p, enc_model&m){
    gguf_init_params gp={false,&m.ctx_w}; gguf_context*g=gguf_init_from_file(p,gp); if(!g)return false;
    auto rd=[&](const char*k,int d){int i=gguf_find_key(g,k);return i<0?d:(int)gguf_get_val_u32(g,i);};
    m.c.d_model=rd("funasr.enc.output_size",512); m.c.n_head=rd("funasr.enc.attention_heads",4);
    m.c.num_blocks=rd("funasr.enc.num_blocks",50); m.c.tp_blocks=rd("funasr.enc.tp_blocks",20);
    m.c.kernel=rd("funasr.enc.kernel_size",11); m.c.adp_llm=rd("funasr.adp.llm_dim",1024);
    m.c.adp_layers=rd("funasr.adp.n_layer",2); m.c.adp_head=rd("funasr.adp.attention_heads",8);
    int n=gguf_get_n_tensors(g); for(int i=0;i<n;i++){const char*nm=gguf_get_tensor_name(g,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
    gguf_free(g); return true;
}
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
    ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
    ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
    ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
    ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
    const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
    ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
    for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
        auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
    q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
    ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
    ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T,bool res){
    auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static ggml_tensor* adp_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.adp_llm,H=m.c.adp_head,dk=D/H; auto r=x;
    auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto q=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_q.weight"),m.g(p+"self_attn.linear_q.bias"),h),dk,H,T),0,2,1,3);
    auto k=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_k.weight"),m.g(p+"self_attn.linear_k.bias"),h),dk,H,T),0,2,1,3);
    auto vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_v.weight"),m.g(p+"self_attn.linear_v.bias"),h),dk,H,T),1,2,0,3));
    auto kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    auto o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    x=ggml_add(c,r,lin(c,m.g(p+"self_attn.linear_out.weight"),m.g(p+"self_attn.linear_out.bias"),o)); r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
    double inc=log(10000.0)/(depth/2.0-1.0);
    for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
        x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}
// fbank [T x F] -> adaptor out [T x adp_llm] row-major
static std::vector<float> run_encoder(enc_model&m,std::vector<float> fbank,int T,int F,int&Dout,int nthreads=8){
    float sc=sqrtf((float)m.c.d_model); for(auto&v:fbank)v*=sc; add_posenc(fbank,T,F);
    ggml_backend_t be=ggml_backend_cpu_init();
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    ggml_tensor*inp=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T); ggml_set_input(inp);
    ggml_tensor*x=sanm_layer(c,m,"audio_encoder.encoders0.0.",inp,T,false);
    for(int i=0;i<m.c.num_blocks-1;i++) x=sanm_layer(c,m,"audio_encoder.encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.after_norm.weight"),m.g("audio_encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) x=sanm_layer(c,m,"audio_encoder.tp_encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.tp_norm.weight"),m.g("audio_encoder.tp_norm.bias"));
    x=lin(c,m.g("audio_adaptor.linear1.weight"),m.g("audio_adaptor.linear1.bias"),x); x=ggml_relu(c,x);
    x=lin(c,m.g("audio_adaptor.linear2.weight"),m.g("audio_adaptor.linear2.bias"),x);
    for(int i=0;i<m.c.adp_layers;i++) x=adp_layer(c,m,"audio_adaptor.blocks."+std::to_string(i)+".",x,T);
    ggml_set_output(x);
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,x);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(inp,fbank.data(),0,ggml_nbytes(inp));
    ggml_backend_cpu_set_n_threads(be,nthreads); ggml_backend_graph_compute(be,gf);
    Dout=(int)x->ne[0]; std::vector<float> out((size_t)Dout*T); ggml_backend_tensor_get(x,out.data(),0,ggml_nbytes(x));
    ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be); return out;
}

// ======================= LLM (llama.cpp) =======================
static int decode_batch(llama_context*ctx,int n,llama_token*tok,float*embd,int n_embd,int&n_past,bool last_logits){
    std::vector<llama_pos> pos(n); std::vector<int32_t> nsid(n,1);
    std::vector<llama_seq_id> s0(1,0); std::vector<llama_seq_id*> sid(n); std::vector<int8_t> lg(n,0);
    for(int i=0;i<n;i++){pos[i]=n_past+i;sid[i]=s0.data();}
    if(last_logits) lg[n-1]=1;
    llama_batch b={n,tok,embd,pos.data(),nsid.data(),sid.data(),lg.data()};
    int r=llama_decode(ctx,b); n_past+=n; return r;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    static const char hex[] = "0123456789abcdef";
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
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

static void usage(const char * argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --enc <encoder.gguf> (-m|--model) <model.gguf> -a <audio> [options]\n"
        "\n"
        "Input and offline window options:\n"
        "  -a, --audio <path>       Audio file (decoded to 16 kHz mono)\n"
        "  --chunk <seconds>       Fixed offline chunk length; default: whole file\n"
        "  --segment-ms <ms>       Alias for fixed offline chunk length\n"
        "  --overlap-ms <ms>       Optional overlap between offline chunks (default: 0)\n"
        "\n"
        "VAD options:\n"
        "  --vad <path>             FSMN-VAD model\n"
        "  --vad-maxseg <ms>        Maximum VAD segment length (default: 20000)\n"
        "  --vad-threshold <F>      Speech/noise threshold (default: 0.5)\n"
        "  --vad-max-end-silence-ms <ms>  Fixed endpoint silence (default: 800)\n"
        "\n"
        "Inference and output options:\n"
        "  --language <auto|zh|en|ja>  Language prompt (default: auto)\n"
        "  -n, --max-tokens <N>     Maximum generated tokens (default: 512)\n"
        "  -t, --threads <N>       Encoder and LLM CPU threads (default: 8)\n"
        "  --rep <F>               Repetition penalty (default: 1.05)\n"
        "  --output <text|jsonl>   Output format (default: text)\n"
        "  -h, --help              Show this help\n"
        "\n"
        "JSONL output contains one final event per offline window with\n"
        "begin_ms/end_ms/inference_ms, followed by a done event.\n",
        argv0);
}

struct CliOptions {
    std::string enc, llm, audio, vad;
    std::string language = "auto";
    std::string output = "text";
    int npred = 512;
    int threads = 8;
    float rep = 1.05f;
    float vad_threshold = 0.5f;
    int vad_maxseg_ms = 20000;
    int vad_max_end_silence_ms = 800;
    double chunk_sec = 0;
    int segment_ms = 0;
    int overlap_ms = 0;
};

static bool parse_options(int argc, char ** argv, CliOptions & o) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s requires a value\n", name); return nullptr; }
            return argv[++i];
        };
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); exit(0); }
        else if (!strcmp(a, "--enc")) { const char * v = need(a); if (!v) return false; o.enc = v; }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) { const char * v = need(a); if (!v) return false; o.llm = v; }
        else if (!strcmp(a, "-a") || !strcmp(a, "--audio")) { const char * v = need(a); if (!v) return false; o.audio = v; }
        else if (!strcmp(a, "-n") || !strcmp(a, "--max-tokens")) { const char * v = need(a); if (!v) return false; o.npred = atoi(v); }
        else if (!strcmp(a, "--chunk")) { const char * v = need(a); if (!v) return false; o.chunk_sec = atof(v); }
        else if (!strcmp(a, "--segment-ms")) { const char * v = need(a); if (!v) return false; o.segment_ms = atoi(v); }
        else if (!strcmp(a, "--overlap-ms")) { const char * v = need(a); if (!v) return false; o.overlap_ms = atoi(v); }
        else if (!strcmp(a, "--vad")) { const char * v = need(a); if (!v) return false; o.vad = v; }
        else if (!strcmp(a, "--vad-maxseg")) { const char * v = need(a); if (!v) return false; o.vad_maxseg_ms = atoi(v); }
        else if (!strcmp(a, "--vad-threshold")) { const char * v = need(a); if (!v) return false; o.vad_threshold = strtof(v, nullptr); }
        else if (!strcmp(a, "--vad-max-end-silence-ms")) { const char * v = need(a); if (!v) return false; o.vad_max_end_silence_ms = atoi(v); }
        else if (!strcmp(a, "--language")) { const char * v = need(a); if (!v) return false; o.language = v; }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { const char * v = need(a); if (!v) return false; o.threads = atoi(v); }
        else if (!strcmp(a, "--rep")) { const char * v = need(a); if (!v) return false; o.rep = strtof(v, nullptr); }
        else if (!strcmp(a, "--output")) { const char * v = need(a); if (!v) return false; o.output = v; }
        else { fprintf(stderr, "unknown option: %s\n", a); usage(argv[0]); return false; }
    }
    if (o.enc.empty() || o.llm.empty() || o.audio.empty() || o.npred <= 0 || o.threads <= 0 ||
        o.vad_maxseg_ms <= 0 || o.vad_max_end_silence_ms <= 0 ||
        o.vad_threshold < 0.0f || o.vad_threshold > 1.0f ||
        o.overlap_ms < 0 ||
        (o.segment_ms < 0) || (o.chunk_sec < 0) ||
        (o.language != "auto" && o.language != "zh" && o.language != "en" && o.language != "ja") ||
        (o.output != "text" && o.output != "jsonl")) {
        fprintf(stderr, "invalid or missing arguments\n"); usage(argv[0]); return false;
    }
    if (o.segment_ms > 0) o.chunk_sec = o.segment_ms / 1000.0;
    return true;
}

int main(int argc,char**argv){
    CliOptions opt;
    if (!parse_options(argc, argv, opt)) return 2;

    std::vector<float> wav;
    if(!funasr_load_audio_16k_mono(opt.audio.c_str(),wav)){fprintf(stderr,"failed to read audio\n");return 1;}
    ggml_time_init(); // Required before ggml_time_us() on Windows.

    enc_model em; if(!load_enc(opt.enc.c_str(),em))return 1;
    ggml_backend_load_all();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model*model=llama_model_load_from_file(opt.llm.c_str(),mp); if(!model)return 1;
    const llama_vocab*vocab=llama_model_get_vocab(model);
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
    cp.n_threads=opt.threads; cp.n_threads_batch=opt.threads;
    llama_context*ctx=llama_init_from_model(model,cp);
    if(!ctx){fprintf(stderr,"failed to create llama context\n");llama_model_free(model);return 1;}
    auto sp=llama_sampler_chain_default_params(); llama_sampler*smpl=llama_sampler_chain_init(sp);
    if(opt.rep!=1.0f) llama_sampler_chain_add(smpl,llama_sampler_init_penalties(256,opt.rep,0.0f,0.0f));
    llama_sampler_chain_add(smpl,llama_sampler_init_greedy());

    std::string language_prompt = "语音转写：";
    if (opt.language == "zh") language_prompt = "语音转写成中文：";
    else if (opt.language == "en") language_prompt = "语音转写成英文：";
    else if (opt.language == "ja") language_prompt = "语音转写成日文：";
    const std::string prefix_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + language_prompt;
    const char*suffix="<|im_end|>\n<|im_start|>assistant\n";
    auto tokenize=[&](const char*s){int n=-llama_tokenize(vocab,s,strlen(s),nullptr,0,false,true);
        std::vector<llama_token> v(n); llama_tokenize(vocab,s,strlen(s),v.data(),n,false,true); return v;};
    auto pre=tokenize(prefix_text.c_str()); auto suf=tokenize(suffix);

    // RTF measures processing after all model/runtime initialization is complete.
    // It includes VAD, window construction, feature extraction and ASR inference,
    // but deliberately excludes encoder/LLM loading and context/sampler setup.
    const int64_t processing_start_us=ggml_time_us();

    // Build the list of [offset,len] windows to transcribe (in samples).
    //   --vad : FSMN-VAD speech segments (single-binary front end, replaces fixed chunking)
    //   --chunk sec : fixed-size chunks ; otherwise the whole file in one window
    std::vector<std::pair<int,int>> wins;   // {sample offset, sample len}
    if(!opt.vad.empty()){
        std::vector<std::pair<int,int>> segs; // ms
        FunASRVadConfig vad_config;
        vad_config.max_end_silence_ms = opt.vad_max_end_silence_ms;
        if(!funasr_vad_segments(opt.vad,wav,opt.vad_maxseg_ms,segs,opt.threads,opt.vad_threshold,nullptr,&vad_config)){fprintf(stderr,"vad failed\n");return 1;}
        for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
            if(end>(int)wav.size())end=wav.size(); if(end-off>0) wins.push_back({off,end-off}); }
        fprintf(stderr,"[vad] %zu segments\n",wins.size());
    } else {
        int chunk_n = opt.chunk_sec > 0 ? std::max(1, (int)(opt.chunk_sec*16000)) : (int)wav.size();
        int overlap_n = std::min((int)opt.overlap_ms * 16, std::max(0, chunk_n - 1));
        for(size_t off=0; off<wav.size(); ) {
            int len = (int)std::min((size_t)chunk_n, wav.size()-off);
            wins.push_back({(int)off,len});
            if ((size_t)len >= wav.size()-off) break;
            off += (size_t)(len - overlap_n);
        }
    }
    int64_t total_inference_us = 0;
    int completed = 0;
    for (auto& w : wins) {
        int off = w.first, len = w.second;
        if (len < WINLEN) continue;                    // too short for one frame
        std::vector<float> seg(wav.begin()+off, wav.begin()+off+len);
        const auto infer_begin = std::chrono::steady_clock::now();
        int T=0; auto fbank=compute_fbank(seg,T);
        int D=0; auto adp=run_encoder(em,fbank,T,560,D,opt.threads);
        int ol=1+(T-3+2)/2; ol=1+(ol-3+2)/2; int n_aud=(ol-1)/2+1;

        llama_memory_clear(llama_get_memory(ctx), true);  // fresh context per chunk
        int n_past=0;
        decode_batch(ctx,pre.size(),pre.data(),nullptr,0,n_past,false);
        decode_batch(ctx,n_aud,nullptr,adp.data(),D,n_past,false);
        decode_batch(ctx,suf.size(),suf.data(),nullptr,0,n_past,true);
        llama_token tk=llama_sampler_sample(smpl,ctx,-1);
        std::string window_text;
        for(int i=0;i<opt.npred;i++){
            if(llama_vocab_is_eog(vocab,tk))break;
            char buf[256]; int k=llama_token_to_piece(vocab,tk,buf,sizeof(buf),0,true);
            if(k>0) window_text.append(buf,k);
            decode_batch(ctx,1,&tk,nullptr,0,n_past,true);
            tk=llama_sampler_sample(smpl,ctx,-1);
        }
        const auto infer_end = std::chrono::steady_clock::now();
        const int64_t infer_us = (int64_t)(std::chrono::duration<double,std::micro>(infer_end-infer_begin).count());
        total_inference_us += infer_us; ++completed;
        const int64_t begin_ms = (int64_t)off * 1000 / FS;
        const int64_t end_ms = (int64_t)(off + len) * 1000 / FS;
        if (opt.output == "text") {
            const int64_t begin_seconds = begin_ms / 1000;
            const int64_t end_seconds = end_ms / 1000;
            printf("%02lld:%02lld - %02lld:%02lld %s\n",
                   (long long)(begin_seconds / 60),
                   (long long)(begin_seconds % 60),
                   (long long)(end_seconds / 60),
                   (long long)(end_seconds % 60),
                   window_text.c_str());
            fflush(stdout);
        }
        if (opt.output == "jsonl") {
            printf("{\"type\":\"final\",\"utterance_id\":%d,\"revision\":1,\"begin_ms\":%lld,\"end_ms\":%lld,\"text\":\"%s\",\"inference_ms\":%lld}\n",
                   completed, (long long)begin_ms, (long long)end_ms,
                   json_escape(window_text).c_str(), (long long)(infer_us / 1000));
            fflush(stdout);
        }
    }
    int64_t t2=ggml_time_us();
    const double audio_ms = (double)wav.size() * 1000.0 / FS;
    const double total_ms = (double)(t2-processing_start_us) / 1000.0;
    if (opt.output == "jsonl") {
        printf("{\"type\":\"done\",\"windows\":%d,\"audio_ms\":%lld,\"total_processing_ms\":%lld,\"total_inference_ms\":%lld,\"rtf\":%.6f}\n",
               completed, (long long)audio_ms, (long long)total_ms,
               (long long)(total_inference_us / 1000), audio_ms > 0 ? total_ms / audio_ms : 0.0);
        fflush(stdout);
    }
    fprintf(stderr,"[done] %.2fs ; chunk=%.3fs ; windows=%d ; inference=%.2fs ; rtf=%.6f\n",
            total_ms / 1000.0, opt.chunk_sec, completed,
            (double)total_inference_us / 1e6, audio_ms > 0 ? total_ms / audio_ms : 0.0);
    llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
    if(em.ctx_w) ggml_free(em.ctx_w);
    return 0;
}
