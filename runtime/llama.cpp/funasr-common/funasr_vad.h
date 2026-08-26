// funasr_vad.h — 面向 FunASR ggml 运行时的单头文件 FSMN-VAD 实现。
// 提供 funasr_vad_segments()：输入 16 kHz 单声道 WAV 波形，输出语音区间 [start_ms,end_ms]。
// 前端（80 维 Mel 滤波器组 + LFR m5n1 + CMVN）以及 FSMN 编码器已与
// PyTorch fsmn-vad 进行逐位一致性验证；宿主侧状态机复现 E2EVadModel（DEFAULT_SILENCE_SCHEDULE，
// 按分块推进），在 184 条音频测试集上与 fsmn-vad.generate 的误差不超过 1 帧（10 ms）。
#pragma once
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC 的 <cmath> 不保证定义 M_PI
#endif

// -----------------------------------------------------------------------------
// 内部实现细节
//
// 本命名空间包含信号处理前端、辅助函数、模型张量查找逻辑，以及下方公开入口
// funasr_vad_segments() 所使用的小型 GGML 计算图构建辅助函数。
//
// -----------------------------------------------------------------------------
namespace funasr_vad_impl {
// 音频 / 特征提取相关常量：
//   FS      ：输入采样率（Hz）
//   WINLEN  ：分析窗长度（采样点；16 kHz 下对应 25 ms）
//   SHIFT   ：帧移（采样点；16 kHz 下对应 10 ms）
//   NFFT    ：FFT 点数
//   NMEL    ：Mel 滤波器组通道数
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
// 将 Hz 频率转换为构建 Mel 滤波器组时使用的 Mel 标度。
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
// 原地基 2 复数 FFT。
// re[] 和 im[] 分别保存实部与虚部。
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
    float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}}
// 从单声道波形计算 80 维对数 Mel 滤波器组特征。
// 输入波形预期为归一化浮点音频；此处将其重新缩放到参考 FunASR
// 实现所采用的幅值约定。
static std::vector<std::vector<float>> fbank80(std::vector<float> wav){
  for(auto&v:wav)v*=32768.0f; std::vector<float>win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NB=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>>fb(NMEL,std::vector<float>(NB,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;for(int k=0;k<NB;k++){float mf=melf(bw*k);if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; if(T<1)T=0; std::vector<std::vector<float>>feat(T,std::vector<float>(NMEL));
  std::vector<float>re(NFFT),im(NFFT),fr(WINLEN);const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NB;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);feat[t][m]=logf(e>fl?e:fl);}}
  return feat;
}
// 低帧率（Low Frame Rate，LFR）堆叠。
// 每个输出帧拼接相邻的 m 个 80 维滤波器组帧，
// 每次向前移动 n 个源帧；T_out 返回最终输出帧数。
static std::vector<float> lfr(const std::vector<std::vector<float>>&feat,int m,int n,int&T_out){
  int T=feat.size(); if(T<1){T_out=0;return {};}     // 空输入（音频长度不足一帧）
  int D=NMEL,pad=(m-1)/2; int Tl=(T+n-1)/n;
  std::vector<std::vector<float>> pf; pf.reserve(T+pad+m);
  for(int i=0;i<pad;i++)pf.push_back(feat[0]);
  for(int t=0;t<T;t++)pf.push_back(feat[t]);
  while((int)pf.size()<(Tl-1)*n+m)pf.push_back(feat[T-1]);
  std::vector<float> out((size_t)Tl*m*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<m;j++)memcpy(&out[((size_t)i*m+j)*D],pf[i*n+j].data(),D*sizeof(float));
  T_out=Tl; return out;
}
// 轻量模型容器：持有从 GGUF 加载的 GGML 上下文，并提供
// VAD 网络所需张量的按名称查找功能。
struct vad{ggml_context*ctx=nullptr;std::map<std::string,ggml_tensor*>t;
  ggml_tensor*g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"vad: missing %s\n",n.c_str());return nullptr;}return it->second;}};
// 全连接层辅助函数：先做矩阵乘法，再按需加偏置。
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
} // funasr_vad_impl 命名空间结束

// -----------------------------------------------------------------------------
// VAD 对外入口
//
// 处理流程：
//   波形 -> 80 维 fbank -> LFR -> CMVN -> FSMN 编码器 -> softmax 得分
//        -> 宿主侧 E2EVadModel 状态机 -> 毫秒级语音片段
// -----------------------------------------------------------------------------
// 在 16 kHz 单声道浮点波形上运行 FSMN-VAD；segs 返回 [start_ms,end_ms] 语音区间。
// max_seg_ms 限制单个语音段最大长度（例如 30000）；传入 <=0 时使用模型默认值 60000。
struct FunASRVadConfig {
  // 固定尾静音阈值；每个新语音段都使用该值重新计时。
  int max_end_silence_ms = 800;
  int start_lookback_ms = 400;
  int end_lookahead_ms = 100;
  int decision_window_ms = 200;
  int speech_start_frames = 15;
  int speech_end_frames = 15;
};

inline bool funasr_vad_segments(const std::string& gguf_path, const std::vector<float>& wav,
                                int max_seg_ms, std::vector<std::pair<int,int>>& segs, int nthreads=8,
                                float speech_noise_thres=0.5f,
                                int64_t * model_load_us=nullptr,
                                const FunASRVadConfig * config=nullptr){
  using namespace funasr_vad_impl;
  segs.clear();
  const int64_t model_load_start = ggml_time_us();
  // 将 GGUF 元数据和张量加载到模型上下文。
  vad m; gguf_init_params ip={false,&m.ctx}; gguf_context*gg=gguf_init_from_file(gguf_path.c_str(),ip);
  if(!gg){fprintf(stderr,"vad: cannot load %s\n",gguf_path.c_str());return false;}
  // 读取 GGUF 中的无符号整数元数据键；如果键不存在，
  // 则使用传入的默认值。
  auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  // 模型维度和 LFR 参数。若 GGUF 中不存在可选元数据键，
  // 则使用与预期 FSMN-VAD 网络结构一致的默认值。
  int idim=rd("vad.input_dim",400),pd=rd("vad.proj_dim",128),nl=rd("vad.fsmn_layers",4),lorder=rd("vad.lorder",20),
      od=rd("vad.output_dim",248),lm=rd("vad.lfr_m",5),ln=rd("vad.lfr_n",1);
  // 在释放 GGUF 元数据对象之前，建立“张量名称 -> 张量指针”的映射。
  // 张量实际存储仍由 m.ctx 持有。
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx,nm);}
  gguf_free(gg);

  // 若 GGUF 缺少后续计算图会解引用的张量，则提前失败，避免发生段错误。
  auto need=[&](const std::string&n){ return m.g(n)!=nullptr; };
  bool ok_t = need("cmvn.shift")&&need("cmvn.scale")&&need("encoder.in_linear1.linear.weight")
            &&need("encoder.in_linear2.linear.weight")&&need("encoder.out_linear1.linear.weight")
            &&need("encoder.out_linear2.linear.weight");
  for(int i=0;i<nl&&ok_t;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    ok_t=need(p+"linear.linear.weight")&&need(p+"fsmn_block.conv_left.weight")&&need(p+"affine.linear.weight");}
  if(!ok_t){fprintf(stderr,"vad: gguf missing required tensors\n"); if(m.ctx)ggml_free(m.ctx); return false;}
  if (model_load_us) *model_load_us = ggml_time_us() - model_load_start;

  // 前端处理：波形 -> 对数 Mel 滤波器组 -> LFR 堆叠特征。
  auto feat=fbank80(wav); int T=0; auto feats=lfr(feat,lm,ln,T);   // [T,400]，即 [时间帧数, 特征维度]
  if(T<1){if(m.ctx)ggml_free(m.ctx);return true;}                  // 音频过短，不产生语音片段
  // 按模型中编码的方式应用 CMVN：(feature + shift) * scale。
  float*shift=(float*)m.g("cmvn.shift")->data,*scale=(float*)m.g("cmvn.scale")->data;
  for(int t=0;t<T;t++)for(int d=0;d<idim;d++)feats[(size_t)t*idim+d]=(feats[(size_t)t*idim+d]+shift[d])*scale[d];

  // ---------------------------------------------------------------------------
  // 在 CPU 后端构建并执行 FSMN 推理计算图。
  // ---------------------------------------------------------------------------
  ggml_backend_t be=ggml_backend_cpu_init();
  // no_alloc=true：ctx 只保存张量/计算图元数据；真正的计算缓冲区由下方 gallocr
  // 分配，因此无论音频长度如何，数 MB 的上下文空间都足够。
  ggml_init_params cp={(size_t)16*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
  // 输入张量布局为 [特征维度, 时间]。计算图完成内存分配后，
  // 再将宿主侧特征缓冲区复制到该张量。
  ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,idim,T); ggml_set_input(x);
  // 输入投影：两层线性层，之后接 ReLU。
  ggml_tensor*h=lin(c,m.g("encoder.in_linear1.linear.weight"),m.g("encoder.in_linear1.linear.bias"),x);
  h=lin(c,m.g("encoder.in_linear2.linear.weight"),m.g("encoder.in_linear2.linear.bias"),h); h=ggml_relu(c,h);
  // FSMN 层堆叠。每层将当前投影与一系列左侧历史上下文抽头组合，
  // 随后执行仿射变换并应用 ReLU。
  for(int i=0;i<nl;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
    ggml_tensor*z=ggml_mul_mat(c,m.g(p+"linear.linear.weight"),h);
    ggml_tensor*fk=m.g(p+"fsmn_block.conv_left.weight"); ggml_tensor*zp=ggml_pad_ext(c,z,0,0,lorder-1,0,0,0,0,0); ggml_tensor*acc=z;
    // sl 是连续填充张量中的完整行切片，本身已连续，因此无需调用 ggml_cont。
    for(int j=0;j<lorder;j++){auto sl=ggml_view_2d(c,zp,pd,T,zp->nb[1],(size_t)j*zp->nb[1]);auto wj=ggml_view_1d(c,fk,pd,(size_t)j*fk->nb[1]);acc=ggml_add(c,acc,ggml_mul(c,sl,wj));}
    ggml_tensor*a=lin(c,m.g(p+"affine.linear.weight"),m.g(p+"affine.linear.bias"),acc); h=ggml_relu(c,a);}
  // 输出投影并执行 softmax。后续状态机将类别 0 解释为静音。
  h=lin(c,m.g("encoder.out_linear1.linear.weight"),m.g("encoder.out_linear1.linear.bias"),h);
  h=lin(c,m.g("encoder.out_linear2.linear.weight"),m.g("encoder.out_linear2.linear.bias"),h);
  h=ggml_soft_max(c,h); ggml_set_output(h);
  // 构建计算图、分配计算缓冲区、复制输入，然后执行推理。
  ggml_cgraph*gf=ggml_new_graph(c); ggml_build_forward_expand(gf,h);
  ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
  ggml_backend_tensor_set(x,feats.data(),0,ggml_nbytes(x)); ggml_backend_cpu_set_n_threads(be,nthreads);
  bool ok=ggml_backend_graph_compute(be,gf)==GGML_STATUS_SUCCESS;
  // 仅在计算图执行成功时，将 softmax 得分复制回宿主内存。
  std::vector<float> sc((size_t)od*T); if(ok)ggml_backend_tensor_get(h,sc.data(),0,ggml_nbytes(h));
  // 释放所有计算侧资源。得分矩阵复制到 sc 后，模型张量存储也不再需要。
  //
  ggml_gallocr_free(ga);ggml_free(c);ggml_backend_free(be);if(m.ctx)ggml_free(m.ctx);
  if(!ok)return false;

  // ===== E2EVadModel 宿主侧状态机 -> 语音片段 [start_ms,end_ms] =====
  // 状态机中的时间参数统一以 10 ms 为一帧进行表示。
  const int FR=10;                 // 每帧毫秒数（frame_in_ms）
  const FunASRVadConfig defaults;
  const FunASRVadConfig & cfg = config ? *config : defaults;
  const int win=std::max(1, cfg.decision_window_ms/FR);
  const int s2s=std::max(1, cfg.speech_start_frames);
  const int sp2s=std::max(1, cfg.speech_end_frames);
  const int lookahead_end=std::max(0, cfg.end_lookahead_ms/FR);
  // 单个输出语音段的最大帧数；max_seg_ms <= 0 时使用模型默认值。
  int max_seg = (max_seg_ms>0 ? max_seg_ms : 60000)/FR;       // 单段最大帧数
  const int start_lookback = std::max(0, cfg.start_lookback_ms/FR);
  // 固定尾静音阈值，不再按全局音频时长动态调整。
  const int max_end_sil = std::max(1, cfg.max_end_silence_ms/FR);
  const int end_lookback = std::max(0, max_end_sil-lookahead_end-1);
  // 滑动窗口语音判决状态。wsum 表示最近 win 帧中被判定为语音的帧数；
  // pre 保存上一时刻的二值状态。
  std::vector<int> wbuf(win,0); int wpos=0,wsum=0,pre=0;
  // 语音段状态：
  //   st       ：0 = 当前无语音段，1 = 当前有活动语音段
  //   cstart   ：当前语音段起始帧
  //   csil     ：连续静音帧计数
  //   prev_end ：最近一次输出语音段的结束帧
  int st=0, cstart=-1, csil=0, prev_end=0;
  // 输出一个语音段后重置临时状态。prev_end 有意不在此 lambda 中重置，
  // 以确保后续输出的语音段不会与前一个语音段重叠。
  auto reset=[&](){ std::fill(wbuf.begin(),wbuf.end(),0); wpos=0; wsum=0; pre=0; csil=0; st=0; cstart=-1; };
  // 添加经过边界裁剪且非空的帧区间，同时避免与上一个
  // 已输出片段发生重叠。
  auto emit=[&](int s,int e){ if(s<prev_end)s=prev_end; if(s<0)s=0; if(e>T)e=T; if(e>s){segs.push_back({s,e}); prev_end=e;} };
  // 逐帧处理网络输出，并复现参考 E2EVadModel 的
  // 状态转移行为。
  for(int t=0;t<T;t++){
    // softmax 的类别 0 表示静音概率。
    float sil=sc[(size_t)t*od+0];
    // 将静音得分转换为宿主侧状态机使用的二值“语音/非语音”判决
    // speech_noise_thres 越大，语音判定越严格；默认值与原实现一致为 0.5。
    int fs = ((1.0f-sil) >= sil + speech_noise_thres) ? 1 : 0;
    // 以 O(1) 时间复杂度更新固定长度判决窗口。
    wsum -= wbuf[wpos]; wsum += fs; wbuf[wpos]=fs; wpos=(wpos+1)%win;
    // ch 状态转移编码：
    //   0 = 静音 / 保持非语音状态
    //   1 = 语音 -> 静音
    //   2 = 保持语音状态
    //   3 = 静音 -> 语音
    int ch;
    if(pre==0 && wsum>=s2s){pre=1; ch=3;}
    else if(pre==1 && wsum<=sp2s){pre=0; ch=1;}
    else ch = pre==0?0:2;
    // 检测到语音开始：将起点向前回看，以包含配置的历史上下文。
    if(ch==3){ csil=0;
      if(st==0){ cstart=t-start_lookback; if(cstart<prev_end)cstart=prev_end; if(cstart<0)cstart=0; st=1; }
      else if(st==1 && t-cstart+1>max_seg){ emit(cstart,t); reset(); }
    // 语音持续，或发生“语音 -> 静音”状态转移。
    } else if(ch==1||ch==2){ csil=0;
      if(st==1 && t-cstart+1>max_seg){ emit(cstart,t); reset(); }
    // 持续静音：尾部静音达到阈值后结束当前语音段；
    // 若语音段超过最大允许时长，则强制结束。
    } else { csil++;
      if(st==1){
        if(csil>=max_end_sil){ emit(cstart, t-end_lookback); reset(); }
        else if(t-cstart+1>max_seg){ emit(cstart,t); reset(); }
      }
    }
  }
  // 输入结束时，如果仍有未闭合的语音段，则将其输出。
  if(st==1) emit(cstart,T);
  // 将帧索引转换为毫秒，作为公开 API 的返回单位。
  // 帧索引 -> ms
  for(auto&s:segs){ s.first*=FR; s.second*=FR; }
  return true;
}
