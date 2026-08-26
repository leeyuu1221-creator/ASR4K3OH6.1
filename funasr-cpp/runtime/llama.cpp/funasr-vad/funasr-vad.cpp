// funasr-vad: FSMN-VAD on ggml. WAV(any fmt/rate) -> speech segments [start_ms,end_ms].
// Front end + FSMN encoder validated bit-exact vs PyTorch; state machine reproduces
// E2EVadModel segmentation to within 1 frame (10ms) of fsmn-vad.generate on the 184-clip set.
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static void usage(const char * argv0) {
  fprintf(stderr,
          "usage: %s -m fsmn-vad.gguf -a audio.wav [options]\n"
          "  -m, --model PATH    FSMN-VAD model in GGUF format\n"
          "  -a, --audio PATH    input audio file\n"
          "  -t, --threads N     number of CPU inference threads (default: 8)\n"
          "      --max-segment-ms N  maximum speech segment length (default: 30000)\n"
          "      --speech-noise-thres F  speech/noise threshold (default: 0.5)\n"
          "      --max-end-silence-ms N  fixed endpoint silence (default: 800)\n"
          "      --start-lookback-ms N  speech start lookback (default: 400)\n"
          "      --end-lookahead-ms N  speech end lookahead (default: 100)\n"
          "      --decision-window-ms N  decision window (default: 200)\n"
          "      --speech-start-frames N  frames to enter speech (default: 15)\n"
          "      --speech-end-frames N  frames to leave speech (default: 15)\n"
          "  -h, --help          show this help message\n",
          argv0);
}

int main(int argc,char**argv){
  std::string gp,wp;
  int n_threads=8;
  int max_seg_ms=30000;
  float speech_noise_thres=0.5f;
  int max_end_silence_ms=800;
  int start_lookback_ms=400;
  int end_lookahead_ms=100;
  int decision_window_ms=200;
  int speech_start_frames=15;
  int speech_end_frames=15;
  for(int i=1;i<argc;i++){
    const char *arg=argv[i];
    auto value=[&](const char *name)->const char *{
      if(i+1>=argc){fprintf(stderr,"%s requires a value\n",name);return nullptr;}
      return argv[++i];
    };
    if(!strcmp(arg,"-h")||!strcmp(arg,"--help")){usage(argv[0]);return 0;}
    if(!strcmp(arg,"-m")||!strcmp(arg,"--model")){const char *v=value(arg);if(!v)return 2;gp=v;}
    else if(!strcmp(arg,"-a")||!strcmp(arg,"--audio")){const char *v=value(arg);if(!v)return 2;wp=v;}
    else if(!strcmp(arg,"-t")||!strcmp(arg,"--threads")){const char *v=value(arg);if(!v)return 2;n_threads=atoi(v);}
    else if(!strcmp(arg,"--max-segment-ms")){const char *v=value(arg);if(!v)return 2;max_seg_ms=atoi(v);}
    else if(!strcmp(arg,"--speech-noise-thres")){const char *v=value(arg);if(!v)return 2;speech_noise_thres=strtof(v,nullptr);}
    else if(!strcmp(arg,"--max-end-silence-ms")){const char *v=value(arg);if(!v)return 2;max_end_silence_ms=atoi(v);}
    else if(!strcmp(arg,"--start-lookback-ms")){const char *v=value(arg);if(!v)return 2;start_lookback_ms=atoi(v);}
    else if(!strcmp(arg,"--end-lookahead-ms")){const char *v=value(arg);if(!v)return 2;end_lookahead_ms=atoi(v);}
    else if(!strcmp(arg,"--decision-window-ms")){const char *v=value(arg);if(!v)return 2;decision_window_ms=atoi(v);}
    else if(!strcmp(arg,"--speech-start-frames")){const char *v=value(arg);if(!v)return 2;speech_start_frames=atoi(v);}
    else if(!strcmp(arg,"--speech-end-frames")){const char *v=value(arg);if(!v)return 2;speech_end_frames=atoi(v);}
    else {fprintf(stderr,"unknown option: %s\n",arg);usage(argv[0]);return 2;}
  }
  if(gp.empty()||wp.empty()||n_threads<=0||max_seg_ms<=0||
     speech_noise_thres<0.0f||speech_noise_thres>1.0f||max_end_silence_ms<=0||
     start_lookback_ms<0||end_lookahead_ms<0||decision_window_ms<=0||
     speech_start_frames<=0||speech_end_frames<=0){
    if(n_threads<=0)fprintf(stderr,"threads must be greater than 0\n");
    if(max_seg_ms<=0)fprintf(stderr,"max-segment-ms must be greater than 0\n");
    if(speech_noise_thres<0.0f||speech_noise_thres>1.0f)fprintf(stderr,"speech-noise-thres must be in [0,1]\n");
    if(max_end_silence_ms<=0)fprintf(stderr,"max-end-silence-ms must be greater than 0\n");
    if(start_lookback_ms<0||end_lookahead_ms<0)fprintf(stderr,"time parameters must be non-negative\n");
    if(decision_window_ms<=0||speech_start_frames<=0||speech_end_frames<=0)fprintf(stderr,"decision parameters must be greater than 0\n");
    usage(argv[0]);return 2;
  }
  std::vector<float> wav; if(!funasr_load_audio_16k_mono(wp.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
  std::vector<std::pair<int,int>> segs;
  const auto infer_begin=std::chrono::steady_clock::now();
  FunASRVadConfig vad_config;
  vad_config.max_end_silence_ms = max_end_silence_ms;
  vad_config.start_lookback_ms = start_lookback_ms;
  vad_config.end_lookahead_ms = end_lookahead_ms;
  vad_config.decision_window_ms = decision_window_ms;
  vad_config.speech_start_frames = speech_start_frames;
  vad_config.speech_end_frames = speech_end_frames;
  if(!funasr_vad_segments(gp,wav,max_seg_ms,segs,n_threads,speech_noise_thres,nullptr,&vad_config)){fprintf(stderr,"vad failed\n");return 1;}
  const auto infer_end=std::chrono::steady_clock::now();
  const double inference_ms=std::chrono::duration<double,std::milli>(infer_end-infer_begin).count();
  const double audio_ms=(double)wav.size()*1000.0/16000.0;
  const double rtf=audio_ms>0.0?inference_ms/audio_ms:0.0;
  for(auto&s:segs) printf("%d %d\n", s.first, s.second);
  fprintf(stderr,"[vad] %zu segments (max_seg=%dms, threshold=%.3f, end_silence=%dms, threads=%d, inference_ms=%.3f, rtf=%.6f)\n",
          segs.size(),max_seg_ms,speech_noise_thres,max_end_silence_ms,n_threads,inference_ms,rtf);
  return 0;
}
