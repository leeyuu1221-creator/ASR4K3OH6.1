# sherpa-onnx：SpacemiT K3 OpenHarmony 构建指南

本目录用于构建 RISC-V64、musl 用户态的 SpacemiT K3/OpenHarmony 板端版本。
构建链路为：sherpa-onnx + SpacemiT ONNX Runtime + SpacemiT Execution Provider。

## 目录说明

```text
source/                         sherpa-onnx 源码
toolchain/                      CMake 交叉编译配置
dependencies/spacemit-ort/     ONNX Runtime、EP 头文件和动态库
dependencies/alsa/              可选的 RISC-V64 musl ALSA 运行库和头文件
```

仓库不包含约 2.1 GB 的完整交叉工具链。请从 SpaceMIT 工具链发布包获取
`spacemit-toolchain-linux-musl-x86_64-oh-20260630`，并放在本目录外，或通过
`RISCV_ROOT_PATH` 指定路径。GitHub 单文件大小限制也不适合提交该工具链压缩包。

## 环境准备

```bash
export SHERPA_ROOT="$PWD"
export TOOLCHAIN="$SHERPA_ROOT/../spacemit-toolchain-linux-musl-x86_64-oh-20260630"
export ORT_ROOT="$SHERPA_ROOT/dependencies/spacemit-ort"
export ALSA_ROOT="$SHERPA_ROOT/dependencies/alsa"
export BUILD_DIR="$SHERPA_ROOT/build-riscv64-ohos-musl"
export INSTALL_DIR="$SHERPA_ROOT/install-riscv64-ohos-musl"
```

工具链至少需要提供 `clang`、`clang++`、`llvm-strip`、RISC-V64 musl sysroot、
libc++ 和 libc++abi。

首次配置时，CMake 还会获取 kaldi-native-fbank、kaldi-decoder、kaldifst、
openfst、kissfft、Eigen、simple-sentencepiece、nlohmann-json 等源码依赖。
离线构建时，应预先准备这些依赖并设置对应的
`FETCHCONTENT_SOURCE_DIR_<NAME>`。

## 配置、编译和安装

```bash
cmake -S "$SHERPA_ROOT/source" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$SHERPA_ROOT/toolchain/oh_riscv64.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_CXX_FLAGS="-U__OHOS__ -I$ALSA_ROOT/include" \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_SPACEMIT=ON \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF \
  -DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME="$ORT_ROOT" \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON \
  -DSHERPA_ONNX_ENABLE_C_API=ON

cmake --build "$BUILD_DIR" --parallel "$(nproc)"
cmake --install "$BUILD_DIR" --strip
```

`-U__OHOS__` 用于当前独立 musl sysroot 不含完整 OpenHarmony `hilog` 和
`rawfile` 头文件的情况。如果换成完整 OpenHarmony Native SDK，应根据 SDK
环境重新评估该选项。

启用 ALSA 麦克风程序时，配置前设置：

```bash
export SHERPA_ONNX_ALSA_LIB_DIR="$ALSA_ROOT/lib"
```

并确保 `dependencies/alsa/include/alsa/asoundlib.h` 和
`dependencies/alsa/lib/libasound.so.2` 存在。

## 板端部署

至少复制以下内容到 K3：

```text
install-riscv64-ohos-musl/bin/sherpa-onnx*
install-riscv64-ohos-musl/lib/libsherpa-onnx-c-api.so
install-riscv64-ohos-musl/lib/libsherpa-onnx-cxx-api.so
dependencies/spacemit-ort/lib/libonnxruntime.so*
dependencies/spacemit-ort/lib/libonnxruntime_providers_shared.so
dependencies/spacemit-ort/lib/libspacemit_ep.so*
```

运行时：

```bash
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
export ALSA_CONFIG_PATH="$PWD/share/alsa/alsa.conf"  # 使用 ALSA 时
./bin/sherpa-onnx --provider=spacemit
```

## 产物检查

```bash
file install-riscv64-ohos-musl/bin/sherpa-onnx
readelf -l install-riscv64-ohos-musl/bin/sherpa-onnx | grep interpreter
readelf -d install-riscv64-ohos-musl/bin/sherpa-onnx | grep NEEDED
```

应看到 ELF64 RISC-V、`lp64d`/double-float ABI、解释器
`/lib/ld-musl-riscv64.so.1`，并依赖 `libonnxruntime.so.1` 和
`libspacemit_ep.so.2`。

## 版本提示

现有板端产物目录名为 `v1.13.3`；仓库中的源码可能是更新的提交。若需要完全
复现现有二进制，应切换到对应的 sherpa-onnx v1.13.3 源码版本后再构建。
